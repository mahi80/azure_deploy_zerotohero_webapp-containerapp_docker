"""
app/predict.py — Model loading, feature engineering, and prediction logic.

The model artifact (joblib file) is loaded ONCE at application startup
and stored in the ModelStore singleton.  Prediction functions read from
this in-memory object — no disk I/O on each request.

Feature engineering here mirrors the training pipeline exactly.
Any change to training features must be reflected here too.
"""
from __future__ import annotations
import math
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from app.config import get_settings
from app.schemas import PredictRequest, PredictResponse, RiskBand

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Scorecard conversion constants ────────────────────────────────────────────
# Standard PDO (Points to Double the Odds) scorecard scaling
# Score = BASE_SCORE + FACTOR * log(odds)
# Odds of approval (1=approved, 0=declined): higher odds → higher score
BASE_SCORE = 600
PDO        = 50     # score increases by 50 when odds double
FACTOR     = PDO / math.log(2)
BASE_ODDS  = 1.0    # odds at the base score

# ── Risk bands ────────────────────────────────────────────────────────────────
RISK_BANDS = [
    (0.80, 1.01, "Very Low",  "820–900", "Excellent credit profile — approve with high limit"),
    (0.60, 0.80, "Low",       "740–819", "Strong profile — standard approval"),
    (0.40, 0.60, "Medium",    "670–739", "Acceptable — approve with moderate limit"),
    (0.20, 0.40, "High",      "580–669", "Elevated risk — conditional approval or lower limit"),
    (0.00, 0.20, "Very High", "300–579", "High default risk — decline or secured card only"),
]

DECISION_THRESHOLD = 0.50   # P(Approved) > this → Approved


def prob_to_score(p: float) -> int:
    """
    Convert approval probability to a FICO-style scorecard integer (600–900).
    Higher probability of approval → higher score.
    """
    eps  = 1e-8
    p    = max(eps, min(1 - eps, p))
    odds = p / (1 - p)    # odds of approval
    score = BASE_SCORE + FACTOR * math.log(odds / BASE_ODDS)
    return int(round(max(300, min(900, score))))


def get_risk_band(prob: float) -> RiskBand:
    """Map approval probability to a named risk band."""
    for lo, hi, band, score_range, description in RISK_BANDS:
        if lo <= prob < hi:
            return RiskBand(band=band, score_range=score_range, description=description)
    # Fallback — very low probability
    return RiskBand(band="Very High", score_range="300–579",
                    description="High default risk — decline or secured card only")


# ══════════════════════════════════════════════════════════════════════════════
# ModelStore — singleton loaded at startup
# ══════════════════════════════════════════════════════════════════════════════

class ModelStore:
    """
    Holds the loaded model artifacts in memory.
    load() is called once in the FastAPI lifespan.
    predict() is called on every request — it only does numpy operations.
    """
    _model        = None
    _scaler       = None
    _features     = None
    _le_dict      = None
    _cat_cols     = None
    _skewed_cols  = None
    _engineered   = None
    _metrics      = None
    _version      = None
    _loaded       = False

    @classmethod
    def load(cls, path: str) -> None:
        """Deserialise the joblib artifact and warm up the model."""
        logger.info(f"Loading model from {path!r} ...")
        art = joblib.load(path)
        cls._model       = art["model"]
        cls._scaler      = art["scaler"]
        cls._features    = art["features"]
        cls._le_dict     = art["le_dict"]
        cls._cat_cols    = art["cat_cols"]
        cls._skewed_cols = art["skewed_cols"]
        cls._metrics     = art["metrics"]
        cls._version     = art["model_version"]
        cls._loaded      = True
        logger.info(f"Model {cls._version!r} loaded — {len(cls._features)} features. "
                    f"AUC={cls._metrics.get('auc')} Gini={cls._metrics.get('gini')}")

        # Warm-up: run one dummy prediction so first real request is fast
        try:
            dummy = np.zeros((1, len(cls._features)))
            cls._model.predict_proba(cls._scaler.transform(dummy))
            logger.info("Model warm-up complete.")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-fatal): {e}")

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    @classmethod
    def features(cls):
        return cls._features

    @classmethod
    def metrics(cls):
        return cls._metrics

    @classmethod
    def version(cls):
        return cls._version

    @classmethod
    def _engineer_features(cls, raw: Dict[str, Any]) -> Dict[str, float]:
        """
        Replicate the training feature engineering pipeline exactly.
        Input: flat dict of raw feature values (from PredictRequest).
        Output: dict mapping feature_name → float value.
        """
        eps = 1e-6
        f   = dict(raw)   # copy

        # ── Engineered ratios ────────────────────────────────────────────────
        f["income_to_limit_ratio"]    = f.get("annual_income", 0) / (f.get("requested_credit_limit", 1) + eps)
        f["expenses_to_income_ratio"] = f.get("total_monthly_expenses", 0) / (f.get("monthly_income", 1) + eps)
        f["savings_to_income_ratio"]  = f.get("savings_account_balance", 0) / (f.get("annual_income", 1) + eps)
        f["bureau_score_mean"]        = np.mean([
            f.get("fico_score", 700), f.get("equifax_score", 700),
            f.get("experian_score", 700), f.get("transunion_score", 700)
        ])
        f["total_debt_burden"]        = f.get("total_liabilities", 0) / (f.get("total_assets", 1) + eps)
        f["disposable_monthly"]       = f.get("monthly_income", 0) - f.get("total_monthly_expenses", 0)
        f["utilization_x_inquiries"]  = (f.get("credit_utilization_ratio", 0) *
                                          f.get("hard_inquiries_last_12mo", 0))
        f["net_worth_to_limit"]       = f.get("net_worth", 0) / (f.get("requested_credit_limit", 1) + eps)

        # ── Log transforms for skewed features ───────────────────────────────
        for col in cls._skewed_cols:
            if col in f:
                f[f"log_{col}"] = float(np.log1p(abs(f[col])))

        # ── Label-encode categoricals ────────────────────────────────────────
        for col in cls._cat_cols:
            if col in cls._le_dict:
                le = cls._le_dict[col]
                raw_val = str(f.get(col, "Unknown"))
                # Handle unseen categories gracefully (use most frequent class = 0)
                classes = list(le.classes_)
                encoded = le.transform([raw_val])[0] if raw_val in classes else 0
                f[f"{col}_enc"] = int(encoded)

        return f

    @classmethod
    def predict(cls, request: "PredictRequest") -> Tuple[float, int, str, RiskBand]:
        """
        Run the full prediction pipeline.
        Returns: (approval_probability, credit_score, decision, risk_band)
        """
        if not cls._loaded:
            raise RuntimeError("Model has not been loaded. Call ModelStore.load() at startup.")

        # Convert Pydantic model to flat dict
        raw = request.model_dump(exclude={"extra_features", "applicant_ref"})

        # Merge extra_features if provided
        if request.extra_features:
            raw.update(request.extra_features)

        # Engineer features
        features_dict = cls._engineer_features(raw)

        # Build the feature vector in the exact order the model expects
        row = np.array([[
            float(features_dict.get(fname, 0.0))
            for fname in cls._features
        ]])

        # Scale (scaler was fit on SMOTE-augmented training data)
        row_scaled = cls._scaler.transform(row)

        # Predict
        prob = float(cls._model.predict_proba(row_scaled)[0, 1])  # P(Approved)

        # Derive outputs
        score    = prob_to_score(prob)
        decision = "Approved" if prob >= DECISION_THRESHOLD else "Declined"
        band     = get_risk_band(prob)

        return prob, score, decision, band
