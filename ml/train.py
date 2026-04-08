"""
ml/train.py — Standalone, production-ready training pipeline.

Refactored from the notebook solution. Run this script to re-train
the model and overwrite the joblib artifact.

Usage:
    python ml/train.py
    python ml/train.py --data path/to/data.csv --out models/cc_model_v2.joblib

Pipeline stages (mirrors notebook exactly, production-hardened):
  1. Load raw CSV
  2. Drop identifiers, leakage cols, high-missingness cols
  3. Impute missing values (median for numeric, mode for categorical)
  4. Encode target (Yes=1 / No=0)
  5. Feature engineering (ratios, log transforms, label encoding)
  6. Correlation filter (drop one of any pair with |r| > 0.95)
  7. Stratified 80/20 train-test split (time-ordered data → use OOT in production)
  8. SMOTE on training data only
  9. StandardScaler fitted on SMOTE-augmented training data
 10. LightGBM with early stopping
 11. Evaluate — AUC, Gini, KS, confusion matrix
 12. Save joblib artifact (model + scaler + feature metadata)
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 42

# ── Columns to always drop ────────────────────────────────────────────────────
# These are identifiers, post-event leakage, or alternative targets — none
# should be available at prediction time for a live applicant.
DROP_ALWAYS = [
    "applicant_id",
    "application_date",
    "target_credit_limit_assigned",   # post-decision outcome
    "underwriting_model_score_v1",    # leakage — derived from target
    "underwriting_model_score_v2",
    "underwriting_model_score_v3",
    "combined_risk_score",
    "predicted_default_probability",  # leakage
    "loss_given_default_estimate",    # post-event
]

SKEWED_COLS = [
    "annual_income", "monthly_income", "net_worth",
    "total_assets", "savings_account_balance",
    "requested_credit_limit", "total_liabilities",
]

MAX_MISSING_PCT   = 50.0   # drop cols with > 50% missing
MAX_CAT_CARDINALITY = 30   # only encode cats with <= 30 unique values
CORR_THRESHOLD    = 0.95   # drop one of any pair with |r| > this


# ══════════════════════════════════════════════════════════════════
# Stage 1: Load
# ══════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading: {path}")
    df = pd.read_csv(path)
    log.info(f"  Raw shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# Stage 2: Drop columns
# ══════════════════════════════════════════════════════════════════

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    # High-missingness columns
    miss_pct    = df.isnull().mean() * 100
    high_miss   = miss_pct[miss_pct > MAX_MISSING_PCT].index.tolist()
    if high_miss:
        log.info(f"  Dropping {len(high_miss)} high-missingness cols: {high_miss}")

    all_drop = list(set(DROP_ALWAYS + high_miss))
    df = df.drop(columns=all_drop, errors="ignore")
    log.info(f"  Shape after drops: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════
# Stage 3: Encode target + identify feature types
# ══════════════════════════════════════════════════════════════════

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = (df["target_approved"] == "Yes").astype(int)
    df = df.drop(columns=["target_approved"])
    counts = df["target"].value_counts()
    rate   = counts.get(1, 0) / len(df)
    log.info(f"  Target: {counts.to_dict()}  event_rate={rate:.1%}")
    return df


def get_feature_types(df: pd.DataFrame):
    num_cols = (df
                .select_dtypes(include="number")
                .columns
                .drop("target", errors="ignore")
                .tolist())
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if df[c].nunique() <= MAX_CAT_CARDINALITY
    ]
    log.info(f"  Numeric: {len(num_cols)}  Categorical (≤{MAX_CAT_CARDINALITY} uniq): {len(cat_cols)}")
    return num_cols, cat_cols


# ══════════════════════════════════════════════════════════════════
# Stage 4: Impute
# ══════════════════════════════════════════════════════════════════

def impute(df: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
    # Numeric → median (robust to skew)
    impute_stats = {}
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            impute_stats[col] = med

    # Categorical → mode
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            impute_stats[col] = mode_val

    remaining_missing = df.isnull().sum().sum()
    log.info(f"  Imputation complete. Remaining nulls: {remaining_missing}")
    return df, impute_stats


# ══════════════════════════════════════════════════════════════════
# Stage 5: Feature Engineering
# ══════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, cat_cols: list):
    eps = 1e-6

    # ── Ratio / interaction features ──────────────────────────────
    df["income_to_limit_ratio"]    = df["annual_income"]           / (df["requested_credit_limit"] + eps)
    df["expenses_to_income_ratio"] = df["total_monthly_expenses"]  / (df["monthly_income"] + eps)
    df["savings_to_income_ratio"]  = df["savings_account_balance"] / (df["annual_income"] + eps)
    df["bureau_score_mean"]        = df[["fico_score","equifax_score","experian_score","transunion_score"]].mean(axis=1)
    df["total_debt_burden"]        = df["total_liabilities"]       / (df["total_assets"] + eps)
    df["disposable_monthly"]       = df["monthly_income"]          - df["total_monthly_expenses"]
    df["utilization_x_inquiries"]  = df["credit_utilization_ratio"] * df["hard_inquiries_last_12mo"]
    df["net_worth_to_limit"]       = df["net_worth"]               / (df["requested_credit_limit"] + eps)

    engineered = [
        "income_to_limit_ratio", "expenses_to_income_ratio",
        "savings_to_income_ratio", "bureau_score_mean", "total_debt_burden",
        "disposable_monthly", "utilization_x_inquiries", "net_worth_to_limit",
    ]

    # ── Log transforms for skewed numerics ────────────────────────
    log_feats = []
    for col in SKEWED_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(np.abs(df[col]))
            log_feats.append(f"log_{col}")

    # ── Label-encode categoricals ─────────────────────────────────
    le_dict   = {}
    enc_feats = []
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        le_dict[col]     = le
        enc_feats.append(col + "_enc")

    log.info(f"  Engineered: {len(engineered)}  Log: {len(log_feats)}  Encoded: {len(enc_feats)}")
    return df, engineered, log_feats, enc_feats, le_dict


# ══════════════════════════════════════════════════════════════════
# Stage 6: Build feature matrix + correlation filter
# ══════════════════════════════════════════════════════════════════

def build_feature_matrix(df, num_cols, engineered, log_feats, enc_feats):
    all_feats   = num_cols + engineered + log_feats + enc_feats
    all_feats   = [f for f in all_feats if f in df.columns]
    X           = df[all_feats].fillna(0)   # safety fill

    # Correlation filter — drop one of any pair with |r| > threshold
    corr        = X.corr().abs()
    upper       = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr   = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
    final_feats = [f for f in all_feats if f not in drop_corr]
    X           = X[final_feats]

    log.info(f"  All features: {len(all_feats)}  Dropped (corr): {len(drop_corr)}  Final: {len(final_feats)}")
    return X, final_feats


# ══════════════════════════════════════════════════════════════════
# Stage 7–9: Split, SMOTE, Scale
# ══════════════════════════════════════════════════════════════════

def split_and_preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    log.info(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # SMOTE — applied ONLY on training data
    smote       = SMOTE(random_state=SEED, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)
    log.info(f"  After SMOTE: {X_tr_sm.shape[0]} rows  class dist: {pd.Series(y_tr_sm).value_counts().to_dict()}")

    # StandardScaler — fitted ONLY on SMOTE-augmented training data
    scaler      = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_tr_sm)
    X_te_sc     = scaler.transform(X_test)

    return X_tr_sc, X_te_sc, y_tr_sm, y_test, scaler


# ══════════════════════════════════════════════════════════════════
# Stage 10: Train LightGBM
# ══════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_test, y_test):
    log.info("  Training LightGBM classifier ...")
    model = lgb.LGBMClassifier(
        n_estimators       = 500,
        learning_rate      = 0.05,
        num_leaves         = 63,
        min_child_samples  = 30,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        class_weight       = "balanced",
        random_state       = SEED,
        n_jobs             = -1,
        verbose            = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    log.info(f"  Best iteration: {model.best_iteration_}")
    return model


# ══════════════════════════════════════════════════════════════════
# Stage 11: Evaluate
# ══════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, final_feats):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc  = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks   = float(np.max(tpr - fpr))
    cm   = confusion_matrix(y_test, y_pred)

    log.info("=" * 55)
    log.info("  MODEL EVALUATION")
    log.info(f"  AUC   : {auc:.4f}")
    log.info(f"  Gini  : {gini:.4f}")
    log.info(f"  KS    : {ks:.4f}")
    log.info(f"  Features used: {len(final_feats)}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Declined", "Approved"]))
    log.info(f"  Confusion Matrix:\n{cm}")
    log.info("=" * 55)

    return {"auc": round(auc, 4), "gini": round(gini, 4), "ks": round(ks, 4)}


# ══════════════════════════════════════════════════════════════════
# Stage 12: Save artifact
# ══════════════════════════════════════════════════════════════════

def save_artifact(path: str, model, scaler, final_feats, le_dict,
                  cat_cols, num_cols, engineered, log_feats, enc_feats,
                  metrics: dict, version: str = "v1.0.0"):
    artifact = {
        "model"         : model,
        "scaler"        : scaler,
        "features"      : final_feats,
        "le_dict"       : le_dict,
        "cat_cols"      : cat_cols,
        "num_cols"      : num_cols,
        "engineered"    : engineered,
        "log_feats"     : log_feats,
        "enc_feats"     : enc_feats,
        "skewed_cols"   : SKEWED_COLS,
        "metrics"       : metrics,
        "model_version" : version,
        "n_features"    : len(final_feats),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    log.info(f"  Saved → {path}  ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train CC Underwriting LightGBM model")
    parser.add_argument("--data",    default="cc_underwriting_5k_stratified11.csv")
    parser.add_argument("--out",     default="models/cc_model_v1.joblib")
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()

    log.info("── Stage 1: Load ──")
    df = load_data(args.data)

    log.info("── Stage 2: Drop columns ──")
    df = drop_columns(df)

    log.info("── Stage 3: Encode target ──")
    df = encode_target(df)
    num_cols, cat_cols = get_feature_types(df)

    log.info("── Stage 4: Impute ──")
    df, impute_stats = impute(df, num_cols, cat_cols)

    log.info("── Stage 5: Feature engineering ──")
    df, engineered, log_feats, enc_feats, le_dict = engineer_features(df, cat_cols)

    log.info("── Stage 6: Feature matrix + correlation filter ──")
    X, final_feats = build_feature_matrix(df, num_cols, engineered, log_feats, enc_feats)
    y = df["target"]

    log.info("── Stages 7-9: Split / SMOTE / Scale ──")
    X_tr_sc, X_te_sc, y_tr_sm, y_test, scaler = split_and_preprocess(X, y)

    log.info("── Stage 10: Train LightGBM ──")
    model = train_model(X_tr_sc, y_tr_sm, X_te_sc, y_test)

    log.info("── Stage 11: Evaluate ──")
    metrics = evaluate(model, X_te_sc, y_test, final_feats)

    log.info("── Stage 12: Save artifact ──")
    save_artifact(
        path=args.out, model=model, scaler=scaler,
        final_feats=final_feats, le_dict=le_dict,
        cat_cols=cat_cols, num_cols=num_cols,
        engineered=engineered, log_feats=log_feats, enc_feats=enc_feats,
        metrics=metrics, version=args.version,
    )
    log.info("✅ Training complete.")


if __name__ == "__main__":
    main()
