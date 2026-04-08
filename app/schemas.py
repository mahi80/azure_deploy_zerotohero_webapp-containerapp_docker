"""
app/schemas.py — Pydantic v2 schemas for all request bodies and responses.

Pydantic does three jobs here:
  1. Validates incoming JSON automatically (HTTP 422 on bad input).
  2. Documents the API in Swagger UI (/docs) — every field shows up with description.
  3. Serialises outgoing responses to JSON cleanly.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# ══════════════════════════════════════════════════════════════════
# AUTH schemas
# ══════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    """Body for POST /auth/register"""
    username:  str = Field(..., min_length=3, max_length=64,  description="Unique login name")
    email:     EmailStr
    full_name: Optional[str] = Field(None, max_length=128)
    password:  str = Field(..., min_length=8, description="Min 8 characters")

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores and hyphens allowed)")
        return v.lower()


class UserOut(BaseModel):
    """Response for user info endpoints"""
    id:        int
    username:  str
    email:     str
    full_name: Optional[str]
    is_active: bool
    is_admin:  bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Response from POST /auth/token"""
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = Field(..., description="Token lifetime in seconds")


class TokenData(BaseModel):
    """Decoded JWT payload (internal use only)"""
    username: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# PREDICTION schemas
# ══════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """
    POST /predict request body.
    The applicant's raw feature values.
    All numeric fields default to sensible medians so callers can send
    partial records — but production usage should send all fields.
    """
    applicant_ref: Optional[str] = Field(None, description="Caller-supplied reference ID")

    # Core demographics
    age:                    float = Field(35,    ge=18, le=100)
    annual_income:          float = Field(60000, ge=0)
    monthly_income:         float = Field(5000,  ge=0)
    dependents_count:       float = Field(1,     ge=0)
    years_employed:         float = Field(3,     ge=0)

    # Credit profile
    fico_score:             float = Field(700,   ge=300, le=850)
    equifax_score:          float = Field(700,   ge=300, le=850)
    experian_score:         float = Field(700,   ge=300, le=850)
    transunion_score:       float = Field(700,   ge=300, le=850)
    avg_bureau_score:       float = Field(700,   ge=300, le=850)
    credit_utilization_ratio: float = Field(0.3, ge=0.0, le=1.0)
    debt_to_income_ratio:   float = Field(0.35,  ge=0.0)
    num_total_credit_accounts: float = Field(5,  ge=0)
    num_open_accounts:      float = Field(3,     ge=0)
    hard_inquiries_last_12mo: float = Field(1,   ge=0)
    late_payments_last_24mo:  float = Field(0,   ge=0)
    derogatory_marks_count:   float = Field(0,   ge=0)
    collections_accounts:     float = Field(0,   ge=0)
    bankruptcy_count:         float = Field(0,   ge=0)
    months_since_last_delinquency: float = Field(24, ge=0)
    credit_history_length_months: float = Field(60,  ge=0)

    # Financial position
    net_worth:                    float = Field(50000)
    total_assets:                 float = Field(100000, ge=0)
    total_liabilities:            float = Field(40000,  ge=0)
    savings_account_balance:      float = Field(10000,  ge=0)
    checking_account_balance:     float = Field(3000,   ge=0)
    requested_credit_limit:       float = Field(10000,  ge=500)
    total_monthly_expenses:       float = Field(3500,   ge=0)
    monthly_rent_mortgage:        float = Field(1500,   ge=0)

    # Application
    fraud_risk_score:             float = Field(20,  ge=0, le=100)
    identity_verification_score:  float = Field(85,  ge=0, le=100)
    address_stability_score:      float = Field(70,  ge=0, le=100)

    # Categorical (string) — matched against label encoders at prediction time
    gender:                 str = Field("Male")
    marital_status:         str = Field("Single")
    education_level:        str = Field("Bachelor's")
    employment_status:      str = Field("Full-time")
    housing_status:         str = Field("Renter")
    card_type_requested:    str = Field("Standard")
    application_channel:    str = Field("Online")
    prior_default_flag:     str = Field("No")
    prior_bankruptcy_flag:  str = Field("No")
    thin_file_flag:         str = Field("No")

    # Pass-through extras as a flexible dict for any remaining model features
    extra_features: Optional[Dict[str, Any]] = Field(
        None, description="Any additional feature values keyed by feature name"
    )


class RiskBand(BaseModel):
    band:        str
    score_range: str
    description: str


class PredictResponse(BaseModel):
    """Response from POST /predict"""
    applicant_ref:   Optional[str]
    decision:        str  = Field(..., description="'Approved' or 'Declined'")
    approval_prob:   float = Field(..., description="P(Approved) — 0.0 to 1.0")
    credit_score:    int   = Field(..., description="Scorecard points on 600–900 scale")
    risk_band:       str   = Field(..., description="Very Low / Low / Medium / High / Very High")
    risk_band_detail: RiskBand
    model_version:   str
    prediction_id:   int   = Field(..., description="ID in prediction_log table for audit")
    timestamp:       datetime


class PredictionLogOut(BaseModel):
    """Single row from GET /predictions list"""
    id:            int
    applicant_ref: Optional[str]
    approval_prob: float
    credit_score:  Optional[int]
    decision:      str
    risk_band:     Optional[str]
    model_version: str
    created_at:    datetime

    model_config = {"from_attributes": True}


class PaginatedPredictions(BaseModel):
    """Paginated response from GET /predictions"""
    total_count:  int
    page:         int
    size:         int
    has_next:     bool
    has_prev:     bool
    data:         List[PredictionLogOut]


# ══════════════════════════════════════════════════════════════════
# HEALTH / INFO schemas
# ══════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:        str = "ok"
    app_version:   str
    model_version: str
    model_loaded:  bool
    db_connected:  bool


class ModelInfoResponse(BaseModel):
    model_version: str
    n_features:    int
    feature_names: List[str]
    metrics:       Dict[str, float]
    description:   str
