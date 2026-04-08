"""
app/main.py — FastAPI application entry point.

Route map:
  GET  /health                  — liveness probe (no auth)
  GET  /                        — API info (no auth)

  POST /auth/register           — create new user account
  POST /auth/token              — login → JWT
  GET  /auth/me                 — current user info

  POST /predict                 — run underwriting prediction  [auth required]
  GET  /predictions             — paginated prediction history [auth required]
  GET  /predictions/{id}        — single prediction detail     [auth required]

  GET  /model/info              — model metadata & metrics     [auth required]

  GET  /admin/users             — list all users               [admin only]
  POST /admin/users/{id}/deactivate — deactivate a user       [admin only]
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

import logging
import os

from fastapi import (
    FastAPI, Depends, HTTPException, Request, status, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import get_settings
from app.database import engine, Base, get_db
from app.models import User, PredictionLog
from app.schemas import (
    UserCreate, UserOut, TokenResponse,
    PredictRequest, PredictResponse, RiskBand,
    PaginatedPredictions, PredictionLogOut,
    HealthResponse, ModelInfoResponse,
)
from app.auth import (
    authenticate_user, create_user, create_access_token,
    get_current_user, get_current_admin, hash_password,
)
from app.predict import ModelStore

settings = get_settings()
logger   = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the first request and shutdown after the last.
    - Creates all DB tables (idempotent — safe to run on every restart).
    - Loads the ML model artifact into memory once.
    - Creates a default admin user if the users table is empty.
    """
    logger.info("=== CC Underwriting API starting up ===")

    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created.")

    # Load ML model
    ModelStore.load(settings.model_path)

    # Seed default admin user
    _seed_admin()

    logger.info("Startup complete — API is ready.")
    yield
    logger.info("=== CC Underwriting API shutting down ===")


def _seed_admin():
    """Create default admin if no users exist. Credentials from env vars."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin_username = os.getenv("ADMIN_USERNAME", "admin")
            admin_password = os.getenv("ADMIN_PASSWORD", "Admin@12345")
            admin_email    = os.getenv("ADMIN_EMAIL",    "admin@ccapi.local")
            create_user(db, username=admin_username, email=admin_email,
                        password=admin_password, full_name="System Admin",
                        is_admin=True)
            logger.info(f"Default admin user '{admin_username}' created.")
    except Exception as e:
        logger.warning(f"Admin seed skipped: {e}")
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════
# App instance
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CC Underwriting API",
    description=(
        "Credit card underwriting classification API.\n\n"
        "LightGBM model trained on 4,480 applicants — AUC 0.997, Gini 0.994, KS 0.944.\n\n"
        "**Authentication:** POST /auth/token → copy the access_token → "
        "click 🔒 Authorize above → paste token."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# Health & root
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs":    "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(db: Session = Depends(get_db)):
    """
    Liveness probe.
    Checks that both the ML model and the database are operational.
    Used by Docker health-check, Kubernetes liveness probe, and Azure App Service.
    """
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (ModelStore.is_loaded() and db_ok) else "degraded",
        app_version=settings.app_version,
        model_version=ModelStore.version() or "not-loaded",
        model_loaded=ModelStore.is_loaded(),
        db_connected=db_ok,
    )


# ══════════════════════════════════════════════════════════════════
# Auth routes
# ══════════════════════════════════════════════════════════════════

@app.post("/auth/register", response_model=UserOut,
          status_code=status.HTTP_201_CREATED, tags=["Auth"])
def register(payload: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new API user.
    Username and email must be unique.
    Password is bcrypt-hashed before storage — plain text is never saved.
    """
    try:
        user = create_user(
            db,
            username=payload.username,
            email=str(payload.email),
            password=payload.password,
            full_name=payload.full_name,
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   Session = Depends(get_db),
):
    """
    OAuth2 password flow — exchange username + password for a JWT.

    **How to use in Swagger UI:**
    1. Click this endpoint → Try it out → enter credentials → Execute.
    2. Copy the `access_token` value.
    3. Click the 🔒 Authorize button at the top.
    4. Paste the token → Authorize.
    5. All protected routes now send the token automatically.
    """
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Update last login timestamp
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    expires = timedelta(minutes=settings.access_token_expire_minutes)
    token   = create_access_token({"sub": user.username}, expires_delta=expires)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
def me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user


# ══════════════════════════════════════════════════════════════════
# Prediction routes
# ══════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse,
          status_code=status.HTTP_200_OK, tags=["Prediction"])
def predict(
    payload:      PredictRequest,
    request:      Request,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """
    Run credit card underwriting prediction for one applicant.

    **Returns:**
    - `decision` — "Approved" or "Declined"
    - `approval_prob` — model's probability of approval (0.0–1.0)
    - `credit_score` — FICO-style scorecard points (600–900)
    - `risk_band` — categorical risk tier
    - `prediction_id` — DB row ID for audit trail

    **Auth:** Bearer JWT required.  Obtain token from POST /auth/token.
    """
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded — try again shortly")

    # Run prediction
    try:
        prob, score, decision, band = ModelStore.predict(payload)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Persist to audit log
    log_entry = PredictionLog(
        user_id       = current_user.id,
        applicant_ref = payload.applicant_ref,
        input_features= payload.model_dump(exclude={"extra_features"}),
        approval_prob = prob,
        credit_score  = score,
        decision      = decision,
        risk_band     = band.band,
        model_version = ModelStore.version(),
        request_ip    = request.client.host if request.client else None,
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    return PredictResponse(
        applicant_ref    = payload.applicant_ref,
        decision         = decision,
        approval_prob    = round(prob, 4),
        credit_score     = score,
        risk_band        = band.band,
        risk_band_detail = band,
        model_version    = ModelStore.version(),
        prediction_id    = log_entry.id,
        timestamp        = log_entry.created_at,
    )


@app.get("/predictions", response_model=PaginatedPredictions, tags=["Prediction"])
def list_predictions(
    page:         int     = Query(default=1,  ge=1,  description="Page number (1-indexed)"),
    size:         int     = Query(default=50, ge=1,  le=500, description="Records per page"),
    decision:     str     = Query(default=None, description="Filter: 'Approved' or 'Declined'"),
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """
    Paginated list of prediction history for the authenticated user.

    **Pagination:**
    - `page=1&size=50` returns the first 50 records.
    - `page=2&size=50` returns records 51–100.
    - Response includes `total_count`, `has_next`, `has_prev`.
    - Admin users see all predictions; regular users see only their own.
    """
    query = db.query(PredictionLog)

    # Admins see everything; regular users see only their own
    if not current_user.is_admin:
        query = query.filter(PredictionLog.user_id == current_user.id)

    if decision in ("Approved", "Declined"):
        query = query.filter(PredictionLog.decision == decision)

    total = query.count()
    skip  = (page - 1) * size
    rows  = (query
             .order_by(PredictionLog.created_at.desc())
             .offset(skip)
             .limit(size)
             .all())

    return PaginatedPredictions(
        total_count = total,
        page        = page,
        size        = size,
        has_next    = (skip + size) < total,
        has_prev    = page > 1,
        data        = rows,
    )


@app.get("/predictions/{prediction_id}", response_model=PredictionLogOut, tags=["Prediction"])
def get_prediction(
    prediction_id: int,
    db:            Session = Depends(get_db),
    current_user:  User    = Depends(get_current_user),
):
    """Fetch a single prediction by its ID."""
    row = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    # Regular users may only view their own predictions
    if not current_user.is_admin and row.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return row


# ══════════════════════════════════════════════════════════════════
# Model info
# ══════════════════════════════════════════════════════════════════

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(current_user: User = Depends(get_current_user)):
    """
    Returns metadata about the currently loaded model.
    Useful for monitoring, audit, and integration verification.
    """
    return ModelInfoResponse(
        model_version = ModelStore.version(),
        n_features    = len(ModelStore.features()),
        feature_names = ModelStore.features(),
        metrics       = ModelStore.metrics(),
        description   = (
            "LightGBM classifier trained on CC underwriting data. "
            "Features: demographics, credit bureau, financial ratios, engineered interactions. "
            "SMOTE applied on train set only. StandardScaler fitted on train set."
        ),
    )


# ══════════════════════════════════════════════════════════════════
# Admin routes
# ══════════════════════════════════════════════════════════════════

@app.get("/admin/users", response_model=list[UserOut], tags=["Admin"])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """[Admin only] List all registered users."""
    return db.query(User).offset(skip).limit(limit).all()


@app.post("/admin/users/{user_id}/deactivate", response_model=UserOut, tags=["Admin"])
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """[Admin only] Deactivate a user account (blocks login without deleting history)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = False
    db.commit()
    db.refresh(user)
    return user
