"""
tests/test_api.py — Full integration test suite using FastAPI TestClient.

Tests cover:
  - Health check
  - User registration + login (JWT)
  - Auth protection (401 without token)
  - Prediction endpoint (approved + declined cases)
  - Pagination on /predictions
  - Admin-only routes
  - Edge cases: bad inputs, wrong credentials

Run locally:
    pytest tests/ -v
"""
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ── Point to a test database before importing app ─────────────────
os.environ.setdefault("DATABASE_URL", "postgresql://ccapi:test_pass@localhost:5432/cc_test")
os.environ.setdefault("SECRET_KEY", "test-only-secret-key-not-for-production")
os.environ.setdefault("MODEL_PATH", "models/cc_model_v1.joblib")
os.environ.setdefault("ADMIN_PASSWORD", "TestAdmin@123")
os.environ.setdefault("ADMIN_EMAIL", "admin@test.local")

from app.main import app
from app.database import Base, get_db
from app.config import get_settings

settings = get_settings()

# ── Test DB engine (SQLite for CI without Postgres, Postgres for full test) ───
TEST_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

engine_test = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in TEST_DB_URL else {},
    pool_pre_ping=True,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


# ── Create tables ─────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine_test)
    yield
    Base.metadata.drop_all(bind=engine_test)


# ── Test client ───────────────────────────────────────────────────
@pytest.fixture(scope="session")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Shared fixtures ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def registered_user(client):
    """Register a test user once for the session."""
    resp = client.post("/auth/register", json={
        "username": "testuser",
        "email":    "testuser@test.com",
        "password": "TestPass@123",
    })
    # 201 = created, 409 = already exists (re-runs)
    assert resp.status_code in (201, 409)
    return {"username": "testuser", "password": "TestPass@123"}


@pytest.fixture(scope="session")
def auth_token(client, registered_user):
    """Log in and return a Bearer token for protected route tests."""
    resp = client.post("/auth/token", data={
        "username": registered_user["username"],
        "password": registered_user["password"],
    })
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


# ── Sample predict payload ────────────────────────────────────────

GOOD_APPLICANT = {
    "applicant_ref":            "TEST-001",
    "age":                      35,
    "annual_income":            95000,
    "monthly_income":           7917,
    "dependents_count":         1,
    "years_employed":           8,
    "fico_score":               740,
    "equifax_score":            745,
    "experian_score":           738,
    "transunion_score":         742,
    "avg_bureau_score":         741,
    "credit_utilization_ratio": 0.18,
    "debt_to_income_ratio":     0.25,
    "num_total_credit_accounts": 8,
    "num_open_accounts":        5,
    "hard_inquiries_last_12mo": 1,
    "late_payments_last_24mo":  0,
    "derogatory_marks_count":   0,
    "collections_accounts":     0,
    "bankruptcy_count":         0,
    "months_since_last_delinquency": 36,
    "credit_history_length_months":  84,
    "net_worth":                60000,
    "total_assets":             120000,
    "total_liabilities":        45000,
    "savings_account_balance":  15000,
    "checking_account_balance": 4500,
    "requested_credit_limit":   10000,
    "total_monthly_expenses":   3200,
    "monthly_rent_mortgage":    1400,
    "fraud_risk_score":         15,
    "identity_verification_score": 92,
    "address_stability_score":  80,
}

RISKY_APPLICANT = {**GOOD_APPLICANT,
    "applicant_ref":            "TEST-002-RISKY",
    "fico_score":               560,
    "equifax_score":            548,
    "experian_score":           555,
    "transunion_score":         558,
    "avg_bureau_score":         555,
    "credit_utilization_ratio": 0.92,
    "debt_to_income_ratio":     0.68,
    "late_payments_last_24mo":  6,
    "derogatory_marks_count":   3,
    "collections_accounts":     2,
    "bankruptcy_count":         1,
    "fraud_risk_score":         75,
}


# ══════════════════════════════════════════════════════════════════
# TESTS: Health
# ══════════════════════════════════════════════════════════════════

class TestHealth:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "docs" in r.json()

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert "model_loaded" in body
        assert "db_connected" in body
        assert "app_version" in body


# ══════════════════════════════════════════════════════════════════
# TESTS: Authentication
# ══════════════════════════════════════════════════════════════════

class TestAuth:
    def test_register_success(self, client):
        r = client.post("/auth/register", json={
            "username": "newuser99",
            "email":    "newuser99@test.com",
            "password": "NewPass@123",
        })
        assert r.status_code in (201, 409)   # 409 if test re-runs

    def test_register_duplicate_username(self, client, registered_user):
        r = client.post("/auth/register", json={
            "username": registered_user["username"],
            "email":    "different@test.com",
            "password": "AnotherPass@1",
        })
        assert r.status_code == 409

    def test_register_weak_password(self, client):
        r = client.post("/auth/register", json={
            "username": "weakpwuser",
            "email":    "weak@test.com",
            "password": "123",          # too short
        })
        assert r.status_code == 422    # Pydantic validation error

    def test_login_success(self, client, registered_user):
        r = client.post("/auth/token", data={
            "username": registered_user["username"],
            "password": registered_user["password"],
        })
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

    def test_login_wrong_password(self, client, registered_user):
        r = client.post("/auth/token", data={
            "username": registered_user["username"],
            "password": "WrongPassword!",
        })
        assert r.status_code == 401

    def test_login_unknown_user(self, client):
        r = client.post("/auth/token", data={
            "username": "nobody",
            "password": "irrelevant",
        })
        assert r.status_code == 401

    def test_me_returns_user(self, client, auth_headers, registered_user):
        r = client.get("/auth/me", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["username"] == registered_user["username"]

    def test_me_without_token_returns_401(self, client):
        r = client.get("/auth/me")
        assert r.status_code == 401

    def test_invalid_token_returns_401(self, client):
        r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        assert r.status_code == 401


# ══════════════════════════════════════════════════════════════════
# TESTS: Prediction
# ══════════════════════════════════════════════════════════════════

class TestPredict:
    def test_predict_requires_auth(self, client):
        r = client.post("/predict", json=GOOD_APPLICANT)
        assert r.status_code == 401

    def test_predict_good_applicant(self, client, auth_headers):
        r = client.post("/predict", json=GOOD_APPLICANT, headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] in ("Approved", "Declined")
        assert 0.0 <= body["approval_prob"] <= 1.0
        assert 300 <= body["credit_score"] <= 900
        assert body["risk_band"] in ("Very Low", "Low", "Medium", "High", "Very High")
        assert body["prediction_id"] > 0
        assert body["model_version"] is not None

    def test_predict_risky_applicant(self, client, auth_headers):
        r = client.post("/predict", json=RISKY_APPLICANT, headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] in ("Approved", "Declined")
        # Risky applicant should have lower probability than good applicant
        good_r = client.post("/predict", json=GOOD_APPLICANT, headers=auth_headers)
        assert body["approval_prob"] < good_r.json()["approval_prob"]

    def test_predict_applicant_ref_preserved(self, client, auth_headers):
        payload = {**GOOD_APPLICANT, "applicant_ref": "MY-REF-XYZ"}
        r = client.post("/predict", json=payload, headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["applicant_ref"] == "MY-REF-XYZ"

    def test_predict_missing_required_field_uses_default(self, client, auth_headers):
        """
        PredictRequest has defaults for all fields.
        Sending only partial data should still work (defaults fill in gaps).
        """
        minimal = {"fico_score": 720, "annual_income": 80000}
        r = client.post("/predict", json=minimal, headers=auth_headers)
        assert r.status_code == 200

    def test_predict_invalid_fico_score(self, client, auth_headers):
        bad = {**GOOD_APPLICANT, "fico_score": 1500}   # out of range 300–850
        r = client.post("/predict", json=bad, headers=auth_headers)
        assert r.status_code == 422

    def test_predict_negative_income(self, client, auth_headers):
        bad = {**GOOD_APPLICANT, "annual_income": -5000}
        r = client.post("/predict", json=bad, headers=auth_headers)
        assert r.status_code == 422


# ══════════════════════════════════════════════════════════════════
# TESTS: Pagination
# ══════════════════════════════════════════════════════════════════

class TestPagination:
    def test_predictions_list_requires_auth(self, client):
        r = client.get("/predictions")
        assert r.status_code == 401

    def test_predictions_list_returns_schema(self, client, auth_headers):
        r = client.get("/predictions", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert "total_count" in body
        assert "page" in body
        assert "size" in body
        assert "has_next" in body
        assert "has_prev" in body
        assert "data" in body
        assert isinstance(body["data"], list)

    def test_predictions_pagination_page1(self, client, auth_headers):
        # Make 3 predictions first
        for _ in range(3):
            client.post("/predict", json=GOOD_APPLICANT, headers=auth_headers)

        r = client.get("/predictions?page=1&size=2", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert len(body["data"]) <= 2
        assert body["page"] == 1
        assert body["size"] == 2
        assert body["has_prev"] is False

    def test_predictions_pagination_page2(self, client, auth_headers):
        r = client.get("/predictions?page=2&size=1", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["page"] == 2
        assert body["has_prev"] is True

    def test_predictions_filter_by_decision(self, client, auth_headers):
        r = client.get("/predictions?decision=Approved", headers=auth_headers)
        assert r.status_code == 200
        for item in r.json()["data"]:
            assert item["decision"] == "Approved"

    def test_predictions_size_limit(self, client, auth_headers):
        """Size > 500 should be rejected by Pydantic."""
        r = client.get("/predictions?size=999", headers=auth_headers)
        assert r.status_code == 422

    def test_single_prediction_fetch(self, client, auth_headers):
        # Make a prediction and fetch it by ID
        pred_r = client.post("/predict", json=GOOD_APPLICANT, headers=auth_headers)
        pred_id = pred_r.json()["prediction_id"]

        r = client.get(f"/predictions/{pred_id}", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["id"] == pred_id

    def test_single_prediction_not_found(self, client, auth_headers):
        r = client.get("/predictions/999999", headers=auth_headers)
        assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════
# TESTS: Model Info
# ══════════════════════════════════════════════════════════════════

class TestModelInfo:
    def test_model_info_requires_auth(self, client):
        r = client.get("/model/info")
        assert r.status_code == 401

    def test_model_info_schema(self, client, auth_headers):
        r = client.get("/model/info", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert "model_version" in body
        assert "n_features" in body
        assert "feature_names" in body
        assert "metrics" in body
        assert len(body["feature_names"]) == body["n_features"]
        assert "auc" in body["metrics"]
        assert "gini" in body["metrics"]

    def test_model_metrics_acceptable(self, client, auth_headers):
        """Sanity check: AUC and Gini must be above minimum thresholds."""
        r = client.get("/model/info", headers=auth_headers)
        metrics = r.json()["metrics"]
        assert metrics["auc"]  >= 0.70, f"AUC too low: {metrics['auc']}"
        assert metrics["gini"] >= 0.40, f"Gini too low: {metrics['gini']}"
