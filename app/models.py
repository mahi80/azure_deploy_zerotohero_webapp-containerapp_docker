"""
app/models.py — SQLAlchemy ORM table definitions.

Tables:
  users           — Registered API users with hashed passwords.
  prediction_log  — Audit trail of every prediction request.
                    Used for pagination on GET /predictions.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """
    Stores API credentials.
    Passwords are stored as bcrypt hashes — plain-text is never persisted.
    is_active = False disables a user without deleting their history.
    is_admin   = True grants access to admin-only endpoints (e.g., /admin/users).
    """
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(64),  unique=True, index=True, nullable=False)
    email      = Column(String(128), unique=True, index=True, nullable=False)
    full_name  = Column(String(128), nullable=True)
    hashed_password = Column(String(256), nullable=False)
    is_active  = Column(Boolean, default=True)
    is_admin   = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationship — one user → many prediction log entries
    predictions = relationship("PredictionLog", back_populates="user")

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r}>"


class PredictionLog(Base):
    """
    Audit log of every prediction made via POST /predict.

    Stores:
      - Who made the request (user_id FK)
      - Raw input features (JSON column)
      - Model output (probability + decision)
      - Which model version was used
      - Timestamp

    This table supports:
      - GET /predictions  — paginated history for the calling user
      - Regulatory audit trail
      - Monitoring dashboards
    """
    __tablename__ = "prediction_log"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    applicant_ref   = Column(String(64), nullable=True, index=True)  # optional caller-supplied ID
    input_features  = Column(JSON, nullable=True)     # raw request payload
    approval_prob   = Column(Float, nullable=False)   # P(Approved)
    credit_score    = Column(Integer, nullable=True)  # scorecard points (600-900 scale)
    decision        = Column(String(16), nullable=False)  # "Approved" | "Declined"
    risk_band       = Column(String(32), nullable=True)   # "Very Low" … "Very High"
    model_version   = Column(String(32), nullable=False)
    request_ip      = Column(String(64), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="predictions")

    def __repr__(self):
        return f"<PredictionLog id={self.id} decision={self.decision!r}>"
