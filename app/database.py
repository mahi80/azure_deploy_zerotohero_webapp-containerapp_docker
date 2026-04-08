"""
app/database.py — SQLAlchemy engine, session factory, and Base class.

PostgreSQL is used for:
  - User authentication (hashed passwords + JWT)
  - Audit log of every prediction (prediction_log table)

The engine is created once at import time.
Sessions are injected per-request via FastAPI's Depends().
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping=True — verifies each connection before use, handles restarts
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,   # Set True for SQL debug logging
)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Base class for ORM models ─────────────────────────────────────────────────
Base = declarative_base()


def get_db():
    """
    FastAPI dependency: yields a DB session per request.
    Session is always closed in the finally block — even on errors.

    Usage in route:
        @router.get("/something")
        def my_route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
