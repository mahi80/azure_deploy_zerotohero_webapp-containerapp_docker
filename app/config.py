"""
app/config.py — Environment-driven configuration.
All secrets come from environment variables or a .env file.
Never hard-code credentials in source code.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────
    app_name: str = "CC Underwriting API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Security ─────────────────────────────────────────────────
    # Generate with: openssl rand -hex 32
    secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── Database ─────────────────────────────────────────────────
    # Format: postgresql://user:password@host:port/dbname
    database_url: str = "postgresql://ccapi:ccapi_pass@localhost:5432/cc_underwriting"

    # ── Model ────────────────────────────────────────────────────
    model_path: str = "models/cc_model_v1.joblib"

    # ── Pagination ───────────────────────────────────────────────
    default_page_size: int = 50
    max_page_size: int = 500

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — loaded once at startup."""
    return Settings()
