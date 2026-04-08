"""
app/auth.py — JWT authentication backed by PostgreSQL.

Flow:
  1. POST /auth/token   — user sends username + password
                         → password verified against bcrypt hash in DB
                         → signed JWT returned (expires in N minutes)
  2. Every protected route uses Depends(get_current_user)
     → extracts Bearer token from Authorization header
     → verifies JWT signature + expiry
     → loads User from DB
     → raises HTTP 401 if anything fails

Passwords are NEVER stored in plain text.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User
from app.schemas import TokenData

settings = get_settings()

# ── Password hashing ──────────────────────────────────────────────────────────
# bcrypt is the industry standard: slow-by-design, salted, resistant to GPU attacks
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── OAuth2 token URL ─────────────────────────────────────────────────────────
# FastAPI reads the Bearer token from the Authorization header automatically
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# ── Password utilities ────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plain-text password. Call this on registration."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Compare plain-text against stored hash. Returns True if match."""
    return pwd_context.verify(plain, hashed)


# ── JWT utilities ─────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Sign a JWT.
    - 'sub' claim holds the username.
    - 'exp' claim holds the expiry timestamp (UTC).
    - The token is signed with HMAC-SHA256 using the secret_key.
    Anyone with the token can call protected routes — guard it like a password.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT.
    Raises HTTP 401 if signature is invalid or token is expired.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exc
        return TokenData(username=username)
    except JWTError:
        raise credentials_exc


# ── DB-backed user operations ─────────────────────────────────────────────────

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Verify username + password against the DB.
    Returns the User object on success, None on failure.
    Timing-safe: always runs verify_password even if user not found
    (prevents username enumeration via timing attack).
    """
    user = get_user_by_username(db, username)
    if not user:
        # Run hash anyway to keep constant time
        pwd_context.dummy_verify()
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user


def create_user(db: Session, username: str, email: str,
                password: str, full_name: str = None,
                is_admin: bool = False) -> User:
    """Create and persist a new user. Raises ValueError on duplicate."""
    if get_user_by_username(db, username):
        raise ValueError(f"Username '{username}' is already registered")
    if get_user_by_email(db, email):
        raise ValueError(f"Email '{email}' is already registered")

    user = User(
        username=username.lower(),
        email=email.lower(),
        full_name=full_name,
        hashed_password=hash_password(password),
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ── FastAPI dependency: get current user from Bearer token ────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency injected into every protected route.
    Decodes JWT → looks up user in DB → validates account is active.
    """
    token_data = decode_token(token)
    user = get_user_by_username(db, token_data.username)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or account deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """Additional dependency — restricts endpoint to admin users only."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user
