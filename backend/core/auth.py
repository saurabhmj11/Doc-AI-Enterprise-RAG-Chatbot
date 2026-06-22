"""
Authentication Module

JWT-based security for FastAPI endpoints.
Handles password hashing, token creation, and dependency-based auth.

User profiles support: designation, department, and role (system access level).
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from jwt.exceptions import PyJWTError as JWTError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# =====================================================
# SECURITY CONSTANTS
# =====================================================

# SECRET_KEY MUST be set in .env — no insecure fallback in production
SECRET_KEY: str = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "SECRET_KEY not set in environment — generated a temporary key. "
        "All existing JWTs will be invalidated on restart. "
        "Set SECRET_KEY in your .env file for production."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# =====================================================
# USER STORE — JSON file backed, multi-user
# =====================================================

_USERS_FILE = Path(os.getenv("USERS_FILE", "./users.json"))


def _load_users() -> dict:
    """Load user registry from JSON file."""
    if not _USERS_FILE.exists():
        return {}
    try:
        with open(_USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load users file: {exc}")
        return {}


def _save_users(users: dict) -> None:
    """Persist user registry to JSON file."""
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _ensure_default_admin() -> None:
    """
    Bootstrap a default admin user if the user store is empty.
    Credentials are read from env vars — never hardcoded.
    """
    users = _load_users()
    if users:
        return  # already seeded

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "")

    if not admin_password:
        import secrets
        admin_password = secrets.token_urlsafe(16)
        logger.warning(
            f"ADMIN_PASSWORD not set — generated a temporary password: {admin_password!r}. "
            "Set ADMIN_PASSWORD in your .env file."
        )

    users[admin_username] = {
        "username": admin_username,
        "full_name": os.getenv("ADMIN_FULL_NAME", "System Administrator"),
        "email": os.getenv("ADMIN_EMAIL", f"{admin_username}@example.com"),
        "department": os.getenv("ADMIN_DEPARTMENT", "Platform Engineering"),
        "designation": os.getenv("ADMIN_DESIGNATION", "AI/ML Engineer"),
        "role": "admin",
        "hashed_password": get_password_hash(admin_password),
        "disabled": False,
    }
    _save_users(users)
    logger.info(f"Default admin user '{admin_username}' created in {_USERS_FILE}")


# =====================================================
# PASSWORD UTILITIES
# =====================================================

def get_password_hash(password: str) -> str:
    """Hash a plain-text password using bcrypt (rounds=12 for production strength)."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


# =====================================================
# TOKEN UTILITIES
# =====================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT with an expiry."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# =====================================================
# AUTH DEPENDENCIES (FastAPI)
# =====================================================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Decode JWT and return the user record."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    users = _load_users()
    user = users.get(username)
    if not user:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Ensure the authenticated user is not disabled."""
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_role(*roles: str):
    """
    Dependency factory that enforces a minimum role.

    Usage:
        @router.delete("/docs/{id}", dependencies=[Depends(require_role("admin", "editor"))])
    """
    async def _check(current_user: dict = Depends(get_current_active_user)):
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {roles}",
            )
        return current_user
    return _check


# =====================================================
# USER MANAGEMENT HELPERS
# =====================================================

def get_all_users() -> list[dict]:
    """Return all users (hashed_password excluded)."""
    return [
        {k: v for k, v in u.items() if k != "hashed_password"}
        for u in _load_users().values()
    ]


def create_user(
    username: str,
    password: str,
    full_name: str,
    email: str,
    department: str,
    designation: str,
    role: str = "viewer",
) -> dict:
    """
    Create a new user and persist to the user store.

    Roles: admin | editor | viewer
    """
    users = _load_users()
    if username in users:
        raise ValueError(f"User '{username}' already exists")

    valid_roles = {"admin", "editor", "viewer"}
    if role not in valid_roles:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {valid_roles}")

    user = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "department": department,
        "designation": designation,
        "role": role,
        "hashed_password": get_password_hash(password),
        "disabled": False,
    }
    users[username] = user
    _save_users(users)
    logger.info(f"User '{username}' created (role={role}, designation={designation})")
    return {k: v for k, v in user.items() if k != "hashed_password"}


def update_profile(
    username: str,
    *,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    department: Optional[str] = None,
    designation: Optional[str] = None,
) -> dict:
    """Update non-sensitive profile fields for a user."""
    users = _load_users()
    if username not in users:
        raise KeyError(f"User '{username}' not found")

    user = users[username]
    if full_name   is not None: user["full_name"]   = full_name
    if email       is not None: user["email"]        = email
    if department  is not None: user["department"]   = department
    if designation is not None: user["designation"]  = designation

    _save_users(users)
    return {k: v for k, v in user.items() if k != "hashed_password"}


# =====================================================
# BOOTSTRAP — seed default admin on first run
# =====================================================
_ensure_default_admin()
