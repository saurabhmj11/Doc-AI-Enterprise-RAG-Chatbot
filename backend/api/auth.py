"""
Authentication Router

Endpoints for login, token exchange, profile management, and user admin.
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.schemas import (
    Token,
    UserProfileResponse,
    UserProfileUpdate,
    UserCreateRequest,
)
from core.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    verify_password,
    get_current_active_user,
    require_role,
    create_user,
    update_profile,
    get_all_users,
    _load_users,
)

router = APIRouter()


# =====================================================
# TOKEN ENDPOINTS
# =====================================================

def _authenticate(username: str, password: str):
    user = _load_users().get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


@router.post("/login", response_model=Token, summary="Get access token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login.  Returns a JWT bearer token.
    """
    user = _authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


# =====================================================
# PROFILE ENDPOINTS
# =====================================================

@router.get("/me", response_model=UserProfileResponse, summary="My profile")
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """Return the authenticated user's profile (no password hash)."""
    return {k: v for k, v in current_user.items() if k != "hashed_password"}


@router.patch("/me", response_model=UserProfileResponse, summary="Update my profile")
async def update_my_profile(
    body: UserProfileUpdate,
    current_user: dict = Depends(get_current_active_user),
):
    """
    Update designation, department, full_name, or email.
    Password changes are not supported here (use a dedicated endpoint if needed).
    """
    try:
        updated = update_profile(
            current_user["username"],
            full_name=body.full_name,
            email=body.email,
            department=body.department,
            designation=body.designation,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return updated


# =====================================================
# ADMIN — USER MANAGEMENT  (role: admin only)
# =====================================================

@router.get(
    "/users",
    response_model=list[UserProfileResponse],
    summary="List all users (admin)",
    dependencies=[Depends(require_role("admin"))],
)
async def list_users():
    """Return all registered users without password hashes. Requires admin role."""
    return get_all_users()


@router.post(
    "/users",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user (admin)",
    dependencies=[Depends(require_role("admin"))],
)
async def create_new_user(body: UserCreateRequest):
    """
    Create a new user with a designation and department.
    Requires admin role.
    """
    try:
        return create_user(
            username=body.username,
            password=body.password,
            full_name=body.full_name,
            email=body.email,
            department=body.department,
            designation=body.designation,
            role=body.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
