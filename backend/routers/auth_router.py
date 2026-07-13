"""
routers/auth_router.py
──────────────────────
Authentication endpoints: sync and profile.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel


from backend.auth import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

class UserProfile(BaseModel):
    id:          int
    clerk_id:    str
    email:       str
    first_name:  str
    last_name:   str
    role:        str
    created_at:  str

@router.get("/me", response_model=UserProfile)
def get_profile(current_user=Depends(get_current_user)):
    """Return the current authenticated user's profile from the local database."""
    return UserProfile(
        id=current_user["id"],
        clerk_id=current_user["clerk_id"],
        email=current_user["email"],
        first_name=current_user["first_name"] or "",
        last_name=current_user["last_name"] or "",
        role=current_user["role"],
        created_at=str(current_user["created_at"]),
    )
