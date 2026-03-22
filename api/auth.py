"""
auth.py
───────
Clerk JWT authentication for HealthGuard-XAI.
Verifies tokens using Clerk's JWKS and SDK.
Syncs users to SQLite automatically.
"""

from __future__ import annotations

import os
import json
import urllib.request
from typing import Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from clerk_backend_api import Clerk

from api.database import get_db

# Load env variables including the new .env.local
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"), override=True)
except ImportError:
    pass

CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
clerk_client = Clerk(bearer_auth=CLERK_SECRET_KEY) if CLERK_SECRET_KEY else None

_bearer_scheme = HTTPBearer(auto_error=False)
_jwks_cache = {}

def get_jwks(issuer: str):
    """Fetch the JWKS from Clerk to verify the JWT."""
    if issuer in _jwks_cache:
        return _jwks_cache[issuer]
    try:
        url = f"{issuer}/.well-known/jwks.json"
        req = urllib.request.Request(url, headers={'User-Agent': 'HealthGuard-XAI'})
        # Add a 5 second timeout to prevent infinite hanging
        with urllib.request.urlopen(req, timeout=5) as response:
            jwks = json.loads(response.read().decode())
            _jwks_cache[issuer] = jwks
            return jwks
    except Exception as e:
        print(f"Failed to fetch JWKS: {e}")
        return None

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """
    FastAPI dependency: extracts and validates the Clerk Bearer token.
    Automatically provisions new users into SQLite.
    Returns the user record dict or raises 401.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials
    try:
        # Get unverified claims just to find the issuer
        unverified_claims = jwt.get_unverified_claims(token)
        issuer = unverified_claims.get("iss")
        if not issuer:
            raise HTTPException(status_code=401, detail="Token missing issuer")

        jwks = get_jwks(issuer)
        if not jwks:
            raise HTTPException(status_code=401, detail="Could not retrieve JWKS for issuer")

        # Verify the signature against the JWKS
        payload = jwt.decode(token, jwks, algorithms=["RS256"], options={"verify_aud": False})
        clerk_id = payload.get("sub")
        if not clerk_id:
            raise HTTPException(status_code=401, detail="Invalid token payload missing subject")

        # Database Sync Logic
        try:
            with get_db() as conn:
                row = conn.execute("SELECT * FROM users WHERE clerk_id = ?", (clerk_id,)).fetchone()
                
                if not row:
                    if not clerk_client:
                        raise ValueError("Backend CLERK_SECRET_KEY missing. Check your .env.local file.")
                    
                    # Fetch fresh profile from Clerk using keyword arguments natively required by v2 SDK
                    clerk_user = clerk_client.users.get(user_id=clerk_id)
                    
                    # Safely extract email and name, handling Clerk's object model
                    email = "unknown@clerk.local"
                    if getattr(clerk_user, "email_addresses", None) and len(clerk_user.email_addresses) > 0:
                        email = clerk_user.email_addresses[0].email_address
                    elif getattr(clerk_user, "email", None):
                        email = clerk_user.email

                    first_name = getattr(clerk_user, "first_name", "Guest") or "Guest"
                    last_name = getattr(clerk_user, "last_name", "") or ""

                    conn.execute(
                        "INSERT INTO users (clerk_id, email, first_name, last_name) VALUES (?, ?, ?, ?)",
                        (clerk_id, email, first_name, last_name)
                    )
                    conn.commit()
                    row = conn.execute("SELECT * FROM users WHERE clerk_id = ?", (clerk_id,)).fetchone()

            return dict(row)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Developer Error Syncing User: {type(e).__name__} - {str(e)}")

    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")
