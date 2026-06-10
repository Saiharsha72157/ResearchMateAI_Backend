import os
import logging
import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Security scheme for extract token
security_scheme = HTTPBearer(auto_error=False)

# Load Supabase JWT Secret from environment variables
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

if not SUPABASE_JWT_SECRET:
    logger.warning(
        "[AUTH] SUPABASE_JWT_SECRET is missing from the environment variables. "
        "JWT token verification will be bypassed in development mode (returning mock user)."
    )

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)) -> Dict[str, Any]:
    """
    Dependency injection for FastAPI endpoints to authenticate requests via Supabase JWT.
    
    Returns a dictionary containing user information:
    {
        "user_id": str,
        "email": str,
        "role": str
    }
    """
    # 1. Strict JWT Secret Configuration Check
    if not SUPABASE_JWT_SECRET:
        logger.error("[AUTH] Critical configuration error: SUPABASE_JWT_SECRET is missing.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error. Authentication cannot be processed."
        )

    # 2. Require credentials if Secret is configured
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication credentials are required."
        )

    token = credentials.credentials
    try:
        # 3. Decode and verify signature of Supabase JWT
        # Supabase JWT signature uses HS256 algorithm and sets audience to 'authenticated'
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated"
        )
        
        user_id = payload.get("sub")
        email = payload.get("email")
        role = payload.get("role", "authenticated")
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload: Subject (sub) is missing."
            )
            
        return {
            "user_id": user_id,
            "email": email,
            "role": role,
            "is_dev": False
        }
        
    except jwt.ExpiredSignatureError as e:
        logger.error(f"[AUTH] Expired token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Your session has expired. Please log in again."
        )
    except jwt.InvalidSignatureError as e:
        logger.error(f"[AUTH] Invalid signature: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication signature is invalid."
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"[AUTH] Invalid token structure: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token."
        )
    except Exception as e:
        logger.error(f"[AUTH] Unexpected authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )
