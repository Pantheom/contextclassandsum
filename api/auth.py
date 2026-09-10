"""
api/auth.py
-----------
X-API-Key header authentication.

Accepted keys are read from the API_KEY environment variable (or .env file).
Multiple keys are supported via a comma-separated list, e.g.:
    API_KEY=key-one,key-two,key-three

This allows you to issue separate keys to different callers (e.g. the backend
engineer's service gets their own key) without rotating everyone else's key.

On AWS: set API_KEY in your ECS Task Definition environment variables, or
store it in AWS Secrets Manager and inject it at container startup.

Usage in api/main.py — add to any endpoint's dependencies list:
    dependencies=[Depends(_require_models_ready), Depends(require_auth)]
"""
from __future__ import annotations

import os

from fastapi import Request
from fastapi import HTTPException
from fastapi.security import APIKeyHeader

# ---------------------------------------------------------------------------
# Key loading — read once at import time (after load_dotenv() in main.py)
# ---------------------------------------------------------------------------

_raw = os.environ.get("API_KEY", "")
_ACCEPTED_KEYS: frozenset[str] = frozenset(
    k.strip() for k in _raw.split(",") if k.strip()
)

# FastAPI's built-in header extractor — shows a lock icon on /docs and
# automatically adds "X-API-Key" to the OpenAPI security scheme.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def require_auth(request: Request) -> None:
    """FastAPI dependency — enforces X-API-Key header authentication.

    Raises HTTP 401 if:
      - API_KEY env var is not set (server misconfiguration)
      - The X-API-Key header is missing from the request
      - The provided key is not in the accepted keys list

    Add to any endpoint via:
        dependencies=[Depends(_require_models_ready), Depends(require_auth)]
    """
    # Guard: if API_KEY was never configured, fail loudly rather than
    # silently accepting all requests.
    if not _ACCEPTED_KEYS:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: API_KEY environment variable is not set.",
        )

    provided_key = request.headers.get("X-API-Key", "")

    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header.",
        )

    if provided_key not in _ACCEPTED_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )
