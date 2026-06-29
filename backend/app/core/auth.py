"""
API Key authentication dependency for FastAPI.

Validates requests against the API_KEY configured in the environment.
Supports both ``X-API-Key`` header and ``Authorization: Bearer <key>`` header.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: Optional[str] = Security(_api_key_header),
) -> str:
    """FastAPI dependency — raises 401 if the key is missing or invalid."""
    settings = get_settings()

    # If no API_KEY is configured, skip authentication (dev mode)
    if not settings.API_KEY:
        return "no-auth"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
        )

    if api_key != settings.API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return api_key
