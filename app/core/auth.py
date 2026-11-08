"""Authentication and Tenant Isolation provider.

Provides swappable user context resolution (Mock today, Clerk/OIDC JWT tomorrow).
Default resolution falls back to 'root' user to maintain backwards compatibility
with internal scripts and existing test fixtures.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UserContext(BaseModel):
    """Authenticated user context containing identity, role, and tenant isolation keys."""

    user_id: str = Field(
        default="root",
        description="Tenant / user key for database row-level scoping",
    )
    email: str | None = Field(
        default="root@baseagent.ai",
        description="User email address",
    )
    role: str = Field(
        default="root",
        description="User role ('root', 'admin', 'user')",
    )
    is_admin: bool = Field(
        default=True,
        description="True if user has administrative privileges (can manage storage, inspect cross-tenant stats)",
    )


async def get_current_user(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_user_email: Annotated[str | None, Header(alias="X-User-Email")] = None,
    x_user_role: Annotated[str | None, Header(alias="X-User-Role")] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    query_user_id: Annotated[str | None, Query(alias="user_id", description="Fallback user ID for query-based tenant scoping")] = None,
) -> UserContext:
    """Resolves authenticated user context.

    Current implementation:
    - If 'X-User-ID' header is supplied, scopes to that user tenant.
    - If 'Authorization: Bearer mock_<user_id>' is supplied, parses mock token.
    - If 'user_id' query param is supplied (and no header override), scopes to that tenant.
    - If neither is provided, defaults to the 'root' administrator.

    Future swap for Clerk / OIDC:
    - Replace the token parsing logic below with Clerk SDK / PyJWT verification.
    - Downstream endpoint signatures remain completely untouched.
    """
    resolved_id = "root"
    resolved_email = "root@baseagent.ai"
    resolved_role = "root"

    if x_user_id:
        resolved_id = x_user_id.strip()
        resolved_email = x_user_email.strip() if x_user_email else f"{resolved_id}@baseagent.local"
        resolved_role = x_user_role.strip() if x_user_role else ("root" if resolved_id == "root" else "user")
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()
        if token.startswith("mock_"):
            resolved_id = token.replace("mock_", "", 1)
            resolved_email = f"{resolved_id}@baseagent.local"
            resolved_role = "root" if resolved_id == "root" else "user"
    elif query_user_id:
        resolved_id = query_user_id.strip()
        resolved_email = f"{resolved_id}@baseagent.local"
        resolved_role = "root" if resolved_id in ("root", "admin") else "user"

    is_admin = resolved_role in ("root", "admin") or resolved_id in ("root", "admin")

    return UserContext(
        user_id=resolved_id,
        email=resolved_email,
        role=resolved_role,
        is_admin=is_admin,
    )


async def require_admin(
    current_user: Annotated[UserContext, Depends(get_current_user)],
) -> UserContext:
    """Enforces administrative / root privileges.

    Raises:
        HTTPException(403) if the authenticated user is not an administrator.
    """
    if not current_user.is_admin:
        logger.warning(
            "Access denied: user '%s' (role '%s') lacks administrative privileges.",
            current_user.user_id,
            current_user.role,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Access denied: This operation requires administrative privileges. "
                "Regular users must upload and manage assets through scoped endpoints."
            ),
        )
    return current_user
