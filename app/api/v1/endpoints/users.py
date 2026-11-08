"""API endpoints for User account management, profile inspection, and cascade deletion."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import UserContext, get_current_user, require_admin
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.models.user import User
from app.schemas.user import (
    UserCreate,
    UserDeleteResponse,
    UserListResponse,
    UserResponse,
    UserStats,
    UserUpdate,
)
from app.services import user_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["User Management"])


def _to_user_response(user: User, stats: UserStats | None = None) -> UserResponse:
    """Helper to convert ORM User model and optional stats into UserResponse."""
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        metadata_json=user.metadata_json,
        created_at=user.created_at,
        updated_at=user.updated_at,
        stats=stats,
    )


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
    description="Creates a new user profile and automatically provisions a default assistant agent.",
)
@limiter.limit("20/minute")
async def create_user_endpoint(
    request: Request,
    user_in: UserCreate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Registers a new user account. Only administrators can assign 'admin' or 'root' roles."""
    if user_in.id == "root":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot register or create the 'root' administrator account via this endpoint.",
        )

    if user_in.role in ("admin", "root") and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can create users with elevated roles ('admin', 'root').",
        )

    user = await user_service.create_user(db, user_in=user_in)
    return _to_user_response(user)


@router.get(
    "",
    response_model=UserListResponse,
    summary="List user accounts",
    description="Retrieves a paginated list of user accounts with optional search and filters. Requires admin privileges.",
)
async def list_users_endpoint(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search by user ID, email, or name"),
    role: str | None = Query(None, description="Filter by role ('user', 'admin', 'root')"),
    is_active: bool | None = Query(None, description="Filter by active status"),
    include_stats: bool = Query(False, description="Include counts of owned resources"),
    current_user: UserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UserListResponse:
    """Lists all user accounts in the platform."""
    items, total = await user_service.list_users(
        session=db,
        page=page,
        page_size=page_size,
        search=search,
        role=role,
        is_active=is_active,
        include_stats=include_stats,
    )
    return UserListResponse(
        items=[_to_user_response(u, stats) for u, stats in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Returns the profile and resource statistics for the currently authenticated user session.",
)
async def get_me_endpoint(
    include_stats: bool = Query(True, description="Include entity counts"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Returns profile for current session user."""
    user, stats = await user_service.get_user_by_id(
        session=db,
        user_id=current_user.user_id,
        include_stats=include_stats,
    )
    return _to_user_response(user, stats)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user account details",
    description="Retrieves profile and optional stats for a user. Non-admins may only view their own profile.",
)
async def get_user_endpoint(
    user_id: str,
    include_stats: bool = Query(False, description="Include entity counts"),
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Retrieves account details for the specified user ID."""
    if current_user.user_id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only view your own account profile.",
        )

    user, stats = await user_service.get_user_by_id(
        session=db,
        user_id=user_id,
        include_stats=include_stats,
    )
    return _to_user_response(user, stats)


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user account",
    description="Modifies user profile fields. Role escalation requires administrative privileges.",
)
@limiter.limit("20/minute")
async def update_user_endpoint(
    request: Request,
    user_id: str,
    user_update: UserUpdate,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Updates the specified user profile."""
    if current_user.user_id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only update your own account profile.",
        )

    user = await user_service.update_user(
        session=db,
        user_id=user_id,
        user_update=user_update,
        is_admin=current_user.is_admin,
    )
    return _to_user_response(user)


@router.delete(
    "/{user_id}",
    response_model=UserDeleteResponse,
    summary="Cascade delete user and all associated data",
    description="Permanently deletes the user and cascades across database records, R2 storage files, vector chunks, and memory checkpoints.",
)
async def delete_user_endpoint(
    user_id: str,
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserDeleteResponse:
    """Permanently deletes a user and cascades all associated data."""
    if current_user.user_id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only delete your own account.",
        )

    counts = await user_service.delete_user(session=db, user_id=user_id)
    return UserDeleteResponse(
        id=user_id,
        message=f"User '{user_id}' and all associated data successfully deleted.",
        deleted_counts=counts,
    )
