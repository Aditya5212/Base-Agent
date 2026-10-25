"""Service managing Thread persistence, Root User, and Thread Uploaded Assets."""

import logging
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.thread import Thread, ThreadAsset
from app.models.user import User
from app.services.r2_storage import r2_storage_service

logger = logging.getLogger(__name__)


from fastapi import HTTPException, status


async def get_or_create_user(
    db: AsyncSession,
    user_id: str = "root",
    email: str | None = None,
    name: str | None = None,
) -> User:
    """Ensures a user entity exists in PostgreSQL, auto-creating if absent."""
    user = await db.get(User, user_id)
    if not user:
        user = User(
            id=user_id,
            email=email or f"{user_id}@baseagent.local",
            name=name or user_id.capitalize(),
            role="root" if user_id == "root" else "user",
            metadata_json={"auto_created": True},
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def get_or_create_thread(
    db: AsyncSession,
    thread_id: str,
    user_id: str = "root",
    title: str | None = None,
    is_admin: bool = False,
) -> Thread:
    """Retrieves an existing conversation thread or creates a new one bound to user_id.

    If the thread already exists, verifies ownership so one tenant cannot post into
    another tenant's thread.
    """
    # Ensure user exists first to satisfy foreign key
    await get_or_create_user(db, user_id=user_id)

    thread = await db.get(Thread, thread_id)
    if not thread:
        thread = Thread(
            id=thread_id,
            user_id=user_id,
            title=title or "New Conversation",
            metadata_json={
                "created_by": user_id,
                "assets_count": 0,
            },
        )
        db.add(thread)
        await db.commit()
        await db.refresh(thread)
    elif thread.user_id != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Conversation thread '{thread_id}' belongs to user '{thread.user_id}'.",
        )
    return thread


async def verify_thread_ownership(
    db: AsyncSession,
    thread_id: str,
    user_id: str,
    is_admin: bool = False,
    allow_missing: bool = False,
) -> Thread | None:
    """Verifies that the thread exists and is accessible by user_id.

    Raises:
        HTTPException(404) if the thread does not exist (unless is_admin or allow_missing is True).
        HTTPException(403) if the thread belongs to another user.
    """
    thread = await db.get(Thread, thread_id)
    if not thread:
        if allow_missing or is_admin:
            return None
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation thread '{thread_id}' not found.",
        )
    if thread.user_id != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Conversation thread '{thread_id}' belongs to user '{thread.user_id}'.",
        )
    return thread


async def add_thread_asset(
    db: AsyncSession,
    thread_id: str,
    user_id: str,
    file_name: str,
    r2_key: str,
    mime_type: str,
    file_size: int,
    asset_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ThreadAsset:
    """Registers an uploaded image or document to a specific conversation thread."""
    await get_or_create_thread(db, thread_id=thread_id, user_id=user_id)

    if not asset_type:
        asset_type = "image" if mime_type.startswith("image/") else "document"

    asset = ThreadAsset(
        id=uuid.uuid4(),
        thread_id=thread_id,
        user_id=user_id,
        asset_type=asset_type,
        file_name=file_name,
        r2_key=r2_key,
        mime_type=mime_type,
        file_size=file_size,
        metadata_json=metadata or {},
    )
    db.add(asset)

    # Update thread assets counter and timestamp
    thread = await db.get(Thread, thread_id)
    if thread:
        meta = dict(thread.metadata_json or {})
        meta["assets_count"] = meta.get("assets_count", 0) + 1
        thread.metadata_json = meta

    await db.commit()
    await db.refresh(asset)
    return asset


async def get_thread_assets(db: AsyncSession, thread_id: str) -> list[ThreadAsset]:
    """Retrieves all asset records associated with a thread."""
    stmt = (
        select(ThreadAsset)
        .where(ThreadAsset.thread_id == thread_id)
        .order_by(ThreadAsset.created_at.asc())
    )
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def get_thread_assets_with_signed_urls(
    thread_id: str,
    db: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetches thread assets and generates fresh presigned download URLs for each.

    Can be called with an existing session or will open an isolated session.
    """
    if db is None:
        async with AsyncSessionLocal() as session:
            assets = await get_thread_assets(session, thread_id)
    else:
        assets = await get_thread_assets(db, thread_id)

    results: list[dict[str, Any]] = []
    for asset in assets:
        try:
            url = await r2_storage_service.generate_presigned_url(asset.r2_key)
        except Exception:
            url = r2_storage_service.get_public_url(asset.r2_key)

        results.append({
            "id": str(asset.id),
            "thread_id": asset.thread_id,
            "user_id": asset.user_id,
            "asset_type": asset.asset_type,
            "file_name": asset.file_name,
            "mime_type": asset.mime_type,
            "file_size": asset.file_size,
            "r2_key": asset.r2_key,
            "presigned_url": url,
            "created_at": asset.created_at.isoformat() if asset.created_at else None,
            "metadata": asset.metadata_json,
        })

    return results


def format_thread_assets_context(assets: list[dict[str, Any]]) -> str:
    """Formats structured markdown context detailing uploaded images and documents for the LLM."""
    if not assets:
        return ""

    lines = [
        "=== Uploaded Thread Assets & Media Files ===",
        "The user has uploaded the following files/images to this active conversation thread:",
    ]
    for a in assets:
        type_tag = a.get("asset_type", "file").upper()
        name = a.get("file_name", "unnamed")
        url = a.get("presigned_url") or "URL not available"
        lines.append(f"- [{type_tag}] {name} (Signed URL: {url})")

    lines.append(
        "You can reference these uploaded files directly, describe images if asked, "
        "or provide the signed URLs when referring to user-uploaded files."
    )
    lines.append("============================================")
    return "\n".join(lines)
