"""Service managing Thread persistence, Root User, and Thread Uploaded Assets."""

import asyncio
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
    check_deleting: bool = True,
) -> Thread:
    """Retrieves an existing conversation thread or creates a new one bound to user_id.

    If the thread already exists, verifies ownership so one tenant cannot post into
    another tenant's thread. Rejects access if the thread is currently being deleted.
    """
    if thread_id.startswith("workflow:"):
        raise HTTPException(status_code=422, detail="Reserved workflow checkpoint namespace")
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
    else:
        if thread.user_id != user_id and not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Conversation thread '{thread_id}' belongs to user '{thread.user_id}'.",
            )
        if check_deleting and (thread.metadata_json or {}).get("status") == "DELETING":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Conversation thread '{thread_id}' is currently being deleted.",
            )
    return thread


async def verify_thread_ownership(
    db: AsyncSession,
    thread_id: str,
    user_id: str,
    is_admin: bool = False,
    allow_missing: bool = False,
    check_deleting: bool = True,
    for_share: bool = False,
    for_update: bool = False,
) -> Thread | None:
    """Verifies that the thread exists and is accessible by user_id.

    Raises:
        HTTPException(404) if the thread does not exist (unless is_admin or allow_missing is True).
        HTTPException(403) if the thread belongs to another user.
        HTTPException(409) if the thread is currently marked DELETING.
    """
    if thread_id.startswith("workflow:"):
        raise HTTPException(status_code=422, detail="Reserved workflow checkpoint namespace")

    thread = None
    if (for_share or for_update) and hasattr(db, "get"):
        opts = {"read": True} if for_share else True
        try:
            thread = await db.get(Thread, thread_id, with_for_update=opts)
        except (TypeError, Exception):
            thread = await db.get(Thread, thread_id)
    else:
        thread = await db.get(Thread, thread_id)

    if not thread:
        if allow_missing or is_admin:
            return None
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation thread '{thread_id}' not found.",
        )
    if hasattr(thread, "user_id") and isinstance(thread.user_id, str) and thread.user_id != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Conversation thread '{thread_id}' belongs to user '{thread.user_id}'.",
        )
    meta = getattr(thread, "metadata_json", None) or {}
    if check_deleting and isinstance(meta, dict) and meta.get("status") == "DELETING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Conversation thread '{thread_id}' is currently being deleted.",
        )
    return thread


async def verify_thread_admission(
    db: AsyncSession,
    thread_id: str,
    user_id: str,
    is_admin: bool = False,
    for_share: bool = True,
) -> Thread:
    """Atomically validates thread accessibility and ensures the thread is not currently DELETING.

    Acquires a PostgreSQL row-level shared lock (`FOR SHARE`) when `for_share=True` to serialize
    with concurrent `delete_thread_cascade` transactions (`FOR UPDATE`).
    """
    thread = await verify_thread_ownership(
        db,
        thread_id=thread_id,
        user_id=user_id,
        is_admin=is_admin,
        allow_missing=False,
        check_deleting=True,
        for_share=for_share,
    )
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation thread '{thread_id}' not found.",
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
    thread = await verify_thread_admission(db, thread_id=thread_id, user_id=user_id, for_share=True)

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
    meta = dict(getattr(thread, "metadata_json", None) or {})
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


async def delete_thread_cascade(db: AsyncSession, thread_id: str, user_id: str) -> bool:
    """Cascade deletes all resources associated with a thread:

    1. Acquires document advisory locks for all thread-scoped documents to coordinate
       with active ingestion tasks. If any document is actively PARSING, marks status FAILED.
    2. Purges vector chunks from PGVector for each thread document.
    3. Purges LightRAG thread workspace entries.
    4. Deletes raw document files from Cloudflare R2.
    5. Deletes thread media asset files from Cloudflare R2.
    6. Purges LangGraph checkpointer memory (checkpoints, checkpoint_blobs, checkpoint_writes).
    7. Verifies that all external resources were deleted cleanly. If any external deletion
       fails, raises RuntimeError and DOES NOT delete the database records, preserving
       them for safe retry.
    8. Deletes thread database record (cascading in SQL to documents, chunks, and assets).
    """
    from app.models.document import Document, DocumentStatus
    from app.services.document_lock import document_lock
    from app.services.memory import delete_thread_memory
    from app.services.vector_store import delete_documents_by_metadata

    # 1. Acquire exclusive FOR UPDATE lock on the Thread record to serialize with concurrent uploads
    try:
        thread = await db.get(Thread, thread_id, with_for_update=True)
    except (TypeError, Exception):
        thread = await db.get(Thread, thread_id)

    if not thread:
        logger.info("Thread '%s' already does not exist; deletion idempotent.", thread_id)
        return True

    # Mark thread as DELETING immediately to block concurrent chat, uploads, and workers
    meta = dict(getattr(thread, "metadata_json", None) or {})
    meta["status"] = "DELETING"
    thread.metadata_json = meta

    cleanup_errors: list[str] = []

    # 2. Enumerate thread-scoped documents while holding the exclusive lock
    doc_stmt = select(Document).where(Document.thread_id == thread_id)
    doc_res = await db.execute(doc_stmt)
    thread_docs = list(doc_res.scalars().all())

    # 3. Enumerate thread assets while holding the exclusive lock
    asset_stmt = select(ThreadAsset).where(ThreadAsset.thread_id == thread_id)
    asset_res = await db.execute(asset_stmt)
    thread_assets = list(asset_res.scalars().all())

    # Commit status = DELETING and release row lock now that all documents and assets are enumerated
    await db.commit()

    for doc in thread_docs:
        doc_id_str = str(doc.id)

        # Abort active ingestion if document is currently PARSING
        if doc.status == DocumentStatus.PARSING:
            doc.status = DocumentStatus.FAILED
            doc.error_message = "Aborted: Thread deleted during active ingestion."
            await db.commit()

        # Coordinate with document_lock (retry up to 3 seconds if locked by active task)
        acquired = False
        for _ in range(6):
            async with document_lock(doc_id_str) as has_lock:
                if has_lock:
                    acquired = True
                    # A. Purge vector store chunks
                    try:
                        await delete_documents_by_metadata("document_id", doc_id_str)
                    except Exception as exc:
                        logger.error("Failed to purge vector chunks for thread doc %s: %s", doc_id_str, exc)
                        cleanup_errors.append(f"Vector chunks for doc {doc_id_str}: {exc}")

                    # B. Purge LightRAG entries (treat failure as cleanup error)
                    try:
                        from app.core.rag.factory import get_provider
                        from app.core.rag.workspace import thread_workspace
                        lightrag_provider = await get_provider("lightrag", thread_workspace(user_id, thread_id))
                        await lightrag_provider.delete_document(doc_id_str)
                    except Exception as exc:
                        logger.error("LightRAG thread cleanup failed for doc %s: %s", doc_id_str, exc)
                        cleanup_errors.append(f"LightRAG graph cleanup for doc {doc_id_str}: {exc}")

                    # C. Delete raw document from R2
                    if doc.r2_key:
                        try:
                            await r2_storage_service.delete_file(doc.r2_key)
                        except Exception as exc:
                            logger.error("Failed to delete R2 file for thread doc %s: %s", doc_id_str, exc)
                            cleanup_errors.append(f"R2 file for doc {doc_id_str} ({doc.r2_key}): {exc}")
                    break
            await asyncio.sleep(0.5)

        if not acquired:
            logger.warning("Could not acquire lock for document %s during thread deletion; lock is busy", doc_id_str)
            cleanup_errors.append(f"Document {doc_id_str} is busy in active ingestion; lock could not be acquired")

    # 3. Delete thread media assets from R2
    for asset in thread_assets:
        if asset.r2_key:
            try:
                await r2_storage_service.delete_file(asset.r2_key)
            except Exception as exc:
                logger.error("Failed to delete R2 file for thread asset %s: %s", asset.id, exc)
                cleanup_errors.append(f"R2 asset {asset.id} ({asset.r2_key}): {exc}")

    # 4. Purge checkpointer memory (verify boolean return value)
    try:
        mem_ok = await delete_thread_memory(thread_id)
        if not mem_ok:
            cleanup_errors.append(f"Checkpointer memory deletion failed for thread {thread_id}")
    except Exception as exc:
        logger.error("Failed to delete thread memory for thread %s: %s", thread_id, exc)
        cleanup_errors.append(f"Thread memory: {exc}")

    # 5. Check for cleanup failures. If any occurred, DO NOT delete DB record!
    if cleanup_errors:
        err_detail = "; ".join(cleanup_errors)
        logger.error(
            "Cascade deletion of thread %s encountered errors. DB records preserved for retry: %s",
            thread_id,
            err_detail,
        )
        raise RuntimeError(f"Thread cleanup failed ({err_detail}). Records preserved for retry.")

    # 6. Delete thread record from DB (cascades to remaining DB records)
    await db.delete(thread)
    await db.commit()
    logger.info("Thread %s and all associated assets/documents cleanly deleted.", thread_id)
    return True
