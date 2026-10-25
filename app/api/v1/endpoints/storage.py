"""Cloudflare R2 / S3 Object Storage Management API endpoints.

Admin-only endpoints: Raw object storage operations are restricted to root/admin administrators.
Regular users must manage documents and media via /documents and /chat/{thread_id}/assets.
"""

import io
import logging
import re
import uuid
from typing import Annotated
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.schemas.storage import (
    PresignedUrlRequest,
    PresignedUrlResponse,
    StorageDeleteResponse,
    StorageObjectDetailsResponse,
    StorageObjectItem,
    StorageObjectListResponse,
    StorageStatsResponse,
    StorageUploadResponse,
    format_bytes,
)
from app.services.r2_storage import r2_storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["Object Storage Management (R2 / S3)"])

# 50MB max upload size for generic storage objects
MAX_STORAGE_UPLOAD_SIZE = 50 * 1024 * 1024


def verify_admin_access(user_id: str = "root") -> None:
    """Restricts raw bucket storage operations to root administrator users."""
    if user_id != "root":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Access denied: Raw storage bucket operations are restricted to root administrator. "
                "Regular users must upload and manage assets via /api/v1/documents or /api/v1/chat/{thread_id}/assets."
            ),
        )


def sanitize_and_validate_key(key: str) -> str:
    """Validates and sanitizes a storage key, strictly preventing path traversal and malformed inputs."""
    if not key or not isinstance(key, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid storage key: Storage key cannot be empty.",
        )

    # Reject path traversal and control sequences
    if ".." in key or "//" in key or "\\" in key or key.startswith(("/", "\\")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid storage key: Path traversal sequences ('..', leading slashes, backslashes) are forbidden.",
        )

    cleaned = key.strip("/")
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid storage key: Key resolves to an empty path.",
        )
    return cleaned


def sanitize_filename(filename: str) -> str:
    """Cleans filename of non-alphanumeric special characters while preserving extension."""
    clean = re.sub(r"[^\w\.\-]", "_", filename).strip("_")
    return clean or "uploaded_file"


@router.get(
    "/stats",
    response_model=StorageStatsResponse,
    summary="Get object storage statistics (Admin Only)",
    description="Returns bucket configuration, region, endpoint, total file count, and aggregated storage consumption.",
)
async def get_storage_stats(
    prefix: str = Query(default="", description="Optional prefix to filter storage calculations"),
    user_id: str = Query(default="root", description="Admin user identifier"),
) -> StorageStatsResponse:
    """Calculates and returns aggregated storage metrics from Cloudflare R2."""
    verify_admin_access(user_id)
    clean_prefix = sanitize_and_validate_key(prefix) if prefix else ""
    try:
        raw_stats = await r2_storage_service.get_storage_stats(prefix=clean_prefix)
        return StorageStatsResponse(
            configured=raw_stats["configured"],
            bucket_name=raw_stats["bucket_name"],
            region=raw_stats["region"],
            endpoint_url=raw_stats["endpoint_url"],
            total_files=raw_stats["total_files"],
            total_bytes=raw_stats["total_bytes"],
            formatted_storage=format_bytes(raw_stats["total_bytes"]),
        )
    except Exception as exc:
        logger.error("Failed to fetch storage stats: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch storage metrics: {exc}",
        )


@router.get(
    "/objects",
    response_model=StorageObjectListResponse,
    summary="List objects in storage bucket (Admin Only)",
    description="Lists stored files matching an optional prefix with sizes, ETags, timestamps, and presigned URLs.",
)
async def list_storage_objects(
    prefix: str = Query(default="", description="Object key prefix filter (e.g. 'users/root/', 'uploads/')"),
    user_id: str = Query(default="root", description="Admin user identifier"),
    max_keys: int = Query(default=100, ge=1, le=1000, description="Maximum number of items per page"),
    continuation_token: str | None = Query(default=None, description="Token for fetching next page of results"),
) -> StorageObjectListResponse:
    """Lists objects in the Cloudflare R2 bucket for admin inspection."""
    verify_admin_access(user_id)
    clean_prefix = sanitize_and_validate_key(prefix) if prefix else ""
    try:
        data = await r2_storage_service.list_objects(
            prefix=clean_prefix,
            max_keys=max_keys,
            continuation_token=continuation_token,
        )

        items = [
            StorageObjectItem(
                key=obj["key"],
                size=obj["size"],
                formatted_size=format_bytes(obj["size"]),
                last_modified=obj["last_modified"],
                etag=obj["etag"],
                presigned_url=obj["presigned_url"],
                public_url=obj["public_url"],
            )
            for obj in data.get("objects", [])
        ]

        return StorageObjectListResponse(
            prefix=clean_prefix,
            total_count=len(items),
            is_truncated=data.get("is_truncated", False),
            next_continuation_token=data.get("next_continuation_token"),
            objects=items,
        )
    except Exception as exc:
        logger.error("Failed to list storage objects: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list objects: {exc}",
        )


@router.post(
    "/upload",
    response_model=StorageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload any file to object storage (Admin Only)",
    description="Uploads an asset to Cloudflare R2 under a deterministic admin-scoped storage key.",
)
async def upload_storage_object(
    file: UploadFile = File(..., description="File payload to upload"),
    prefix: str = Form(default="uploads", description="Folder/prefix in bucket (e.g. 'uploads', 'assets')"),
    user_id: str = Form(default="root", description="Admin user identifier"),
) -> StorageUploadResponse:
    """Uploads a file directly to Cloudflare R2 under the administrator namespace."""
    verify_admin_access(user_id)

    contents = await file.read()
    file_size = len(contents)

    if file_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if file_size > MAX_STORAGE_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_STORAGE_UPLOAD_SIZE // (1024 * 1024)}MB.",
        )

    raw_file_name = file.filename or "uploaded_file"
    file_name = sanitize_filename(raw_file_name)
    content_type = file.content_type or "application/octet-stream"

    clean_prefix = sanitize_and_validate_key(prefix) if prefix else "uploads"
    file_uuid = uuid.uuid4()
    # Server generates deterministic key: users/{user_id}/{prefix}/{uuid}_{filename}
    storage_key = f"users/{user_id}/{clean_prefix}/{file_uuid}_{file_name}"

    try:
        stored_key = await r2_storage_service.upload_file(
            file_data=contents,
            file_key=storage_key,
            content_type=content_type,
        )
        presigned_url = await r2_storage_service.generate_presigned_url(stored_key)
        public_url = r2_storage_service.get_public_url(stored_key)

        return StorageUploadResponse(
            key=stored_key,
            file_name=file_name,
            size=file_size,
            formatted_size=format_bytes(file_size),
            content_type=content_type,
            presigned_url=presigned_url,
            public_url=public_url,
            message="File uploaded successfully to Cloudflare R2.",
        )
    except Exception as exc:
        logger.error("Failed to upload object '%s': %s", storage_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        )


@router.get(
    "/objects/{file_key:path}",
    response_model=StorageObjectDetailsResponse,
    summary="Get object metadata (Admin Only)",
    description="Retrieves metadata, content type, ETag, and presigned download URL for a specific file key.",
)
async def get_storage_object_metadata(
    file_key: str,
    user_id: str = Query(default="root", description="Admin user identifier"),
) -> StorageObjectDetailsResponse:
    """Fetches metadata for an object in Cloudflare R2."""
    verify_admin_access(user_id)
    clean_key = sanitize_and_validate_key(file_key)
    try:
        meta = await r2_storage_service.get_object_metadata(clean_key)
        return StorageObjectDetailsResponse(
            key=meta["key"],
            size=meta["size"],
            formatted_size=format_bytes(meta["size"]),
            content_type=meta["content_type"],
            last_modified=meta.get("last_modified"),
            etag=meta.get("etag"),
            presigned_url=meta["presigned_url"],
            public_url=meta.get("public_url"),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Object '{clean_key}' not found.")
    except Exception as exc:
        logger.error("Failed to fetch metadata for '%s': %s", clean_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve object metadata: {exc}",
        )


@router.get(
    "/download/{file_key:path}",
    summary="Download object (Admin Only)",
    description="Downloads raw binary bytes of a stored file directly from Cloudflare R2.",
)
async def download_storage_object(
    file_key: str,
    user_id: str = Query(default="root", description="Admin user identifier"),
    inline: bool = Query(default=False, description="Preview inline in browser instead of attachment download"),
):
    """Streams the stored file directly to the admin client."""
    verify_admin_access(user_id)
    clean_key = sanitize_and_validate_key(file_key)
    try:
        data = await r2_storage_service.download_file(clean_key)
        meta = await r2_storage_service.get_object_metadata(clean_key)
        content_type = meta.get("content_type", "application/octet-stream")
        file_name = clean_key.split("/")[-1]
        disposition = "inline" if inline else "attachment"

        return StreamingResponse(
            io.BytesIO(data),
            media_type=content_type,
            headers={
                "Content-Disposition": f'{disposition}; filename="{file_name}"',
                "Content-Length": str(len(data)),
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Object '{clean_key}' not found.")
    except Exception as exc:
        logger.error("Download failed for '%s': %s", clean_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {exc}",
        )


@router.delete(
    "/objects/{file_key:path}",
    response_model=StorageDeleteResponse,
    summary="Delete object from storage (Admin Only)",
    description="Deletes a raw object key from Cloudflare R2.",
)
async def delete_storage_object(
    file_key: str,
    user_id: str = Query(default="root", description="Admin user identifier"),
) -> StorageDeleteResponse:
    """Deletes an object from Cloudflare R2."""
    verify_admin_access(user_id)
    clean_key = sanitize_and_validate_key(file_key)
    try:
        deleted = await r2_storage_service.delete_file(clean_key)
        return StorageDeleteResponse(
            key=clean_key,
            deleted=deleted,
            message=f"Object '{clean_key}' successfully deleted from storage.",
        )
    except Exception as exc:
        logger.error("Failed to delete object '%s': %s", clean_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {exc}",
        )


@router.post(
    "/presigned-url",
    response_model=PresignedUrlResponse,
    summary="Generate presigned S3 URL (Admin Only)",
    description="Generates a temporary signed URL for direct client-to-R2 upload (put_object) or download (get_object).",
)
async def create_presigned_url(
    request: PresignedUrlRequest,
    user_id: str = Query(default="root", description="Admin user identifier"),
) -> PresignedUrlResponse:
    """Generates direct presigned S3/R2 URLs for admin operations."""
    verify_admin_access(user_id)
    clean_key = sanitize_and_validate_key(request.file_key)
    try:
        if request.operation == "put_object":
            url = await r2_storage_service.generate_presigned_upload_url(
                file_key=clean_key,
                content_type=request.content_type,
                expires_in=request.expires_in,
            )
        else:
            url = await r2_storage_service.generate_presigned_url(
                file_key=clean_key,
                expires_in=request.expires_in,
            )

        return PresignedUrlResponse(
            url=url,
            file_key=clean_key,
            operation=request.operation,
            expires_in=request.expires_in,
        )
    except Exception as exc:
        logger.error("Presigned URL generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate presigned URL: {exc}",
        )
