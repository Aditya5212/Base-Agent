"""Cloudflare R2 / S3 Object Storage Management API endpoints."""

import io
import logging
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


@router.get(
    "/stats",
    response_model=StorageStatsResponse,
    summary="Get object storage statistics",
    description="Returns bucket configuration, region, endpoint, total file count, and aggregated storage consumption.",
)
async def get_storage_stats(
    prefix: str = Query(default="", description="Optional prefix to filter storage calculations"),
) -> StorageStatsResponse:
    """Calculates and returns aggregated storage metrics from Cloudflare R2."""
    try:
        raw_stats = await r2_storage_service.get_storage_stats(prefix=prefix)
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
    summary="List objects in storage bucket",
    description="Lists stored files matching an optional prefix with sizes, ETags, timestamps, and presigned URLs.",
)
async def list_storage_objects(
    prefix: str = Query(default="", description="Object key prefix filter (e.g. 'documents/', 'uploads/')"),
    max_keys: int = Query(default=100, ge=1, le=1000, description="Maximum number of items per page"),
    continuation_token: str | None = Query(default=None, description="Token for fetching next page of results"),
) -> StorageObjectListResponse:
    """Lists objects in the Cloudflare R2 bucket."""
    try:
        data = await r2_storage_service.list_objects(
            prefix=prefix,
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
            prefix=prefix,
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
    summary="Upload any file to object storage",
    description="Directly uploads an asset, document, or attachment to Cloudflare R2.",
)
async def upload_storage_object(
    file: UploadFile = File(..., description="File payload to upload"),
    prefix: str = Form(default="uploads", description="Folder/prefix in bucket (e.g. 'uploads', 'assets')"),
    custom_key: str | None = Form(default=None, description="Optional custom object key override"),
) -> StorageUploadResponse:
    """Uploads a file directly to Cloudflare R2."""
    contents = await file.read()
    file_size = len(contents)

    if file_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if file_size > MAX_STORAGE_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_STORAGE_UPLOAD_SIZE // (1024 * 1024)}MB.",
        )

    file_name = file.filename or "uploaded_file"
    content_type = file.content_type or "application/octet-stream"

    # Compute key
    if custom_key:
        storage_key = custom_key.lstrip("/")
    else:
        clean_prefix = prefix.strip("/")
        storage_key = f"{clean_prefix}/{file_name}" if clean_prefix else file_name

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
    summary="Get object metadata",
    description="Retrieves metadata, content type, ETag, and presigned download URL for a specific file key.",
)
async def get_storage_object_metadata(file_key: str) -> StorageObjectDetailsResponse:
    """Fetches metadata for an object in Cloudflare R2."""
    try:
        meta = await r2_storage_service.get_object_metadata(file_key)
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Object '{file_key}' not found.")
    except Exception as exc:
        logger.error("Failed to fetch metadata for '%s': %s", file_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve object metadata: {exc}",
        )


@router.get(
    "/download/{file_key:path}",
    summary="Download object",
    description="Downloads raw binary bytes of a stored file directly from Cloudflare R2.",
)
async def download_storage_object(file_key: str):
    """Streams the stored file directly to the client."""
    try:
        data = await r2_storage_service.download_file(file_key)
        meta = await r2_storage_service.get_object_metadata(file_key)
        content_type = meta.get("content_type", "application/octet-stream")
        file_name = file_key.split("/")[-1]

        return StreamingResponse(
            io.BytesIO(data),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{file_name}"',
                "Content-Length": str(len(data)),
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Object '{file_key}' not found.")
    except Exception as exc:
        logger.error("Download failed for '%s': %s", file_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {exc}",
        )


@router.delete(
    "/objects/{file_key:path}",
    response_model=StorageDeleteResponse,
    summary="Delete object from storage",
    description="Deletes a raw object key from Cloudflare R2.",
)
async def delete_storage_object(file_key: str) -> StorageDeleteResponse:
    """Deletes an object from Cloudflare R2."""
    try:
        deleted = await r2_storage_service.delete_file(file_key)
        return StorageDeleteResponse(
            key=file_key,
            deleted=deleted,
            message=f"Object '{file_key}' successfully deleted from storage.",
        )
    except Exception as exc:
        logger.error("Failed to delete object '%s': %s", file_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {exc}",
        )


@router.post(
    "/presigned-url",
    response_model=PresignedUrlResponse,
    summary="Generate presigned S3 URL",
    description="Generates a temporary signed URL for direct client-to-R2 upload (put_object) or download (get_object).",
)
async def create_presigned_url(request: PresignedUrlRequest) -> PresignedUrlResponse:
    """Generates direct presigned S3/R2 URLs."""
    try:
        if request.operation == "put_object":
            url = await r2_storage_service.generate_presigned_upload_url(
                file_key=request.file_key,
                content_type=request.content_type,
                expires_in=request.expires_in,
            )
        else:
            url = await r2_storage_service.generate_presigned_url(
                file_key=request.file_key,
                expires_in=request.expires_in,
            )

        return PresignedUrlResponse(
            url=url,
            file_key=request.file_key,
            operation=request.operation,
            expires_in=request.expires_in,
        )
    except Exception as exc:
        logger.error("Presigned URL generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate presigned URL: {exc}",
        )
