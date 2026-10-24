"""Pydantic schemas for Cloudflare R2 / S3 object storage management."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


def format_bytes(size_bytes: int) -> str:
    """Formats raw bytes into a human-readable size string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class StorageObjectItem(BaseModel):
    """Metadata item representing an individual object stored in R2."""

    key: str = Field(..., description="Unique object storage key")
    size: int = Field(..., description="Size in bytes")
    formatted_size: str = Field(..., description="Human-readable size string")
    last_modified: datetime | None = Field(default=None, description="UTC last modified timestamp")
    etag: str | None = Field(default=None, description="Object ETag hash")
    presigned_url: str | None = Field(default=None, description="Temporary presigned download URL")
    public_url: str | None = Field(default=None, description="Public CDN / custom domain URL if configured")


class StorageObjectListResponse(BaseModel):
    """Response containing a list of objects matching a prefix in R2."""

    prefix: str = Field(default="", description="Search prefix filter")
    total_count: int = Field(..., description="Number of objects returned in this batch")
    is_truncated: bool = Field(default=False, description="Whether more items are available")
    next_continuation_token: str | None = Field(default=None, description="Token to fetch next batch")
    objects: list[StorageObjectItem] = Field(..., description="List of storage objects")


class StorageUploadResponse(BaseModel):
    """Response returned upon successful file upload to R2."""

    key: str = Field(..., description="Object storage key")
    file_name: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    formatted_size: str = Field(..., description="Human-readable size string")
    content_type: str = Field(..., description="MIME content type")
    presigned_url: str = Field(..., description="Presigned download URL")
    public_url: str | None = Field(default=None, description="Public CDN URL if configured")
    message: str = Field(..., description="Status summary")


class StorageObjectDetailsResponse(BaseModel):
    """Detailed metadata for a specific stored object in R2."""

    key: str = Field(..., description="Object storage key")
    size: int = Field(..., description="Size in bytes")
    formatted_size: str = Field(..., description="Human-readable size string")
    content_type: str = Field(..., description="MIME content type")
    last_modified: datetime | None = Field(default=None, description="Last modified timestamp")
    etag: str | None = Field(default=None, description="Object ETag")
    presigned_url: str = Field(..., description="Temporary presigned download URL")
    public_url: str | None = Field(default=None, description="Public CDN URL if configured")


class StorageDeleteResponse(BaseModel):
    """Response returned after deleting an object from R2."""

    key: str = Field(..., description="Deleted object key")
    deleted: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status summary")


class PresignedUrlRequest(BaseModel):
    """Request payload to generate a presigned S3 URL."""

    file_key: str = Field(..., description="Target object storage key")
    operation: Literal["get_object", "put_object"] = Field(
        default="get_object",
        description="S3 operation: 'get_object' (download) or 'put_object' (direct client upload)",
    )
    content_type: str = Field(
        default="application/octet-stream",
        description="Content type for put_object operations",
    )
    expires_in: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="URL expiry duration in seconds (1 min to 24 hours)",
    )


class PresignedUrlResponse(BaseModel):
    """Response containing generated presigned URL."""

    url: str = Field(..., description="Signed S3/R2 direct access URL")
    file_key: str = Field(..., description="Target object key")
    operation: str = Field(..., description="Signed operation (get_object or put_object)")
    expires_in: int = Field(..., description="Expiry duration in seconds")


class StorageStatsResponse(BaseModel):
    """Aggregated storage statistics for Cloudflare R2 bucket."""

    configured: bool = Field(..., description="Whether live R2 credentials are fully configured")
    bucket_name: str = Field(..., description="Cloudflare R2 bucket name")
    region: str = Field(..., description="Configured bucket region")
    endpoint_url: str = Field(..., description="S3 endpoint URL")
    total_files: int = Field(..., description="Total number of objects stored")
    total_bytes: int = Field(..., description="Total size in bytes")
    formatted_storage: str = Field(..., description="Human-readable formatted total storage (e.g. 1.25 MB)")
