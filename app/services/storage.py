"""Cloudflare R2 Object Storage Service shim for backward compatibility.

Canonical implementation lives in app.services.r2_storage.
"""

from app.services.r2_storage import (
    R2StorageService,
    R2StorageService as StorageService,
    r2_storage_service,
    r2_storage_service as storage_service,
)

__all__ = [
    "R2StorageService",
    "StorageService",
    "r2_storage_service",
    "storage_service",
]
