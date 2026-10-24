"""Cloudflare R2 Object Storage Service (S3-compatible).

Handles secure file upload, streaming download, presigned URL generation,
and object deletion via aioboto3.
Includes development/fallback storage when R2 credentials are not configured.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import io
import logging
from typing import Any, BinaryIO

import aioboto3
from botocore.config import Config

from app.core.config import settings

logger = logging.getLogger(__name__)


class R2StorageService:
    """Asynchronous object storage service compatible with Cloudflare R2 and AWS S3."""

    def __init__(self) -> None:
        self.session = aioboto3.Session()
        self.access_key = settings.r2_effective_access_key
        self.secret_key = settings.r2_effective_secret_key
        self.bucket_name = settings.r2_effective_bucket_name
        self.endpoint_url = settings.r2_effective_endpoint_url
        self.region_name = settings.R2_REGION or "auto"
        self.public_base_url = settings.R2_PUBLIC_BASE_URL
        # In-memory storage fallback for local development/testing without R2 credentials
        self._dev_storage: dict[str, tuple[bytes, str]] = {}

    @property
    def is_configured(self) -> bool:
        """Checks if Cloudflare R2 credentials are fully configured."""
        return bool(self.access_key and self.secret_key and self.bucket_name and self.endpoint_url)

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[Any, None]:
        """Context manager providing an initialized async S3 client."""
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name=self.region_name,
        ) as client:
            yield client

    async def upload_file(
        self,
        file_data: bytes | BinaryIO,
        file_key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Uploads a file to Cloudflare R2 bucket.

        Args:
            file_data: File payload as bytes or readable binary stream.
            file_key: Unique destination storage key (e.g. 'documents/user1/doc.pdf').
            content_type: MIME type of the file.

        Returns:
            The stored file key.
        """
        if not self.is_configured:
            logger.warning("Cloudflare R2 not configured. Storing '%s' in memory for development.", file_key)
            raw = file_data if isinstance(file_data, bytes) else file_data.read()
            self._dev_storage[file_key] = (raw, content_type)
            return file_key

        body = file_data if isinstance(file_data, bytes) else file_data.read()
        async with self.get_client() as s3:
            await s3.put_object(
                Bucket=self.bucket_name,
                Key=file_key,
                Body=body,
                ContentType=content_type,
            )
        logger.info("Uploaded file to Cloudflare R2: key='%s' in bucket='%s'", file_key, self.bucket_name)
        return file_key

    async def download_file(self, file_key: str) -> bytes:
        """Downloads a file's raw bytes from Cloudflare R2.

        Args:
            file_key: Storage key of the object to retrieve.

        Returns:
            The binary file content.
        """
        if not self.is_configured:
            if file_key in self._dev_storage:
                return self._dev_storage[file_key][0]
            raise FileNotFoundError(f"File key '{file_key}' not found in development storage.")

        async with self.get_client() as s3:
            response = await s3.get_object(Bucket=self.bucket_name, Key=file_key)
            async with response["Body"] as stream:
                return await stream.read()

    async def delete_file(self, file_key: str) -> bool:
        """Deletes an object from Cloudflare R2.

        Args:
            file_key: Storage key of the object to delete.

        Returns:
            True if deletion was successful.
        """
        if not self.is_configured:
            self._dev_storage.pop(file_key, None)
            return True

        async with self.get_client() as s3:
            await s3.delete_object(Bucket=self.bucket_name, Key=file_key)
        logger.info("Deleted file from Cloudflare R2: key='%s'", file_key)
        return True

    async def generate_presigned_url(self, file_key: str, expires_in: int = 3600) -> str:
        """Generates a presigned GET URL for downloading private R2 files."""
        if not self.is_configured:
            return f"/api/v1/documents/dev-download/{file_key}"

        async with self.get_client() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": file_key},
                ExpiresIn=expires_in,
            )
            return url

    def get_public_url(self, file_key: str) -> str | None:
        """Returns the public CDN / custom domain URL if R2_PUBLIC_BASE_URL is configured."""
        if not self.public_base_url:
            return None
        return f"{self.public_base_url.rstrip('/')}/{file_key.lstrip('/')}"


# Global singleton instance
r2_storage_service = R2StorageService()

# Backward-compatibility aliases
StorageService = R2StorageService
storage_service = r2_storage_service
