"""Cloudflare R2 Object Storage Service (S3-compatible).

Handles secure file upload, streaming download, presigned URL generation,
and object deletion via aioboto3.
Includes development/fallback storage when R2 credentials are not configured.
"""

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import io
import logging
from typing import Any, BinaryIO

import aioboto3
from botocore.config import Config
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)


def _is_transient_r2_error(exc: BaseException) -> bool:
    """Returns True if the exception represents a transient network, timeout, or 5xx S3 error."""
    exc_name = exc.__class__.__name__
    if any(k in exc_name for k in ("Timeout", "Connection", "EndpointConnection", "IncompleteRead")):
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if status_code in (429, 500, 502, 503, 504):
            return True
        code = response.get("Error", {}).get("Code", "")
        if code in ("RequestTimeout", "ThrottlingException", "SlowDown", "InternalError", "ServiceUnavailable"):
            return True
    return False


async def _run_with_retry(coro_fn, *args: Any, **kwargs: Any) -> Any:
    """Executes a coroutine with exponential backoff retries for transient S3/R2 errors."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception(_is_transient_r2_error),
        reraise=True,
    ):
        with attempt:
            return await coro_fn(*args, **kwargs)


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
        client_context = self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name=self.region_name,
        )
        if not isinstance(client_context, AbstractAsyncContextManager):
            raise TypeError("The storage SDK did not provide an asynchronous client context")
        async with client_context as client:
            yield client

    async def file_exists(self, file_key: str) -> bool:
        """Checks if a file exists in the storage bucket."""
        if not self.is_configured:
            return file_key in self._dev_storage

        async def _head() -> bool:
            async with self.get_client() as s3:
                try:
                    await s3.head_object(Bucket=self.bucket_name, Key=file_key)
                    return True
                except Exception as exc:
                    response = getattr(exc, "response", None)
                    if isinstance(response, dict):
                        code = response.get("Error", {}).get("Code")
                        if code in ("404", "NoSuchKey", "NotFound"):
                            return False
                    if "404" in str(exc) or "NotFound" in str(exc):
                        return False
                    raise exc

        try:
            return await _run_with_retry(_head)
        except Exception as exc:
            logger.warning("Error checking file existence for '%s': %s", file_key, exc)
            return False

    async def upload_file(
        self,
        file_data: bytes | BinaryIO,
        file_key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Uploads a file to Cloudflare R2 bucket with automated retry.

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

        async def _put():
            async with self.get_client() as s3:
                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=file_key,
                    Body=body,
                    ContentType=content_type,
                )

        await _run_with_retry(_put)
        logger.info("Uploaded file to Cloudflare R2: key='%s' in bucket='%s'", file_key, self.bucket_name)
        return file_key

    async def download_file(self, file_key: str) -> bytes:
        """Downloads a file's raw bytes from Cloudflare R2 with automated retry.

        Args:
            file_key: Storage key of the object to retrieve.

        Returns:
            The binary file content.
        """
        if not self.is_configured:
            if file_key in self._dev_storage:
                return self._dev_storage[file_key][0]
            raise FileNotFoundError(f"File key '{file_key}' not found in development storage.")

        async def _get() -> bytes:
            async with self.get_client() as s3:
                response = await s3.get_object(Bucket=self.bucket_name, Key=file_key)
                async with response["Body"] as stream:
                    return await stream.read()

        return await _run_with_retry(_get)

    async def delete_file(self, file_key: str) -> bool:
        """Deletes an object from Cloudflare R2 with automated retry.

        Args:
            file_key: Storage key of the object to delete.

        Returns:
            True if deletion was successful.
        """
        if not self.is_configured:
            self._dev_storage.pop(file_key, None)
            return True

        async def _delete():
            async with self.get_client() as s3:
                await s3.delete_object(Bucket=self.bucket_name, Key=file_key)

        await _run_with_retry(_delete)
        logger.info("Deleted file from Cloudflare R2: key='%s'", file_key)
        return True

    async def generate_presigned_url(
        self,
        file_key: str,
        expires_in: int = 3600,
        inline: bool = False,
        filename: str | None = None,
    ) -> str:
        """Generates a presigned GET URL for downloading or viewing private R2 files.

        Args:
            file_key: Storage key in bucket.
            expires_in: Lifetime of the signed URL in seconds.
            inline: If True, sets ResponseContentDisposition to 'inline' for browser viewing/citations.
            filename: Optional clean filename for Content-Disposition header.
        """
        if not self.is_configured:
            query_string = "?inline=true" if inline else ""
            return f"/api/v1/storage/dev-download/{file_key}{query_string}"

        params: dict[str, Any] = {"Bucket": self.bucket_name, "Key": file_key}
        if inline:
            display_name = filename or file_key.split("/")[-1]
            params["ResponseContentDisposition"] = f'inline; filename="{display_name}"'

        async with self.get_client() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params=params,
                ExpiresIn=expires_in,
            )
            return url

    async def generate_presigned_upload_url(
        self,
        file_key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 3600,
    ) -> str:
        """Generates a presigned PUT URL allowing clients to upload directly to R2."""
        if not self.is_configured:
            return f"/api/v1/storage/dev-upload/{file_key}"

        async with self.get_client() as s3:
            url = await s3.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket_name, "Key": file_key, "ContentType": content_type},
                ExpiresIn=expires_in,
            )
            return url

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 100,
        continuation_token: str | None = None,
    ) -> dict[str, Any]:
        """Lists objects stored in Cloudflare R2 matching a prefix."""
        if not self.is_configured:
            matched = [
                {
                    "key": k,
                    "size": len(v[0]),
                    "content_type": v[1],
                    "last_modified": None,
                    "etag": None,
                    "presigned_url": f"/api/v1/storage/dev-download/{k}",
                    "public_url": self.get_public_url(k),
                }
                for k, v in self._dev_storage.items()
                if k.startswith(prefix)
            ]
            return {
                "objects": matched[:max_keys],
                "total_count": len(matched),
                "is_truncated": len(matched) > max_keys,
                "next_continuation_token": None,
            }

        params: dict[str, Any] = {
            "Bucket": self.bucket_name,
            "MaxKeys": min(max(1, max_keys), 1000),
        }
        if prefix:
            params["Prefix"] = prefix
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        async with self.get_client() as s3:
            res = await s3.list_objects_v2(**params)
            items = []
            for item in res.get("Contents", []):
                key = item["Key"]
                items.append({
                    "key": key,
                    "size": item["Size"],
                    "last_modified": item.get("LastModified"),
                    "etag": item.get("ETag", "").strip('"'),
                    "presigned_url": await self.generate_presigned_url(key),
                    "public_url": self.get_public_url(key),
                })

            return {
                "objects": items,
                "total_count": len(items),
                "is_truncated": res.get("IsTruncated", False),
                "next_continuation_token": res.get("NextContinuationToken"),
            }

    async def get_object_metadata(self, file_key: str) -> dict[str, Any]:
        """Retrieves metadata headers for a specific object."""
        if not self.is_configured:
            if file_key not in self._dev_storage:
                raise FileNotFoundError(f"File key '{file_key}' not found in development storage.")
            data, content_type = self._dev_storage[file_key]
            return {
                "key": file_key,
                "size": len(data),
                "content_type": content_type,
                "last_modified": None,
                "etag": None,
                "presigned_url": f"/api/v1/storage/dev-download/{file_key}",
                "public_url": self.get_public_url(file_key),
            }

        async with self.get_client() as s3:
            res = await s3.head_object(Bucket=self.bucket_name, Key=file_key)
            return {
                "key": file_key,
                "size": res.get("ContentLength", 0),
                "content_type": res.get("ContentType", "application/octet-stream"),
                "last_modified": res.get("LastModified"),
                "etag": res.get("ETag", "").strip('"'),
                "presigned_url": await self.generate_presigned_url(file_key),
                "public_url": self.get_public_url(file_key),
            }

    async def get_storage_stats(self, prefix: str = "") -> dict[str, Any]:
        """Calculates aggregated storage metrics for the bucket or prefix."""
        if not self.is_configured:
            total_size = sum(len(v[0]) for k, v in self._dev_storage.items() if k.startswith(prefix))
            total_files = len([k for k in self._dev_storage if k.startswith(prefix)])
            return {
                "configured": False,
                "bucket_name": self.bucket_name or "in_memory_dev",
                "total_files": total_files,
                "total_bytes": total_size,
                "endpoint_url": "in_memory",
                "region": "dev",
            }

        total_files = 0
        total_size = 0
        continuation_token = None

        async with self.get_client() as s3:
            while True:
                params: dict[str, Any] = {"Bucket": self.bucket_name, "MaxKeys": 1000}
                if prefix:
                    params["Prefix"] = prefix
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                res = await s3.list_objects_v2(**params)
                for item in res.get("Contents", []):
                    total_files += 1
                    total_size += item["Size"]

                if res.get("IsTruncated"):
                    continuation_token = res.get("NextContinuationToken")
                else:
                    break

        return {
            "configured": True,
            "bucket_name": self.bucket_name,
            "total_files": total_files,
            "total_bytes": total_size,
            "endpoint_url": self.endpoint_url,
            "region": self.region_name,
        }

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
