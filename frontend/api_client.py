"""Typed HTTP client for the Base Agent API.

Every public method maps to exactly one FastAPI route. ENDPOINTS documents the
mapping (route -> client method -> UI location) and is rendered on the System page.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

import httpx

API_PREFIX = "/api/v1"

# (group, HTTP method, route, client method, where it is used in the UI)
ENDPOINTS: list[tuple[str, str, str, str, str]] = [
    ("System", "GET", "/", "root", "System → Service info"),
    ("System", "GET", "/api/v1/health", "health", "Sidebar status, System → Health"),
    ("System", "GET", "/api/v1/ping", "ping", "Sidebar status, System → Health"),
    ("Chat", "POST", "/api/v1/chat", "send_message", "Chat → send (streaming off)"),
    ("Chat", "POST", "/api/v1/chat/stream", "stream_message", "Chat → send (streaming on)"),
    ("Chat", "POST", "/api/v1/chat/{thread_id}/assets", "upload_thread_asset", "Chat → Attachments → Upload"),
    ("Chat", "GET", "/api/v1/chat/{thread_id}/assets", "list_thread_assets", "Chat → Attachments list"),
    ("Chat", "GET", "/api/v1/chat/{thread_id}/details", "get_thread_details", "Chat → Thread details"),
    ("Chat", "GET", "/api/v1/chat/{thread_id}/history", "get_thread_history", "Chat → open / reload thread"),
    ("Chat", "DELETE", "/api/v1/chat/{thread_id}", "clear_thread_memory", "Chat → Clear memory"),
    ("Documents", "GET", "/api/v1/documents/stats", "get_kb_stats", "Knowledge Base → Overview metrics"),
    ("Documents", "GET", "/api/v1/documents", "list_documents", "Knowledge Base → Documents table"),
    ("Documents", "POST", "/api/v1/documents/upload", "upload_document", "Knowledge Base → Upload"),
    ("Documents", "GET", "/api/v1/documents/{document_id}", "get_document", "Knowledge Base → Document details"),
    ("Documents", "GET", "/api/v1/documents/{document_id}/chunks", "get_document_chunks", "Knowledge Base → Chunk inspector"),
    ("Documents", "POST", "/api/v1/documents/{document_id}/reindex", "reindex_document", "Knowledge Base → Re-index"),
    ("Documents", "GET", "/api/v1/documents/{document_id}/download", "download_document", "Knowledge Base → Download / Preview, Chat citations"),
    ("Documents", "DELETE", "/api/v1/documents/{document_id}", "delete_document", "Knowledge Base → Delete"),
    ("Documents", "POST", "/api/v1/documents/search", "search_documents", "Knowledge Base → Search playground"),
    ("Storage", "GET", "/api/v1/storage/stats", "get_storage_stats", "Storage → Bucket metrics"),
    ("Storage", "GET", "/api/v1/storage/objects", "list_objects", "Storage → Browse"),
    ("Storage", "POST", "/api/v1/storage/upload", "upload_object", "Storage → Upload"),
    ("Storage", "GET", "/api/v1/storage/objects/{key}", "get_object_metadata", "Storage → Object details"),
    ("Storage", "GET", "/api/v1/storage/download/{key}", "download_object", "Storage → Download"),
    ("Storage", "DELETE", "/api/v1/storage/objects/{key}", "delete_object", "Storage → Delete"),
    ("Storage", "POST", "/api/v1/storage/presigned-url", "create_presigned_url", "Storage → Presigned URL"),
]


class ApiError(Exception):
    """Raised when the API returns a 4xx/5xx response."""

    def __init__(self, status_code: int, detail: str, method: str, path: str):
        self.status_code = status_code
        self.detail = detail
        self.method = method
        self.path = path
        super().__init__(f"{method} {path} → {status_code}: {detail}")


@dataclass
class Identity:
    """Mock-auth identity sent on every request (see app/core/auth.py)."""

    user_id: str = "root"
    role: str = "root"

    def headers(self) -> dict[str, str]:
        return {"X-User-ID": self.user_id, "X-User-Role": self.role}


@dataclass
class DownloadedFile:
    content: bytes
    filename: str
    content_type: str


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text or resp.reason_phrase
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body
        if isinstance(detail, list):  # FastAPI validation errors
            return "; ".join(
                f"{'.'.join(str(p) for p in item.get('loc', []))}: {item.get('msg')}" for item in detail
            )
        return str(detail)
    return str(body)


def _clean(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None and v != ""}


def _filename_from_headers(resp: httpx.Response, fallback: str) -> str:
    disposition = resp.headers.get("content-disposition", "")
    if "filename=" in disposition:
        return disposition.split("filename=", 1)[1].strip().strip('"') or fallback
    return fallback


class BaseAgentClient:
    """Synchronous client for the Base Agent REST API."""

    def __init__(self, base_url: str, identity: Identity | None = None, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.identity = identity or Identity()
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5.0))

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ core

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        files: Any = None,
        data: dict[str, Any] | None = None,
        accept_status: tuple[int, ...] = (),
    ) -> httpx.Response:
        url = path if path == "/" else f"{API_PREFIX}{path}"
        resp = self._http.request(
            method,
            url,
            params=_clean(params),
            json=json_body,
            files=files,
            data=_clean(data),
            headers=self.identity.headers(),
        )
        if resp.status_code >= 400 and resp.status_code not in accept_status:
            raise ApiError(resp.status_code, _error_detail(resp), method, url)
        return resp

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._send(method, path, **kwargs).json()

    def browser_url(self, path: str, **params: Any) -> str:
        """Absolute URL for links opened directly in the browser.

        Browsers cannot send X-User-ID, so identity goes in the user_id query
        param, which get_current_user also accepts.
        """
        query = _clean({"user_id": self.identity.user_id, **params}) or {}
        return f"{self.base_url}{path}?{urlencode(query)}"

    # ---------------------------------------------------------------- system

    def root(self) -> dict[str, Any]:
        return self._json("GET", "/")

    def health(self) -> dict[str, Any]:
        # 503 still carries a HealthResponse body (status="degraded")
        return self._json("GET", "/health", accept_status=(503,))

    def ping(self) -> dict[str, Any]:
        return self._json("GET", "/ping")

    # ------------------------------------------------------------------ chat

    def send_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/chat", json_body=payload)

    def stream_message(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yields parsed SSE events: {"event": token|reasoning|end|error, "data": {...}}."""
        url = f"{API_PREFIX}/chat/stream"
        with self._http.stream(
            "POST",
            url,
            json=payload,
            headers=self.identity.headers(),
            timeout=httpx.Timeout(600.0, connect=5.0),
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise ApiError(resp.status_code, _error_detail(resp), "POST", url)

            event_name: str | None = None
            data_lines: list[str] = []
            for line in resp.iter_lines():
                if line == "":
                    if data_lines:
                        yield {"event": event_name or "message", "data": json.loads("\n".join(data_lines))}
                    event_name, data_lines = None, []
                elif line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip())
            if data_lines:
                yield {"event": event_name or "message", "data": json.loads("\n".join(data_lines))}

    def upload_thread_asset(self, thread_id: str, filename: str, content: bytes, content_type: str) -> dict[str, Any]:
        return self._json(
            "POST",
            f"/chat/{quote(thread_id, safe='')}/assets",
            files={"file": (filename, content, content_type)},
            data={"user_id": self.identity.user_id},
        )

    def list_thread_assets(self, thread_id: str) -> list[dict[str, Any]]:
        return self._json("GET", f"/chat/{quote(thread_id, safe='')}/assets")

    def get_thread_details(self, thread_id: str) -> dict[str, Any]:
        return self._json("GET", f"/chat/{quote(thread_id, safe='')}/details")

    def get_thread_history(self, thread_id: str) -> dict[str, Any]:
        return self._json("GET", f"/chat/{quote(thread_id, safe='')}/history")

    def clear_thread_memory(self, thread_id: str) -> dict[str, Any]:
        return self._json("DELETE", f"/chat/{quote(thread_id, safe='')}")

    # ------------------------------------------------------------- documents

    def get_kb_stats(self) -> dict[str, Any]:
        return self._json("GET", "/documents/stats")

    def list_documents(
        self,
        search: str | None = None,
        status_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._json(
            "GET",
            "/documents",
            params={"search": search, "status_filter": status_filter, "limit": limit, "offset": offset},
        )

    def upload_document(self, filename: str, content: bytes, content_type: str, preset: str) -> dict[str, Any]:
        return self._json(
            "POST",
            "/documents/upload",
            files={"file": (filename, content, content_type)},
            data={"preset": preset},
        )

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self._json("GET", f"/documents/{document_id}")

    def get_document_chunks(self, document_id: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return self._json("GET", f"/documents/{document_id}/chunks", params={"limit": limit, "offset": offset})

    def reindex_document(self, document_id: str, preset: str) -> dict[str, Any]:
        return self._json("POST", f"/documents/{document_id}/reindex", json_body={"preset": preset})

    def download_document(self, document_id: str, inline: bool = False) -> DownloadedFile:
        resp = self._send("GET", f"/documents/{document_id}/download", params={"inline": str(inline).lower()})
        return DownloadedFile(
            content=resp.content,
            filename=_filename_from_headers(resp, f"{document_id}.bin"),
            content_type=resp.headers.get("content-type", "application/octet-stream"),
        )

    def document_view_url(self, document_id: str) -> str:
        return self.browser_url(f"{API_PREFIX}/documents/{document_id}/download", inline="true")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        # The first delete per user initializes LightRAG server-side, which can take minutes.
        url = f"{API_PREFIX}/documents/{document_id}"
        resp = self._http.request(
            "DELETE", url, headers=self.identity.headers(), timeout=httpx.Timeout(600.0, connect=5.0)
        )
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, _error_detail(resp), "DELETE", url)
        return resp.json()

    def search_documents(self, query: str, preset: str = "balanced", top_k: int | None = None) -> dict[str, Any]:
        return self._json(
            "POST",
            "/documents/search",
            json_body={"query": query, "preset": preset, "top_k": top_k, "user_id": self.identity.user_id},
        )

    # --------------------------------------------------------------- storage
    # The storage routes also take a `user_id` query param that must equal
    # "root"; we leave it at its default and send identity via headers only.

    def get_storage_stats(self, prefix: str = "") -> dict[str, Any]:
        return self._json("GET", "/storage/stats", params={"prefix": prefix})

    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 100,
        continuation_token: str | None = None,
    ) -> dict[str, Any]:
        return self._json(
            "GET",
            "/storage/objects",
            params={"prefix": prefix, "max_keys": max_keys, "continuation_token": continuation_token},
        )

    def upload_object(self, filename: str, content: bytes, content_type: str, prefix: str) -> dict[str, Any]:
        return self._json(
            "POST",
            "/storage/upload",
            files={"file": (filename, content, content_type)},
            data={"prefix": prefix},
        )

    def get_object_metadata(self, key: str) -> dict[str, Any]:
        return self._json("GET", f"/storage/objects/{quote(key, safe='/')}")

    def download_object(self, key: str, inline: bool = False) -> DownloadedFile:
        resp = self._send(
            "GET", f"/storage/download/{quote(key, safe='/')}", params={"inline": str(inline).lower()}
        )
        return DownloadedFile(
            content=resp.content,
            filename=_filename_from_headers(resp, key.rsplit("/", 1)[-1]),
            content_type=resp.headers.get("content-type", "application/octet-stream"),
        )

    def delete_object(self, key: str) -> dict[str, Any]:
        return self._json("DELETE", f"/storage/objects/{quote(key, safe='/')}")

    def create_presigned_url(
        self,
        file_key: str,
        operation: str = "get_object",
        content_type: str = "application/octet-stream",
        expires_in: int = 3600,
    ) -> dict[str, Any]:
        return self._json(
            "POST",
            "/storage/presigned-url",
            json_body={
                "file_key": file_key,
                "operation": operation,
                "content_type": content_type,
                "expires_in": expires_in,
            },
        )
