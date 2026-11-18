"""HTTP REST API tool executor.

Handles async HTTP requests with template interpolation, authentication injection,
retries on transient failures, and JSONPath / response traversal.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any
import urllib.parse

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.crypto import decrypt_secret
from app.schemas.tool import HttpAuthType, HttpToolConfig

logger = logging.getLogger(__name__)


class HttpToolExecutionError(Exception):
    """Raised when HTTP tool execution fails."""


def _interpolate_string(template: str, args: dict[str, Any], url_encode: bool = False) -> str:
    """Replaces {key} placeholders in template with corresponding values from args."""
    result = template
    for k, v in args.items():
        placeholder = f"{{{k}}}"
        if placeholder in result:
            val_str = str(v)
            if url_encode:
                val_str = urllib.parse.quote(val_str, safe="")
            result = result.replace(placeholder, val_str)
    return result


def _interpolate_obj(obj: Any, args: dict[str, Any]) -> Any:
    """Recursively interpolates placeholders in nested dicts/lists/strings."""
    if isinstance(obj, str):
        return _interpolate_string(obj, args)
    if isinstance(obj, dict):
        return {k: _interpolate_obj(v, args) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_obj(item, args) for item in obj]
    return obj


def _extract_response_path(data: Any, path: str) -> Any:
    """Extracts a nested property using dot and bracket notation.

    Examples:
        'data.items' -> data['data']['items']
        'results[0].name' -> data['results'][0]['name']
    """
    if not path or not path.strip():
        return data

    current = data
    parts = path.strip().split(".")
    for part in parts:
        if "[" in part and part.endswith("]"):
            key, idx_str = part[:-1].split("[", 1)
            if key:
                if not isinstance(current, dict) or key not in current:
                    return current
                current = current[key]
            try:
                idx = int(idx_str)
                if isinstance(current, (list, tuple)) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return current
            except ValueError:
                return current
        else:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return current
    return current


class HttpToolExecutor:
    """Executes configured HTTP API tools against external REST services."""

    def __init__(self, config: HttpToolConfig):
        self.config = config

    async def execute(
        self,
        arguments: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Executes the HTTP request using the resolved arguments."""
        args = dict(arguments or {})

        # 1. Resolve URL template
        url = _interpolate_string(self.config.url_template, args, url_encode=True)

        # 2. Build Headers & Authentication
        headers: dict[str, str] = {}
        for k, v in self.config.headers_template.items():
            headers[k] = _interpolate_string(v, args)

        idempotency_key = (
            (metadata or {}).get("idempotency_key")
            or (metadata or {}).get("tool_idempotency_key")
            or args.pop("__idempotency_key__", None)
            or args.get("idempotency_key")
        )
        if idempotency_key and "X-Idempotency-Key" not in headers:
            headers["X-Idempotency-Key"] = str(idempotency_key)

        # Decrypt auth token if configured
        auth_token = self.config.auth_token
        if auth_token:
            if auth_token.startswith("gAAAAA"):
                auth_token = decrypt_secret(auth_token)

        if auth_token:
            if self.config.auth_type == HttpAuthType.BEARER:
                headers["Authorization"] = f"Bearer {auth_token}"
            elif self.config.auth_type == HttpAuthType.API_KEY:
                header_name = self.config.auth_header_name or "Authorization"
                headers[header_name] = auth_token
            elif self.config.auth_type == HttpAuthType.BASIC:
                encoded = base64.b64encode(auth_token.encode("utf-8")).decode("ascii")
                headers["Authorization"] = f"Basic {encoded}"

        # 3. Build Body & Query Params
        method = self.config.method.upper()
        json_body = None
        params: dict[str, Any] = {}

        if method in ("GET", "DELETE"):
            # Any parameter not present in the URL path is passed as a query param
            for k, v in args.items():
                if f"{{{k}}}" not in self.config.url_template:
                    params[k] = v
        else:
            # POST, PUT, PATCH
            if self.config.body_template is not None:
                json_body = _interpolate_obj(self.config.body_template, args)
            else:
                # If no body template, pass all non-URL arguments as JSON payload
                json_body = {k: v for k, v in args.items() if f"{{{k}}}" not in self.config.url_template}

        # 4. Execute with retries
        retry_count = max(0, min(self.config.retry_count, 5))
        timeout = httpx.Timeout(self.config.timeout_seconds)

        async def _send_request() -> httpx.Response:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params or None,
                    json=json_body if json_body is not None else None,
                )
                if resp.status_code >= 500:
                    resp.raise_for_status()
                return resp

        response: httpx.Response | None = None
        try:
            if retry_count > 0:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(retry_count + 1),
                    wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
                    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
                    reraise=True,
                ):
                    with attempt:
                        response = await _send_request()
            else:
                response = await _send_request()

        except Exception as exc:
            logger.error("HTTP Tool execution failed for '%s %s': %s", method, url, exc)
            raise HttpToolExecutionError(f"HTTP request failed: {exc}") from exc

        # 5. Process and extract response
        if response is None:
            raise HttpToolExecutionError("HTTP retry loop completed without a response")
        status = response.status_code
        content_type = response.headers.get("content-type", "").lower()

        if 200 <= status < 300:
            if "application/json" in content_type:
                try:
                    data = response.json()
                    if self.config.response_path:
                        data = _extract_response_path(data, self.config.response_path)
                    return data
                except Exception:
                    return response.text
            return response.text
        else:
            raise HttpToolExecutionError(
                f"HTTP {status} {response.reason_phrase}: {response.text[:300]}"
            )

