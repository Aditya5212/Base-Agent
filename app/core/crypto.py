"""Cryptographic utilities for encrypting sensitive credentials and secrets at rest.

Uses symmetric Fernet encryption (AES-128 in CBC mode with HMAC-SHA256 authentication).
Key derivation supports:
1. Explicit SECRET_ENCRYPTION_KEY in environment/settings.
2. Deterministic derivation from DATABASE_URL for seamless local development.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings

logger = logging.getLogger(__name__)


class CryptoError(Exception):
    """Raised when encryption or decryption fails."""


@lru_cache(maxsize=1)
def get_fernet() -> Fernet:
    """Returns a singleton Fernet cipher instance initialized with the application key."""
    raw_key = settings.SECRET_ENCRYPTION_KEY

    if raw_key and raw_key.strip():
        key_str = raw_key.strip()
        # If it is already a 44-character base64-encoded 32-byte key
        try:
            key_bytes = key_str.encode("utf-8")
            decoded = base64.urlsafe_b64decode(key_bytes)
            if len(decoded) == 32:
                return Fernet(key_bytes)
        except Exception:
            pass

        # Otherwise derive a 32-byte key via SHA-256
        derived_digest = hashlib.sha256(key_str.encode("utf-8")).digest()
        urlsafe_key = base64.urlsafe_b64encode(derived_digest)
        return Fernet(urlsafe_key)

    # Deterministic fallback derived from DATABASE_URL + static salt for dev/test stability
    seed = f"{settings.DATABASE_URL}:base_agent_fernet_secret_salt"
    derived_digest = hashlib.sha256(seed.encode("utf-8")).digest()
    urlsafe_key = base64.urlsafe_b64encode(derived_digest)
    return Fernet(urlsafe_key)


def encrypt_secret(plaintext: str | None) -> str | None:
    """Encrypts plaintext string using Fernet and returns URL-safe base64 string.

    Args:
        plaintext: The raw secret to encrypt. If None or empty, returns None.

    Returns:
        Encrypted ciphertext string, or None if input was empty.
    """
    if not plaintext:
        return None

    try:
        f = get_fernet()
        encrypted_bytes = f.encrypt(plaintext.encode("utf-8"))
        return encrypted_bytes.decode("utf-8")
    except Exception as exc:
        logger.error("Failed to encrypt secret: %s", exc)
        raise CryptoError(f"Encryption failed: {exc}") from exc


def decrypt_secret(ciphertext: str | None) -> str | None:
    """Decrypts a Fernet-encrypted ciphertext string back to plaintext.

    Args:
        ciphertext: The encrypted string. If None or empty, returns None.

    Returns:
        Decrypted plaintext string.

    Raises:
        CryptoError: If decryption fails or token is invalid.
    """
    if not ciphertext:
        return None

    try:
        f = get_fernet()
        decrypted_bytes = f.decrypt(ciphertext.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except InvalidToken as exc:
        logger.error("Invalid token during secret decryption: cipher payload corrupted or key changed")
        raise CryptoError("Failed to decrypt secret: invalid or corrupted token") from exc
    except Exception as exc:
        logger.error("Failed to decrypt secret: %s", exc)
        raise CryptoError(f"Decryption failed: {exc}") from exc


def mask_secret(secret: str | None, prefix_len: int = 3, suffix_len: int = 3) -> str:
    """Masks a secret string for safe display in logs and client responses.

    Example:
        'sk-ant-api03-123456789' -> 'sk-***789'
    """
    if not secret:
        return ""

    s = secret.strip()
    if len(s) <= prefix_len + suffix_len:
        return "***"

    return f"{s[:prefix_len]}***{s[-suffix_len:]}"

