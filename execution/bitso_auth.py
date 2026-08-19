from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Callable


def canonical_json_bytes(payload: dict[str, Any] | None) -> bytes:
    return b"" if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def generate_nonce_v2(
    *,
    clock: Callable[[], float] = time.time,
    randbelow: Callable[[int], int] = secrets.randbelow,
) -> str:
    milliseconds = int(clock() * 1_000)
    salt = 100_000 + randbelow(900_000)
    return f"{milliseconds}{salt:06d}"


def sign_request(nonce: str, method: str, path_with_query: str, body: bytes, secret: str) -> str:
    message = nonce.encode() + method.upper().encode() + path_with_query.encode() + body
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def authorization_header(key: str, nonce: str, signature: str) -> str:
    return f"Bitso {key}:{nonce}:{signature}"
