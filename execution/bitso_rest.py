from __future__ import annotations

import asyncio
import random
import re
from typing import Any

import httpx

from config import BitsoConfig

from .bitso_auth import authorization_header, canonical_json_bytes, generate_nonce_v2, sign_request
from .rate_limit import TokenBucket

_ORIGIN_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class BitsoAPIError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(f"Bitso API error {status_code}/{code}: {message}")
        self.status_code = status_code
        self.code = code


class UncertainOrderError(RuntimeError):
    pass


class BitsoRESTClient:
    def __init__(
        self,
        config: BitsoConfig,
        *,
        api_key: str,
        api_secret: str,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
    ):
        self.config = config
        self._api_key = api_key
        self._api_secret = api_secret
        self.client = client or httpx.AsyncClient(timeout=config.request_timeout_seconds)
        self.public_bucket = TokenBucket(config.public_requests_per_minute)
        self.private_bucket = TokenBucket(config.private_requests_per_minute)
        self.max_retries = max_retries

    def __repr__(self) -> str:
        return f"BitsoRESTClient(base_url={self.config.rest_url!r})"

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        private: bool = False,
        idempotent: bool | None = None,
        cancellation: bool = False,
    ) -> Any:
        method = method.upper()
        idempotent = method in {"GET", "HEAD", "DELETE"} if idempotent is None else idempotent
        body = canonical_json_bytes(payload)
        url = f"{self.config.rest_url.rstrip('/')}/{path.lstrip('/')}"
        bucket = self.private_bucket if private else self.public_bucket
        attempts = self.max_retries + 1 if idempotent else 1

        for attempt in range(attempts):
            await bucket.acquire(exempt=cancellation)
            request = self.client.build_request(method, url, params=params, content=body or None)
            if private:
                nonce = generate_nonce_v2()
                signed_path = request.url.raw_path.decode("ascii")
                signature = sign_request(nonce, method, signed_path, body, self._api_secret)
                request.headers["Authorization"] = authorization_header(self._api_key, nonce, signature)
            if body:
                request.headers["Content-Type"] = "application/json"
            try:
                response = await self.client.send(request)
            except httpx.TransportError:
                if attempt + 1 == attempts:
                    raise
            else:
                if response.status_code < 400:
                    data = response.json()
                    if data.get("success") is True:
                        return data.get("payload")
                    error = data.get("error", {})
                    raise BitsoAPIError(response.status_code, str(error.get("code", "unknown")), str(error.get("message", "request failed")))
                if response.status_code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                    try:
                        error = response.json().get("error", {})
                    except ValueError:
                        error = {}
                    raise BitsoAPIError(response.status_code, str(error.get("code", "http")), str(error.get("message", "request failed")))
            await asyncio.sleep(min(2**attempt, 8) + random.random() * 0.25)
        raise AssertionError("unreachable")

    async def place_order(self, payload: dict[str, Any]) -> Any:
        origin_id = str(payload.get("origin_id", ""))
        if not _ORIGIN_ID.fullmatch(origin_id):
            raise ValueError("origin_id must be 1-40 letters, digits, underscores, or dashes")
        try:
            return await self.request("POST", "/orders", payload=payload, private=True, idempotent=False)
        except (httpx.TransportError, BitsoAPIError) as exc:
            if isinstance(exc, BitsoAPIError) and exc.status_code < 500:
                raise
            orders = await self.request("GET", "/orders", params={"origin_ids": origin_id}, private=True)
            if orders:
                return orders[0]
            trades = await self.request("GET", "/order_trades", params={"origin_id": origin_id}, private=True)
            if trades:
                return {"origin_id": origin_id, "status": "completed", "trades": trades}
            raise UncertainOrderError(f"order result for origin_id={origin_id!r} could not be reconciled") from exc

    async def cancel_order(self, order_id: str) -> Any:
        return await self.request("DELETE", f"/orders/{order_id}", private=True, cancellation=True)

    async def close(self) -> None:
        await self.client.aclose()
