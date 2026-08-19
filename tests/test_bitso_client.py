from __future__ import annotations

import unittest

from execution import (
    LocalOrderBook,
    SequenceGapError,
    TokenBucket,
    canonical_json_bytes,
    generate_nonce_v2,
    sign_request,
)


class BitsoClientTests(unittest.IsolatedAsyncioTestCase):
    def test_fixed_signature_nonce_and_exact_json_bytes(self) -> None:
        payload = {"book": "btc_usd", "major": "0.1", "side": "buy", "type": "market"}
        body = canonical_json_bytes(payload)
        self.assertEqual(body, b'{"book":"btc_usd","major":"0.1","side":"buy","type":"market"}')
        signature = sign_request("1731349200123123456", "POST", "/api/v3/orders", body, "test-secret")
        self.assertEqual(signature, "1226423d80c46a188e6f7ede1cb2054b3b2559bd4431ca200b8fde934c3043f2")
        self.assertEqual(generate_nonce_v2(clock=lambda: 1_731_349_200.123, randbelow=lambda _: 0), "1731349200123100000")

    async def test_token_bucket_and_cancellation_exemption(self) -> None:
        now = [0.0]
        waits: list[float] = []

        async def sleep(delay: float) -> None:
            waits.append(delay)
            now[0] += delay

        bucket = TokenBucket(60, clock=lambda: now[0], sleeper=sleep)
        bucket.capacity = bucket.tokens = 1
        await bucket.acquire()
        await bucket.acquire()
        await bucket.acquire(exempt=True)
        self.assertEqual(waits, [1.0])

    def test_order_book_gap_detection_and_snapshot_recovery(self) -> None:
        book = LocalOrderBook("btc_usd")
        book.bootstrap(
            {
                "sequence": "10",
                "bids": [{"oid": "b1", "price": "99", "amount": "2"}],
                "asks": [{"oid": "a1", "price": "101", "amount": "3"}],
            }
        )
        book.apply_diff(
            {"sequence": 11, "payload": [{"o": "b1", "r": "99", "a": "1", "t": 0, "s": "open"}]}
        )
        with self.assertRaises(SequenceGapError):
            book.apply_diff({"sequence": 13, "payload": []})
        book.bootstrap({"sequence": "13", "bids": [], "asks": []})
        self.assertFalse(book.apply_diff({"sequence": 12, "payload": []}))
        self.assertEqual(book.sequence, 13)


if __name__ == "__main__":
    unittest.main()
