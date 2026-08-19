from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import httpx

from config import RuntimeRiskParams
from execution import BaseExecutionEngine, ExecutionJournal, RiskManager
from ui import DashboardController, EventHub, create_app


class _Engine(BaseExecutionEngine):
    def __init__(self, journal):
        super().__init__(journal, RiskManager(RuntimeRiskParams()), "paper")

    async def cancel_order(self, order_id: str) -> None:
        self.open_order_ids.discard(order_id)

    async def liquidate(self) -> bool:
        self.position = None
        return True


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.journal = ExecutionJournal(Path(self.folder.name) / "journal.sqlite3")
        self.controller = DashboardController({"paper": _Engine(self.journal)})
        app = create_app(self.controller, "a-secure-test-token")
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.headers = {"Authorization": "Bearer a-secure-test-token"}

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.journal.close()
        self.folder.cleanup()

    async def test_auth_validation_and_unsafe_mode_change(self) -> None:
        self.assertEqual((await self.client.get("/api/state")).status_code, 401)
        response = await self.client.put(
            "/api/parameters", headers=self.headers, json={"risk_fraction": "0.01", "model_id": "unsafe"}
        )
        self.assertEqual(response.status_code, 422)
        response = await self.client.put("/api/parameters", headers=self.headers, json={"risk_fraction": "0.01"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.controller.engine.risk.params.risk_fraction, Decimal("0.01"))
        response = await self.client.put("/api/mode", headers=self.headers, json={"mode": "live"})
        self.assertEqual(response.status_code, 409)

    async def test_event_stream_is_bounded_and_redacted(self) -> None:
        hub = EventHub(max_queue=2)
        queue = hub.subscribe()
        hub.publish("log", {"message": "old"})
        hub.publish("log", {"api_token": "secret"})
        hub.publish("log", {"message": "new"})
        self.assertEqual(queue.qsize(), 2)
        self.assertEqual((await queue.get())["payload"]["api_token"], "[redacted]")

    async def test_kill_is_latched_against_api_restart(self) -> None:
        response = await self.client.post("/api/kill", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertLess(response.json()["dispatch_latency_ms"], 500)
        response = await self.client.post("/api/lifecycle", headers=self.headers, json={"action": "start"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.controller.state()["lifecycle"], "killed")


if __name__ == "__main__":
    unittest.main()
