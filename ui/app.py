from __future__ import annotations

import asyncio
import secrets
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from config import RuntimeRiskParams, required_secret
from execution import BaseExecutionEngine, KillSwitch, LiveExecutionEngine, safe_payload


class RuntimeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_fraction: Decimal | None = Field(default=None, gt=0, le=Decimal("0.03"))
    max_drawdown_fraction: Decimal | None = Field(default=None, gt=0, lt=1)
    max_position_usd: Decimal | None = Field(default=None, gt=0)
    max_daily_loss_usd: Decimal | None = Field(default=None, gt=0)


class LifecycleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["start", "pause"]


class ModeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["paper", "live"]


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Literal["smoke", "full"] = "smoke"
    symbol: Literal["BTC/USD", "ETH/USD"] | None = None


class EventHub:
    def __init__(self, max_queue: int = 256):
        self.max_queue = max_queue
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self.max_queue)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, "payload": safe_payload(payload)}
        for queue in tuple(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)


BacktestRunner = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class DashboardController:
    def __init__(
        self,
        engines: dict[Literal["paper", "live"], BaseExecutionEngine],
        *,
        mode: Literal["paper", "live"] = "paper",
        backtest_runner: BacktestRunner | None = None,
    ):
        if mode not in engines:
            raise ValueError("active mode must have a configured execution engine")
        self.engines = engines
        self.mode = mode
        self.hub = EventHub()
        self.backtest_runner = backtest_runner

    @property
    def engine(self) -> BaseExecutionEngine:
        return self.engines[self.mode]

    def state(self) -> dict[str, Any]:
        position = self.engine.position.payload() if self.engine.position else None
        killed = bool((self.engine.journal.get_state("kill_latch") or {}).get("latched", False))
        balances = self.engine.journal.get_state("balances") or {}
        if not balances and hasattr(self.engine, "equity"):
            balances = {"usd": {"available": str(self.engine.equity), "locked": "0"}}
        return safe_payload(
            {
                "mode": self.mode,
                "lifecycle": "killed" if killed else "paused" if self.engine.frozen else "running",
                "flat": self.engine.is_flat,
                "reconciled": not self.engine.open_order_ids,
                "position": position,
                "balances": balances,
                "parameters": asdict(self.engine.risk.params),
            }
        )

    def update_parameters(self, update: RuntimeUpdate) -> dict[str, Any]:
        values = update.model_dump(exclude_none=True)
        params = replace(self.engine.risk.params, **values)
        for engine in self.engines.values():
            engine.risk.params = params
        self.engine.journal.append("runtime_parameters", asdict(params))
        self.hub.publish("parameters", asdict(params))
        return safe_payload(asdict(params))

    def lifecycle(self, action: Literal["start", "pause"]) -> dict[str, Any]:
        killed = bool((self.engine.journal.get_state("kill_latch") or {}).get("latched", False))
        if action == "start" and killed:
            raise PermissionError("kill switch is latched")
        self.engine.frozen = action == "pause"
        self.engine.persist()
        self.hub.publish("state", self.state())
        return self.state()

    def change_mode(self, mode: Literal["paper", "live"]) -> dict[str, Any]:
        if mode == self.mode:
            return self.state()
        if not self.engine.is_flat or self.engine.open_order_ids:
            raise PermissionError("current engine must be flat and reconciled")
        target = self.engines.get(mode)
        if target is None:
            raise PermissionError("requested mode is not configured")
        if mode == "live" and not (
            isinstance(target, LiveExecutionEngine)
            and target.enabled
            and target.is_flat
            and not target.open_order_ids
        ):
            raise PermissionError("live engine must pass approval and reconciliation preflight")
        self.mode = mode
        self.hub.publish("state", self.state())
        return self.state()

    async def kill(self) -> dict[str, Any]:
        report = await KillSwitch(self.engine).trigger("dashboard")
        payload = asdict(report)
        self.hub.publish("kill", payload)
        return payload


def create_app(controller: DashboardController, token: str | None = None) -> FastAPI:
    bearer_token = token or required_secret("DASHBOARD_TOKEN")
    if len(bearer_token) < 16:
        raise ValueError("DASHBOARD_TOKEN must contain at least 16 characters")

    async def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {bearer_token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unauthorized")

    app = FastAPI(title="Tradingbot Bitso", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.controller = controller

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.get("/api/state", dependencies=[Depends(authenticate)])
    async def state_endpoint() -> dict[str, Any]:
        return controller.state()

    @app.get("/api/parameters", dependencies=[Depends(authenticate)])
    async def get_parameters() -> dict[str, Any]:
        return safe_payload(asdict(controller.engine.risk.params))

    @app.put("/api/parameters", dependencies=[Depends(authenticate)])
    async def put_parameters(update: RuntimeUpdate) -> dict[str, Any]:
        return controller.update_parameters(update)

    @app.post("/api/lifecycle", dependencies=[Depends(authenticate)])
    async def lifecycle_endpoint(update: LifecycleUpdate) -> dict[str, Any]:
        try:
            return controller.lifecycle(update.action)
        except PermissionError:
            raise HTTPException(status.HTTP_409_CONFLICT, "lifecycle transition refused") from None

    @app.put("/api/mode", dependencies=[Depends(authenticate)])
    async def mode_endpoint(update: ModeUpdate) -> dict[str, Any]:
        try:
            return controller.change_mode(update.mode)
        except PermissionError:
            raise HTTPException(status.HTTP_409_CONFLICT, "mode transition refused") from None

    @app.post("/api/backtests", dependencies=[Depends(authenticate)])
    async def backtest_endpoint(request: BacktestRequest) -> dict[str, Any]:
        if controller.backtest_runner is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "backtest service unavailable")
        return await controller.backtest_runner(request.model_dump(exclude_none=True))

    @app.post("/api/kill", dependencies=[Depends(authenticate)])
    async def kill_endpoint() -> dict[str, Any]:
        return await controller.kill()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            auth = await asyncio.wait_for(websocket.receive_json(), timeout=5)
            supplied = auth.get("token", "") if isinstance(auth, dict) else ""
            if not secrets.compare_digest(str(supplied), bearer_token):
                await websocket.close(code=1008)
                return
            queue = controller.hub.subscribe()
            await websocket.send_json({"type": "state", "payload": controller.state()})
            disconnected = asyncio.create_task(websocket.receive())
            try:
                while True:
                    event = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait({event, disconnected}, return_when=asyncio.FIRST_COMPLETED)
                    if disconnected in done:
                        event.cancel()
                        return
                    await websocket.send_json(event.result())
            finally:
                disconnected.cancel()
                controller.hub.unsubscribe(queue)
        except (TimeoutError, WebSocketDisconnect):
            return

    return app
