from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the coordinated Bitso paper/live service.")
    parser.add_argument("--host", default=os.getenv("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8000")))
    parser.add_argument("--no-telegram", action="store_true")
    return parser.parse_args()


async def serve(args: argparse.Namespace) -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 is required")
    if not 1 <= args.port <= 65_535:
        raise ValueError("dashboard port must be in 1..65535")

    import uvicorn
    import pandas as pd
    from dotenv import load_dotenv

    from config import AppConfig, env_flag, required_secret
    from execution import (
        BitsoRESTClient,
        BitsoWebSocketClient,
        BookRules,
        ExecutionJournal,
        LiveExecutionEngine,
        PaperExecutionEngine,
    )
    from rl import LivePolicyRuntime, load_approved_manifest, load_eligible_manifest
    from telegram_bot import BacktestManager, TelegramService, parse_allowed_chat_ids
    from ui import DashboardController, create_app

    load_dotenv()
    config = AppConfig.from_env()
    dashboard_token = required_secret("DASHBOARD_TOKEN")
    journal = ExecutionJournal(config.journal_path)
    live = not config.paper_mode
    rest = BitsoRESTClient(
        config.bitso,
        api_key=required_secret("BITSO_API_KEY") if live else "",
        api_secret=required_secret("BITSO_API_SECRET") if live else "",
    )
    stream = BitsoWebSocketClient(rest, config.bitso.books, config.bitso.websocket_url)
    telegram: TelegramService | None = None
    telegram_started = False
    monitor: asyncio.Task[None] | None = None
    stream_task: asyncio.Task[None] | None = None
    try:
        manifest_path = os.getenv("APPROVED_MODEL_MANIFEST", "").strip()
        manifest = None
        if live:
            manifest = load_approved_manifest(required_secret("APPROVED_MODEL_MANIFEST"))
            engine = LiveExecutionEngine(
                journal,
                rest,
                manifest,
                risk_params=config.risk,
                allow_margin_shorts=config.allow_margin_shorts,
                margin_capability_confirmed=env_flag("BITSO_MARGIN_ACCOUNT_CONFIRMED"),
            )
            await engine.preflight(config.bitso.books)
        else:
            available = await rest.request("GET", "/available_books")
            payloads = {item["book"]: item for item in available}
            if not set(config.bitso.books) <= payloads.keys():
                raise RuntimeError("configured Bitso book is unavailable")
            rules = {book: BookRules.from_payload(payloads[book]) for book in config.bitso.books}
            engine = PaperExecutionEngine(journal, stream.books, rules, risk_params=config.risk)
            if manifest_path:
                manifest = load_eligible_manifest(manifest_path)

        policy = LivePolicyRuntime(manifest, config.data_dir / "bitso_live") if manifest else None
        if policy and policy.book not in stream.books:
            raise RuntimeError("approved policy book is not configured for this service")
        if policy and not engine.is_flat:
            raise RuntimeError("policy inference requires a flat engine at startup")
        policy_position_age = 0
        policy_failed = False
        initial_equity = getattr(engine, "equity", Decimal("0"))
        if not initial_equity:
            usd = (journal.get_state("balances") or {}).get("usd", {})
            initial_equity = Decimal(str(usd.get("available", "0"))) + Decimal(str(usd.get("locked", "0")))

        backtests = BacktestManager(Path(__file__).resolve().parent)
        controller = DashboardController({engine.mode: engine}, mode=engine.mode, backtest_runner=backtests.run)
        candles: dict[str, dict[str, Any]] = {}

        async def execute_policy(closed: dict[str, Any], reference_price: Decimal) -> None:
            nonlocal policy_failed, policy_position_age
            if policy is None or policy_failed or closed["book"] != policy.book:
                return
            try:
                close_time = pd.Timestamp(closed["time"] + 60, unit="s", tz="UTC")
                current_equity = getattr(engine, "equity", initial_equity)
                equity_return = float(current_equity / initial_equity - 1) if initial_equity else 0.0
                decision = policy.on_closed_m1(
                    close_time,
                    closed,
                    position_direction=engine.position.direction if engine.position else 0,
                    position_age_bars=policy_position_age,
                    equity_return=equity_return,
                )
                if decision is None:
                    return
                state_key = f"policy_decision:{policy.book}"
                previous = journal.get_state(state_key) or {}
                if previous.get("decision_time") == str(decision.decision_time):
                    return
                journal.set_state(
                    state_key,
                    {
                        "decision_time": str(decision.decision_time),
                        "direction": decision.intent.direction,
                        "status": "pending",
                    },
                )
                if isinstance(engine, PaperExecutionEngine):
                    await engine.execute(decision.intent, decision.atr)
                else:
                    await engine.execute(decision.intent, decision.atr, reference_price)
                policy_position_age = policy_position_age + 1 if engine.position else 0
                journal.set_state(
                    state_key,
                    {
                        "decision_time": str(decision.decision_time),
                        "direction": decision.intent.direction,
                        "status": "executed",
                    },
                )
                journal.append(
                    "policy_decision",
                    {
                        "book": policy.book,
                        "decision_time": str(decision.decision_time),
                        "direction": decision.intent.direction,
                    },
                )
            except Exception as exc:
                policy_failed = True
                engine.frozen = True
                engine.persist()
                journal.append("policy_failure", {"error_type": type(exc).__name__})
                raise

        async def publish_book(book) -> None:
            bids, asks = book.levels("bids"), book.levels("asks")
            if not bids or not asks:
                return
            price = float((bids[0][0] + asks[0][0]) / Decimal("2"))
            timestamp = int(time.time()) // 60 * 60
            candle = candles.get(book.book)
            if candle is None or candle["time"] != timestamp:
                closed = candle
                candle = {"time": timestamp, "open": price, "high": price, "low": price, "close": price, "book": book.book}
                candles[book.book] = candle
                if closed is not None:
                    await execute_policy(closed, Decimal(str(price)))
            else:
                candle.update(high=max(candle["high"], price), low=min(candle["low"], price), close=price)
            controller.hub.publish("candle", candle)
            if engine.position and engine.position.book == book.book:
                executable = bids[0][0] if engine.position.direction == 1 else asks[0][0]
                if isinstance(engine, PaperExecutionEngine):
                    await engine.trigger_bracket(executable)
                else:
                    await engine.trigger_take_profit(executable)

        async def publish_state() -> None:
            after_id, peak, last_bracket = journal.last_event_id(), Decimal("0"), None
            while True:
                controller.hub.publish("state", controller.state())
                bracket = engine.bracket
                current_bracket = (
                    (bracket.book, bracket.stop_price, bracket.take_profit_price) if bracket else None
                )
                if current_bracket != last_bracket:
                    payload = (
                        {"book": bracket.book, "stop": bracket.stop_price, "take_profit": bracket.take_profit_price}
                        if bracket
                        else {"clear": True}
                    )
                    controller.hub.publish("bracket", payload)
                    last_bracket = current_bracket
                equity = getattr(engine, "equity", None)
                if equity is None:
                    usd = (journal.get_state("balances") or {}).get("usd", {})
                    equity = Decimal(str(usd.get("available", "0"))) + Decimal(str(usd.get("locked", "0")))
                peak = max(peak, equity)
                drawdown = equity / peak - 1 if peak else Decimal("0")
                controller.hub.publish("equity", {"time": int(time.time()), "equity": equity, "drawdown": drawdown})
                events = journal.events(after_id)
                for event in events:
                    after_id = event["id"]
                    controller.hub.publish("log", {"message": event["event_type"]})
                    if event["event_type"] in {"paper_entry", "live_entry", "paper_exit", "live_exit"}:
                        entered = event["event_type"].endswith("entry")
                        controller.hub.publish(
                            "trade",
                            {
                                "time": int(datetime.fromisoformat(event["created_at"]).timestamp()),
                                "book": event["payload"].get("book"),
                                "position": "belowBar" if entered else "aboveBar",
                                "color": "#38d39f" if entered else "#ff627d",
                                "shape": "arrowUp" if entered else "arrowDown",
                                "text": "ENTRY" if entered else "EXIT",
                            },
                        )
                await asyncio.sleep(1)

        stream.on_book_update = publish_book
        app = create_app(controller, dashboard_token)
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
        if not args.no_telegram and os.getenv("TELEGRAM_BOT_TOKEN", "").strip():
            telegram = TelegramService(
                controller,
                token=required_secret("TELEGRAM_BOT_TOKEN"),
                allowed_chat_ids=parse_allowed_chat_ids(required_secret("TELEGRAM_ALLOWED_CHAT_IDS")),
            )
            await telegram.start()
            telegram_started = True
        stream_task = asyncio.create_task(stream.run())
        monitor = asyncio.create_task(publish_state())
        await server.serve()
    finally:
        stream.stop()
        for task in (monitor, stream_task):
            if task:
                task.cancel()
        await asyncio.gather(*(task for task in (monitor, stream_task) if task), return_exceptions=True)
        if telegram and telegram_started:
            await telegram.stop()
        await rest.close()
        journal.close()


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(serve(arguments()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
