from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from decimal import Decimal
from typing import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from execution import TradeIntent
from ui.app import DashboardController, RuntimeUpdate


def parse_allowed_chat_ids(value: str) -> frozenset[int]:
    try:
        result = frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("TELEGRAM_ALLOWED_CHAT_IDS must contain comma-separated integers") from exc
    if not result:
        raise ValueError("TELEGRAM_ALLOWED_CHAT_IDS must not be empty")
    return result


def format_entry_ticket(intent: TradeIntent, *, quantity: Decimal, price: Decimal) -> str:
    side = "LONG" if intent.direction == 1 else "SHORT"
    return (
        f"ENTRY {side} {intent.book}\n"
        f"Quantity: {quantity}\nPrice: {price}\nRisk: {intent.risk_fraction}\n"
        f"SL: {intent.sl_atr_multiplier} ATR · TP/R: {intent.tp_sl_ratio}\n"
        f"Confidence: {intent.confidence:.1%}\nModel: {intent.model_id}"
    )


def format_exit_ticket(*, book: str, quantity: Decimal, price: Decimal, pnl: Decimal, reason: str) -> str:
    return f"EXIT {book}\nQuantity: {quantity}\nPrice: {price}\nPnL: {pnl}\nReason: {reason}"


class TelegramService:
    def __init__(
        self,
        controller: DashboardController,
        *,
        token: str,
        allowed_chat_ids: frozenset[int],
        queue_size: int = 128,
    ):
        if not token.strip() or not allowed_chat_ids:
            raise ValueError("Telegram token and chat allowlist are required")
        self.controller = controller
        self.allowed_chat_ids = allowed_chat_ids
        self._secret_values = {
            value
            for value in (
                token,
                os.getenv("ALPACA_API_KEY"),
                os.getenv("ALPACA_SECRET_KEY"),
                os.getenv("BITSO_API_KEY"),
                os.getenv("BITSO_API_SECRET"),
                os.getenv("DASHBOARD_TOKEN"),
            )
            if value
        }
        self.alerts: asyncio.Queue[str] = asyncio.Queue(queue_size)
        self.application = Application.builder().token(token).build()
        self._worker: asyncio.Task[None] | None = None
        for command, handler in {
            "status": self._status,
            "balance": self._balance,
            "backtest": self._backtest,
            "params": self._params,
            "set_risk": self._set_risk,
            "kill": self._kill,
        }.items():
            self.application.add_handler(CommandHandler(command, self._authorized(handler)))

    def _authorized(
        self,
        handler: Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]],
    ) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
        async def guarded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            chat = update.effective_chat
            if chat is None or chat.id not in self.allowed_chat_ids:
                return
            try:
                await handler(update, context)
            except (PermissionError, ValueError):
                await update.effective_message.reply_text("Request rejected.")
            except Exception:
                await update.effective_message.reply_text("Command failed.")

        return guarded

    async def _status(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        state = self.controller.state()
        await update.effective_message.reply_text(
            f"Mode: {state['mode']}\nState: {state['lifecycle']}\nFlat: {state['flat']}\nReconciled: {state['reconciled']}"
        )

    async def _balance(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        equity = getattr(self.controller.engine, "equity", None)
        balances = self.controller.engine.journal.get_state("balances") or {}
        text = f"Paper equity: {equity}" if equity is not None else "Balances: unavailable"
        if balances:
            text = "Balances:\n" + "\n".join(f"{key}: {value}" for key, value in sorted(balances.items()))
        await update.effective_message.reply_text(text)

    async def _backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.controller.backtest_runner is None:
            raise PermissionError("backtest service unavailable")
        if len(context.args) > 2:
            raise ValueError("too many backtest parameters")
        profile = context.args[0] if context.args else "smoke"
        symbol = context.args[1].upper() if len(context.args) > 1 else None
        result = await self.controller.backtest_runner({"profile": profile, "symbol": symbol})
        await update.effective_message.reply_text(f"Backtest: {result['status']}")

    async def _params(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        params = asdict(self.controller.engine.risk.params)
        await update.effective_message.reply_text("\n".join(f"{key}: {value}" for key, value in params.items()))

    async def _set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if len(context.args) != 1:
            raise ValueError("one risk fraction is required")
        params = self.controller.update_parameters(RuntimeUpdate(risk_fraction=Decimal(context.args[0])))
        await update.effective_message.reply_text(f"Risk fraction: {params['risk_fraction']}")

    async def _kill(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        report = await self.controller.kill()
        await update.effective_message.reply_text(
            f"KILL latched\nDispatch: {report['dispatch_latency_ms']:.1f} ms\n"
            f"Exchange acknowledgement: {report['exchange_ack_latency_ms']:.1f} ms\n"
            f"Confirmed flat: {report['confirmed_flat']}"
        )

    def alert_nowait(self, text: str) -> None:
        for value in self._secret_values:
            text = text.replace(value, "[redacted]")
        if self.alerts.full():
            self.alerts.get_nowait()
        self.alerts.put_nowait(text[:4000])

    async def _alert_worker(self) -> None:
        while True:
            text = await self.alerts.get()
            for chat_id in sorted(self.allowed_chat_ids):
                try:
                    await self.application.bot.send_message(chat_id, text)
                except Exception:
                    continue

    async def start(self) -> None:
        await self.application.initialize()
        await self.application.start()
        if self.application.updater:
            await self.application.updater.start_polling(drop_pending_updates=True)
        self._worker = asyncio.create_task(self._alert_worker())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        if self.application.updater:
            await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()
