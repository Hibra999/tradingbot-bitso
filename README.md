# tradingbot-bitso

A causal, CPCV-validated reinforcement-learning research and Bitso execution platform for `BTC/USD` and `ETH/USD`. Research data comes only from Alpaca; deployment market data and orders come only from matching Bitso `btc_usd` and `eth_usd` books.

The system defaults to a non-promotable smoke profile and paper execution. It does not promise profitability: failed statistical or risk gates produce reports and block live loading.

## Install

Python 3.11 is required. Every direct and transitive dependency is pinned.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put secrets only in `.env` or the process environment. Data, model artifacts, SQLite journals, reports, notebooks, and caches are ignored by Git.

## Research

The safe default is a short, non-promotable verification run:

```bash
.venv/bin/python run_quant_pipeline.py
.venv/bin/python run_quant_pipeline.py --profile smoke --symbol BTC/USD
```

The full profile downloads the complete Alpaca M1 history, fits each feature pipeline inside every CPCV fold, trains RecurrentPPO, SAC, and CVaR QR-DQN independently for each symbol over 10 explicit seeds, and runs 5,000 moving-block Monte Carlo paths:

```bash
.venv/bin/python run_quant_pipeline.py --profile full
```

Full training belongs on the external high-resource machine. This VPS is intended for compile checks and the bounded safety suite.

Research outputs include per-symbol manifests and PNG reports with observed/Monte Carlo equity, underwater drawdown, monthly returns, and return distributions. A smoke manifest can never pass promotion.

## Approval and live safety

Full runs may mark artifacts eligible but never select one. An operator must set `selected_artifact` to one path already listed in `eligible_artifacts`, then set:

```text
MODEL_APPROVED=true
APPROVED_MODEL_MANIFEST=/absolute/path/to/full_manifest.json
```

Live execution additionally requires:

```text
TRADING_MODE=live
BITSO_LIVE_ENABLED=true
BITSO_API_KEY=...
BITSO_API_SECRET=...
```

Shorts remain disabled unless `BITSO_MARGIN_SHORTS_ENABLED=true`, `BITSO_MARGIN_ACCOUNT_CONFIRMED=true`, and book margin capability passes preflight. Spot brackets keep the stop at Bitso and the take-profit synthetic to avoid double-locking assets. Kill is latched across restarts and never auto-resumes.

## Service

Set a random `DASHBOARD_TOKEN` of at least 16 characters, then run:

```bash
.venv/bin/python run_live_service.py
```

The service coordinates the Bitso L2 stream, paper/live engine, FastAPI dashboard, SQLite journal, and optional Telegram bot in one asyncio loop. It binds to `127.0.0.1:8000` by default. Remote binding must be explicit and all REST calls still require `Authorization: Bearer <DASHBOARD_TOKEN>`; the WebSocket authenticates in its first message.

For Telegram, set `TELEGRAM_BOT_TOKEN` and a comma-separated `TELEGRAM_ALLOWED_CHAT_IDS`. Authorized chats receive `/status`, `/balance`, `/backtest`, `/params`, `/set_risk`, and immediate `/kill`; unauthorized chats are ignored.

## Safety invariants

- Features, scaling, wavelets, and HMM probabilities are causal and fitted only on training data.
- Decision H1 bars fill no earlier than the following M1 tick; ambiguous SL/TP bars resolve SL first.
- CPCV purges holding-interval overlap, applies embargo, and resets every disjoint episode flat.
- Paper and live engines share `TradeIntent`, Decimal risk checks, durable state, bracket logic, and kill path.
- Live loading requires full-profile gates, an eligible operator-selected artifact, approval flags, balance/book/fee/order reconciliation, and a flat engine.
- Kill reports local dispatch, exchange acknowledgement, and confirmed-flat latency separately.

Run the deterministic suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
