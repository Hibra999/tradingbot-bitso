# tradingbot-bitso

A causal, signal-first reinforcement-learning research and Bitso execution platform for `BTC/USD` and `ETH/USD`. Alpaca supplies target-venue research bars, free Binance spot/futures/funding data supplies causal market context, and deployment orders use matching Bitso `btc_usd` and `eth_usd` books.

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

The first run after this schema upgrade must populate and validate the public Binance context cache and migrate legacy Alpaca crypto timestamps exactly once:

```bash
.venv/bin/python run_quant_pipeline.py --profile smoke --symbol BTC/USD --no-cache-only
```

The full profile uses chronological 36-month train / 6-month validation / 6-month evaluation folds, keeps the newest six complete months sealed, and trains RecurrentPPO and TQC over five seeds by default. SAC and CVaR QR-DQN remain available as explicitly enabled research challengers. A development-qualified algorithm is retrained before the sealed holdout is evaluated once.

```bash
.venv/bin/python run_quant_pipeline.py --profile full
```

Full training and runtime validation belong on the external high-resource machine. This VPS is for static inspection only.

The first full run must also use `--no-cache-only` so Binance context is backfilled for the complete research window; later runs can return to the cache-only default.

Research outputs include per-symbol manifests plus separate training and evaluation QuantStats reports. Both reports compare RL with cost-adjusted buy-and-hold and deterministic alpha; evaluation also includes volatility-matched buy-and-hold. A smoke manifest can never pass promotion.

Each fold first fits Ridge and shallow gradient-boosted 4h/12h/24h alpha experts on training data only. RL receives their forecasts and uncertainty plus observable risk state, and controls one long/cash target exposure. Evaluation uses configured commission and spread assumptions; stress replay doubles both, adds slippage, one-to-three-minute latency, and feature noise. Promotion requires positive alpha controls, paired superiority over deterministic alpha and volatility-matched buy-and-hold, CSCV PBO, a 90% model-confidence set, five-seed IQM bounds, and the existing risk gates.

## Approval and live safety

Full runs may mark one complete model-plus-feature bundle eligible but never select it. An operator must set `selected_artifact` to the bundle model path already listed in `eligible_artifacts`, then set:

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

The service coordinates the Bitso L2 stream, paper/live engine, approved policy inference, Binance public context refresh, FastAPI dashboard, SQLite journal, and optional Telegram bot. It requires 90 shadow days, persists close-labelled Bitso M1 bars, decides only on a closed H1 bar, and executes on the following market update. Feature/hash/action/context mismatches or excessive standardized feature drift freeze policy execution.

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
