# tradingbot-bitso

A causal, signal-first reinforcement-learning research and Bitso execution platform for `BTC/USD` and `ETH/USD`. Alpaca supplies target-venue research bars, optional free Binance spot/futures/funding data supplies causal market context, and deployment orders use matching Bitso `btc_usd` and `eth_usd` books.

The system defaults to a non-promotable smoke profile and paper execution. It does not promise profitability: failed statistical or risk gates produce reports and block live loading.

## Install

PufferLib 3.0.0 supports Linux and macOS, not native Windows. On a Windows host, run this repository inside WSL2 with `g++`, the Linux NVIDIA CUDA toolkit, and `nvcc` available. Python 3.11 and every direct and transitive dependency are pinned.

```bash
conda create --name tradingbot-bitso python=3.11 -y
conda activate tradingbot-bitso
python -m pip install numpy==1.26.4 setuptools==84.0.0 torch==2.13.0
NO_OCEAN=1 python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
cp .env.example .env
```

The bootstrap install is required because PufferLib imports NumPy and PyTorch while compiling its training extension. Keep `NO_OCEAN=1` to skip unrelated demo environments; do not set `NO_TRAIN`, because PuffeRL requires the compiled advantage extension. Verify `g++ --version`, `nvidia-smi`, and `nvcc --version` inside WSL before installation so the extension builds with CUDA instead of the CPU fallback.

Put secrets only in `.env` or the process environment. Data, model artifacts, SQLite journals, reports, notebooks, and caches are ignored by Git.

## Research

The safe default is a short, non-promotable verification run. Smoke uses one purged chronological fold, one seed, shorter feature warm-ups, and 100 Monte Carlo paths; its metrics are diagnostic only.

```bash
python run_quant_pipeline.py
python run_quant_pipeline.py --profile smoke --symbol BTC/USD
```

To enable the optional public Binance context, set `BINANCE_CONTEXT_ENABLED=true`. Its separate cache must then be populated once alongside any legacy Alpaca timestamp migration:

```bash
python run_quant_pipeline.py --profile smoke --symbol BTC/USD --no-cache-only
```

The full profile uses chronological 36-month train / 6-month validation / 6-month evaluation folds, keeps the newest six complete months sealed, and trains the recurrent `PuffeRL-LSTM` agent over five seeds by default. A development-qualified agent is retrained before the sealed holdout is evaluated once.

```bash
python run_quant_pipeline.py --profile full
```

Full training and runtime validation belong on the external high-resource machine. This VPS is for static inspection only.

When Binance context is enabled, its first full run must use `--no-cache-only` to backfill the complete research window; later runs can return to the cache-only default. With the default disabled, existing Alpaca-only caches require no Binance download.

Research outputs include per-symbol manifests plus separate training and evaluation QuantStats reports. Both reports compare PuffeRL-LSTM with cost-adjusted buy-and-hold and deterministic alpha; evaluation also includes volatility-matched buy-and-hold. A smoke manifest can never pass promotion.

Each fold first fits Ridge and shallow gradient-boosted 4h/12h/24h alpha experts on training data only. PuffeRL-LSTM receives their forecasts and uncertainty plus observable risk state, and controls one long/cash target exposure. Evaluation uses configured commission and spread assumptions; stress replay doubles both, adds slippage, one-to-three-minute latency, and feature noise. Promotion requires positive alpha controls, paired superiority over deterministic alpha and volatility-matched buy-and-hold, CSCV PBO, a 90% model-confidence set, five-seed IQM bounds, and the existing risk gates.

## Approval and live safety

Full runs may mark one complete model-plus-feature bundle eligible but never select it. Manifest schema 4 accepts only PuffeRL-LSTM bundles, so older SB3 artifacts cannot be approved. An operator must set `selected_artifact` to the bundle model path already listed in `eligible_artifacts`, then set:

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
python run_live_service.py
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
python -m unittest discover -s tests -v
```
