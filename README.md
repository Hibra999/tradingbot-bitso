# tradingbot-bitso

A causal, signal-first reinforcement-learning research and Bitso execution platform for `BTC/USD` and `ETH/USD`. Alpaca supplies target-venue research bars, optional free Binance spot/futures/funding data supplies causal market context, and deployment orders use matching Bitso `btc_usd` and `eth_usd` books.

The system defaults to a non-promotable smoke profile and paper execution. It does not promise profitability: failed statistical or risk gates produce reports and block live loading.

## Install

PufferLib 3.0.0 supports Linux and macOS, not native Windows. On a Windows host, run every command below from a WSL2 shell with Linux Conda, `g++`, the Linux NVIDIA CUDA toolkit, and `nvcc` available. Do not use Windows Conda, PowerShell, or `python.exe`. Python 3.11 and every direct and transitive dependency are pinned.

Keep the checkout and training data in WSL's Linux filesystem for maximum I/O throughput. A path below `/mnt/c` works, but is slower than a path below `/home` for Linux workloads. From an existing Windows checkout, copy it once and continue from the Linux copy:

```bash
mkdir -p ~/projects
cp -a /mnt/c/path/to/tradingbot-bitso ~/projects/tradingbot-bitso
cd ~/projects/tradingbot-bitso
```

The Windows NVIDIA driver supplies WSL's CUDA driver. Do not install a Linux display driver inside WSL. The Linux CUDA 13.3 toolkit is still required to compile PufferLib's training extension. Verify the toolchain and expose its default installation path before creating the environment:

```bash
nvidia-smi
nvcc --version
g++ --version
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If `conda` is unavailable in the WSL shell, install the Linux Miniconda distribution; a Windows Conda installation cannot provide the Linux environment.

```bash
conda create --name tradingbot-bitso python=3.11 pip -y
conda activate tradingbot-bitso
python -m pip install numpy==1.26.4 setuptools==84.0.0 torch==2.13.0
bash scripts/install_pufferlib_wsl.sh
NO_OCEAN=1 TORCH_CUDA_ARCH_LIST=12.0 python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python -c "import pufferlib, torch; print('PufferLib:', pufferlib.__version__); print('CUDA:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
cp .env.example .env
```

The bootstrap install is required because PufferLib imports NumPy and PyTorch while compiling its training extension. PufferLib 3.0.0 also references an undefined `c_extension_paths` while preparing `NO_OCEAN=1` metadata. The installer verifies the exact upstream archive, applies the one-line packaging fix in a temporary directory, and installs the resulting wheel before the lock resolves its pinned dependencies. `TORCH_CUDA_ARCH_LIST=12.0` targets the RTX 5070's Blackwell architecture instead of compiling unused GPU targets. Keep `NO_OCEAN=1` to skip unrelated demo environments; do not set `NO_TRAIN`, because PuffeRL requires the compiled advantage extension. The verification command must print `CUDA: True` and the RTX 5070 before training.

Put secrets only in `.env` or the process environment. Data, model artifacts, SQLite journals, reports, notebooks, and caches are ignored by Git.

## Research

The safe default is a short, non-promotable verification run. Smoke uses one purged chronological fold, one seed, shorter feature warm-ups, and 100 Monte Carlo paths; it deliberately skips PuffeRL training and its metrics are diagnostic only.

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

Research outputs include per-symbol manifests plus separate local training and evaluation QuantStats reports. Telegram sends only test/evaluation progress and the evaluation report. Reports display only PuffeRL-LSTM and cost-adjusted Buy & Hold; alpha and volatility-matched controls remain internal promotion gates. A smoke manifest can never pass promotion.

While the research pipeline is running, its status message exposes `Progress`, `Status`, `Help`, and `Clear` buttons; the equivalent `/progress`, `/status`, `/help`, and `/clear` commands remain available. The clear action deletes messages tracked during the current run, then recreates the status message and buttons; Telegram limits deletion to messages under 48 hours and may require group administrator rights. Run only one polling process per bot token: stop the Telegram-enabled live service while a pipeline uses the same bot, or give the two processes different tokens.

Each fold first fits Ridge and shallow gradient-boosted 4h/12h/24h alpha experts on training data only. PuffeRL-LSTM receives their forecasts, causal alpha target, uncertainty, and observable risk state. Its 11 categorical training actions are bounded residual adjustments around that alpha target; deterministic validation and live inference execute their probability-weighted ordered residual so learned probability shifts are measurable before the modal action changes. PPO trains on active, marked, post-cost log return versus the same alpha execution, augmented by bounded Differential Sharpe feedback after a causal warmup and a drawdown-squared penalty; checkpoint selection starts from the zero-residual alpha policy. Training repeatedly samples deterministic contiguous episodes from training segments, so increasing timesteps does not require a longer single history window. Evaluation uses configured commission and spread assumptions; stress replay doubles both, adds slippage, one-to-three-minute latency, and scale-relative feature noise. Promotion requires positive alpha controls, paired superiority over deterministic alpha and volatility-matched buy-and-hold, CSCV PBO, a 90% model-confidence set, five-seed IQM bounds, and the existing risk gates.

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
