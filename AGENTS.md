# Repository Guidelines

## Project Structure & Module Organization

The repository is organized by domain. `config/` owns environment-backed settings; `data/` handles Alpaca bars and storage; `quant/` contains causal features, fractional differentiation, regimes, and wavelets; `validation/` implements CPCV, metrics, seeds, and Monte Carlo analysis. RL models, Gym environments, training, and promotion governance live in `rl/`. Bitso clients, risk controls, journals, and paper/live engines are under `execution/`. Telegram integration is in `telegram_bot/`, while the FastAPI dashboard and static UI are in `ui/`. Entry points are `run_quant_pipeline.py` and `run_live_service.py`. Tests mirror these domains in `tests/test_*.py`; generated reports belong in `outputs/`.

## Build, Test, and Development Commands

Python 3.11, WSL2/Linux, `g++`, `nvcc`, and the fully pinned lock file are required. Native Windows is unsupported because PufferLib 3.0.0 rejects it. PufferLib must be built with its training extension; `NO_OCEAN=1` skips only unrelated demo environments.

```bash
conda create --name tradingbot-bitso python=3.11 pip -y
conda activate tradingbot-bitso
python -m pip install numpy==1.26.4 setuptools==84.0.0 torch==2.13.0
bash scripts/install_pufferlib_wsl.sh
NO_OCEAN=1 TORCH_CUDA_ARCH_LIST=12.0 python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python run_quant_pipeline.py --profile smoke --symbol BTC/USD
python run_live_service.py
```

Use the smoke profile for quick, non-promotable research checks. Run `--profile full` only on a CUDA-capable, high-resource machine; it trains the single recurrent PuffeRL-LSTM agent and performs full validation.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, and `from __future__ import annotations`. Follow existing Python conventions: `snake_case` for functions and modules, `PascalCase` for classes, and uppercase names for constants. Keep financial calculations explicit, deterministic, and causal. Prefer NumPy vectorization over Python array loops, and avoid CPU/GPU transfers inside hot loops. No formatter or linter is configured, so match adjacent code and keep imports grouped.

## Testing Guidelines

Tests use standard-library `unittest`, including `IsolatedAsyncioTestCase` for async services. Name files `test_<domain>.py`, classes `<Domain>Tests`, and methods `test_<behavior>`. Add focused regression coverage for causality, action bounds, risk gates, execution ordering, authentication, and kill-switch behavior. There is no numeric coverage threshold; behavioral safety is the requirement.

## Commit & Pull Request Guidelines

Follow the history's concise Conventional Commit style: `fix: ...`, `perf: ...`, `test(scope): ...`, or `feat(scope): ...`. Keep each commit single-purpose. Pull requests should explain behavioral impact, configuration changes, and safety implications; link the relevant issue and include terminal/UI evidence when output changes.

## Security & Configuration

Never commit or inspect `.env`, credentials, model artifacts, caches, journals, or generated reports. Document variables only in `.env.example` and read runtime values with `os.getenv`. Live trading changes must preserve explicit approval flags, reconciliation, margin checks, and the latched kill switch.
