# Repository Guidelines

## Project Structure & Module Organization

The repository is organized by domain. `config/` owns environment-backed settings; `data/` handles Alpaca bars and storage; `quant/` contains causal features, fractional differentiation, regimes, and wavelets; `validation/` implements CPCV, metrics, seeds, and Monte Carlo analysis. RL models, Gym environments, training, and promotion governance live in `rl/`. Bitso clients, risk controls, journals, and paper/live engines are under `execution/`. Telegram integration is in `telegram_bot/`, while the FastAPI dashboard and static UI are in `ui/`. Entry points are `run_quant_pipeline.py` and `run_live_service.py`. Tests mirror these domains in `tests/test_*.py`; generated reports belong in `outputs/`.

## Build, Test, and Development Commands

Python 3.11 and the fully pinned lock file are required.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_quant_pipeline.py --profile smoke --symbol BTC/USD
.venv/bin/python run_live_service.py
```

Use the smoke profile for quick, non-promotable research checks. Run `--profile full` only on a CUDA-capable, high-resource machine; it trains all enabled algorithms and performs full validation.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, and `from __future__ import annotations`. Follow existing Python conventions: `snake_case` for functions and modules, `PascalCase` for classes, and uppercase names for constants. Keep financial calculations explicit, deterministic, and causal. Prefer NumPy vectorization over Python array loops, and avoid CPU/GPU transfers inside hot loops. No formatter or linter is configured, so match adjacent code and keep imports grouped.

## Testing Guidelines

Tests use standard-library `unittest`, including `IsolatedAsyncioTestCase` for async services. Name files `test_<domain>.py`, classes `<Domain>Tests`, and methods `test_<behavior>`. Add focused regression coverage for causality, action bounds, risk gates, execution ordering, authentication, and kill-switch behavior. There is no numeric coverage threshold; behavioral safety is the requirement.

## Commit & Pull Request Guidelines

Follow the history's concise Conventional Commit style: `fix: ...`, `perf: ...`, `test(scope): ...`, or `feat(scope): ...`. Keep each commit single-purpose. Pull requests should explain behavioral impact, configuration changes, and safety implications; link the relevant issue and include terminal/UI evidence when output changes.

## Security & Configuration

Never commit or inspect `.env`, credentials, model artifacts, caches, journals, or generated reports. Document variables only in `.env.example` and read runtime values with `os.getenv`. Live trading changes must preserve explicit approval flags, reconciliation, margin checks, and the latched kill switch.
