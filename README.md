# GLD (US ETF) M1 → RL-ready feature + bracket-trading pipeline

> **Data source: Alpaca** — the pipeline now downloads OHLCV from
> [Alpaca Markets](https://alpaca.markets) instead of expecting a local
> MT4/MT5 CSV. Alpaca does not offer XAUUSD, so the default symbol is **GLD**
> (the gold ETF, the closest US-market substitute). Stocks/ETFs and crypto
> (e.g. `BTC/USD`) are supported.

This project gives you a complete research pipeline:

1. Load and validate raw M1 data.
2. Parse broker/EET time safely.
3. Resample M1 execution data into M5/M15 decision bars.
4. Build causal stationary features normalized by ATR/price.
5. Visualize candles, indicators, volatility, sessions, feature correlations, trades, equity, and drawdown.
6. Tune a simple trend-following baseline on train/validation splits while keeping the test split sealed.
7. Optionally train PPO with a `MultiDiscrete([direction, SL bucket, TP/R bucket])` action space.
8. Validate with **walk-forward** cross-validation (rolling out-of-sample windows) rather than a single train/val split.
9. Reveal the test split only once via `final_holdout_eval.py` after the model is frozen.

## Quick start

```bash
pip install -r requirements.txt
```

1. Create an Alpaca account (free), then in **Paper Trading → API Keys** grab
   your key pair.
2. Open `.env` (copy from `.env.example` if it does not exist) and paste your
   keys:

   ```text
   ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
   ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   The `.env` file is git-ignored; never commit it.

3. Configure the symbol/range in `config.py` (defaults: `GLD`, M1 execution /
   H1 decision, from 2020-01-01 to today; the first download is cached under
   `data/alpaca_GLD_1min.csv`).

Then run:

```bash
python run_pipeline.py
```

This keeps the test split sealed and writes validation-only artifacts, including
a `walk_forward_baseline.csv` showing baseline stability across the rolling folds.

Train the RL agent with walk-forward validation (one model per fold, aggregate
out-of-sample report; the final fold is saved as the production model):

```bash
python train_ppo.py            # == train_ppo.train_walk_forward()
```

When you are ready for the one-time holdout reveal, run:

```bash
python final_holdout_eval.py
```

Or open:

```text
notebooks/XAUUSD_RL_Pipeline_Demo.ipynb
```

## Important defaults

- `data_source = "alpaca"` in `config.py`: OHLCV comes from Alpaca (symbol `GLD`,
  cached in `data/`); set it to `"csv"` to go back to reading an MT4/MT5 file.
- Alpaca bar timestamps are bar-*open*; they are shifted to bar-*close* before
  resampling/backtesting (same convention the MT4 loader used).
- The data index is UTC (Alpaca timezone), whereas the old CSV path used
  Europe/Helsinki. `SOURCE_TZ` still applies to the CSV fallback only.
- Annualisation (`periods_per_year`) automatically switches to US-stock bars
  per year (6.5 h/day, 252 days) when `data_source = "alpaca"`.
- `spread_price`/`slippage_price` defaults were tuned for XAUUSD price levels;
  for GLD (~$200–$300) you may want to shrink them in `config.py`.
- `DECISION_TIMEFRAME = H1`, while M1 is retained for intrabar TP/SL simulation.
- Features are causal and mostly stationary: ATR-normalized distances, ratios, session flags, candle shape ratios.
- The observation set is **25 features**. Four redundant ones were dropped in the 2026-06-02 collinearity audit (`rsi_centered`, `roc5_atr`, `body_atr`, `ema50_slope5_atr`); each was |r| ≥ 0.96 with a retained feature (two were exact duplicates). The underlying `ema20/50/200` columns are still computed for the baselines and charts.
- **Sliding-window walk-forward** (`train_ppo.train_sliding_walk_forward`, the default `python train_ppo.py` entry point) is the realistic validator: each fold trains on `sliding_train_years` (5y), selects its checkpoint on the next `sliding_val_months` (6m), and is judged out-of-sample on the following `sliding_test_months` (6m); the window then slides `sliding_step_months` (6m) and repeats. This simulates *retraining every 6 months and trading the next 6 months live*. Every fold's test window is stitched into one continuous out-of-sample equity curve (`models/sliding_oos_equity.csv`); the gate runs on the **test** metrics. Because every train window is the same length, equal timesteps per fold = equal passes. ~35 folds over the 23y dataset (raise `sliding_step_months` to reduce).
- **Block walk-forward** (`train_ppo.train_walk_forward`; `config.py`: `n_walk_forward_folds`, `walk_forward_anchored`, `test_frac`) is the alternative: it seals the last `test_frac` of bars and rolls `n_folds` train→val windows over the rest. `test_frac == 1 - train_frac - val_frac`, so the sealed holdout matches the single-split test exactly.
- When TP and SL are both inside the same M1 candle, the simulator assumes SL first. This is deliberately pessimistic.
- Position size is fixed-fractional risk-based; the RL agent controls direction and bracket shape, not size.
- `run_pipeline.py` now performs a temporal train/validation/test split, tunes on train/val, and keeps test sealed by default.
- `training_diagnostics.py` and `train_ppo.py` also keep the test split sealed by default.
- RL rewards are normalized by risk budget rather than raw cash PnL, which is materially more stable for PPO.

## Files

- `config.py`: all main parameters (data source, symbol, timeframes, validation, PPO).
- `alpaca_data.py`: Alpaca Markets downloader with pagination + local CSV cache.
- `data_loader.py`: data-source dispatch (`load_ohlcv`), CSV parsing, timezone handling, validation, resampling.
- `features.py`: causal indicators and stationary feature matrix.
- `leakage_checks.py`: simple future-append stability test.
- `env_bracket.py`: Gymnasium-compatible bracket trading environment.
- `baselines.py`: random and EMA/ATR rule policies.
- `evaluate.py`: metrics, trade-log summary, drawdown.
- `visualize.py`: Plotly visualization functions.
- `train_ppo.py`: optional PPO training scaffold.
- `run_pipeline.py`: one-command pre-test pipeline with validation-only outputs. Prints CUDA/GPU status, shows a progress bar for every step, and writes a single combined HTML report at `outputs/00_combined_report.html` alongside the per-chart files.
- `final_holdout_eval.py`: explicit one-time holdout evaluation entry point.
- `notebooks/XAUUSD_RL_Pipeline_Demo.ipynb`: guided notebook.
