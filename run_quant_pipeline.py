from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal CPCV research for Alpaca BTC/USD and ETH/USD data.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--symbol", choices=("BTC/USD", "ETH/USD"))
    parser.add_argument(
        "--cache-only",
        action="store_true",
        default=None,
        help="Use only cached data (no API downloads). Falls back to CACHE_ONLY env var.",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if sys.version_info[:2] < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required")

    import os

    import numpy as np
    from dotenv import load_dotenv
    from tqdm import tqdm

    from config import AppConfig, RLConfig, ValidationConfig
    from data import load_alpaca_ohlcv, resample_ohlcv
    from rl import CandidateRun, SB3CandidateRunner, TrainingEngine
    from telegram_bot.notifier import PipelineNotifier
    from telegram_bot.reports import generate_report
    from validation import moving_block_monte_carlo

    load_dotenv()

    # Resolve cache-only: CLI flag takes priority, then env var, default False
    if args.cache_only is not None:
        cache_only = args.cache_only
    else:
        cache_only = os.getenv("CACHE_ONLY", "false").strip().lower() in ("1", "true", "yes", "on")

    validation = ValidationConfig()
    if args.profile == "smoke":
        validation = replace(validation, temporal_groups=3, test_groups=1, embargo_bars=24, monte_carlo_paths=100)
    
    rl_config = RLConfig.from_env()
    config = AppConfig(profile=args.profile, validation=validation, rl=rl_config)

    # Initialize Telegram notifier (silent no-op if credentials missing)
    notifier = PipelineNotifier()
    if notifier.enabled:
        tqdm.write("  Telegram notifications: ON")
    else:
        tqdm.write("  Telegram notifications: OFF (set TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_CHAT_IDS to enable)")

    class SmokeRunner:
        def __call__(self, dataset) -> CandidateRun:
            returns = np.concatenate(
                [segment["Close"].pct_change().dropna().to_numpy(dtype=float) for segment in dataset.test_segments]
            )
            artifact = dataset.output_dir / "smoke-verification.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps({"profile": "smoke", "promotable": False}), encoding="utf-8")
            validation_return = float(dataset.validation_segment["Close"].pct_change().dropna().sum())
            return CandidateRun(tuple(returns), str(artifact), validation_return)

    runner = SB3CandidateRunner(
        timesteps=config.rl.timesteps,
        evaluations=config.rl.evaluations
    ) if args.profile == "full" else SmokeRunner()
    symbols = (args.symbol,) if args.symbol else config.research_symbols

    pipeline_start = time.perf_counter()
    symbol_bar = tqdm(symbols, desc="Pipeline", unit="symbol")

    try:
        for symbol in symbol_bar:
            symbol_bar.set_description(f"Pipeline [{symbol}]")

            # Notify pipeline start
            notifier.notify(
                f"<b>-- BACKTEST STARTED --</b>\n"
                f"Symbol: {symbol}\n"
                f"Profile: {args.profile}\n"
                f"Cache-only: {cache_only}"
            )

            # Phase 1: Data Loading
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"  {symbol} -- Profile: {args.profile} -- Cache-only: {cache_only}")
            tqdm.write(f"{'='*60}")
            phase_bar = tqdm(total=5, desc=f"  {symbol} phases", unit="phase", leave=False,
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}")
            phase_bar.set_postfix_str("[1/5] Loading data...")
            m1 = load_alpaca_ohlcv(
                symbol,
                "1min",
                max_days_for_demo=None if args.profile == "full" else 60,
                cache_dir=config.data_dir,
                timestamp_is_bar_open=False,
                cache_only=cache_only,
            )
            tqdm.write(f"  [OK] Data loaded: {len(m1):,} 1-min bars")
            notifier.notify_phase("[1/5] Data loaded", symbol, f"{len(m1):,} 1-min bars")
            phase_bar.update(1)

            # Phase 2: Resample
            phase_bar.set_postfix_str("[2/5] Resampling 1min->1h...")
            decision = resample_ohlcv(m1, "1h")
            tqdm.write(f"  [OK] Resampled: {len(decision):,} 1h bars")
            notifier.notify_phase("[2/5] Resampled", symbol, f"{len(decision):,} 1h bars")
            phase_bar.update(1)

            # Phase 3: Training Engine
            phase_bar.set_postfix_str("[3/5] Training engine...")
            notifier.notify_phase("[3/5] Training", symbol, "Starting algorithm training...")
            engine = TrainingEngine(config, runner=runner, notifier=notifier)
            manifest = engine.run_symbol(symbol, decision, m1)
            tqdm.write(f"  [OK] Training complete -- eligible: {manifest['eligible']}")
            notifier.notify_phase("[3/5] Training complete", symbol, f"Eligible: {manifest['eligible']}")
            phase_bar.update(1)

            # Phase 4: Monte Carlo
            phase_bar.set_postfix_str("[4/5] Monte Carlo simulation...")
            observed_returns = decision["Close"].pct_change().dropna()
            monte_carlo = moving_block_monte_carlo(
                observed_returns.to_numpy(),
                paths=validation.monte_carlo_paths,
                block_size=min(24, len(observed_returns)),
            )
            tqdm.write(f"  [OK] Monte Carlo: {validation.monte_carlo_paths} paths, ruin_20={monte_carlo.ruin_probability_20:.2%}")
            notifier.notify_phase(
                "[4/5] Monte Carlo", symbol,
                f"Paths: {validation.monte_carlo_paths}\n"
                f"Ruin 20%: {monte_carlo.ruin_probability_20:.2%}\n"
                f"Ruin 30%: {monte_carlo.ruin_probability_30:.2%}"
            )
            phase_bar.update(1)

            # Phase 5: Report
            phase_bar.set_postfix_str("[5/5] Generating report...")
            report_path = config.outputs_dir / f"{symbol.replace('/', '_')}_report.png"
            generate_report(observed_returns, monte_carlo, report_path)
            tqdm.write(f"  [OK] Report saved: {report_path}")
            notifier.send_photo(report_path, caption=f"{symbol} | {args.profile} | Eligible: {manifest['eligible']}")
            phase_bar.update(1)
            phase_bar.close()

            print(json.dumps({"symbol": symbol, "profile": args.profile, "eligible": manifest["eligible"]}))

        elapsed = time.perf_counter() - pipeline_start
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"  Pipeline complete in {elapsed:.1f}s")
        tqdm.write(f"{'='*60}")

        notifier.notify(
            f"<b>-- BACKTEST COMPLETE --</b>\n"
            f"Time: {elapsed:.1f}s\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Profile: {args.profile}"
        )
    finally:
        notifier.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
