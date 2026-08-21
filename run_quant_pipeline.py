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
    parser.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if sys.version_info[:2] < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required")

    import numpy as np

    from config import AppConfig
    from data import load_alpaca_ohlcv, resample_ohlcv
    from rl import CandidateRun, SB3CandidateRunner, TrainingEngine
    from telegram_bot import PipelineNotifier
    from telegram_bot.reports import generate_full_report
    from validation import moving_block_monte_carlo

    config = AppConfig.from_env(profile=args.profile)
    if args.cache_only is not None:
        config = replace(config, cache_only=args.cache_only)
    notifier = PipelineNotifier()

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

    runner = (
        SB3CandidateRunner(config.rl.timesteps, config.rl.evaluations, notifier)
        if args.profile == "full"
        else SmokeRunner()
    )
    symbols = (args.symbol,) if args.symbol else config.research_symbols
    started = time.monotonic()
    notifier.start_updates(f"Pipeline starting | {args.profile} | {', '.join(symbols)}")
    try:
        for symbol in symbols:
            notifier.notify_phase("Data loading", symbol, f"cache_only={config.cache_only}")
            m1 = load_alpaca_ohlcv(
                symbol,
                "1min",
                max_days_for_demo=None if args.profile == "full" else 60,
                cache_dir=config.data_dir,
                timestamp_is_bar_open=False,
                cache_only=config.cache_only,
            )
            notifier.notify_phase("Resampling", symbol, f"{len(m1)} M1 bars")
            decision = resample_ohlcv(m1, "1h")
            notifier.notify_phase("Training", symbol, f"{len(decision)} H1 bars")
            manifest = TrainingEngine(config, runner=runner, notifier=notifier).run_symbol(symbol, decision, m1)
            observed_returns = decision["Close"].pct_change().dropna()
            notifier.notify_phase("Monte Carlo", symbol, f"{config.validation.monte_carlo_paths} paths")
            monte_carlo = moving_block_monte_carlo(
                observed_returns.to_numpy(),
                paths=config.validation.monte_carlo_paths,
                block_size=min(24, len(observed_returns)),
            )
            destination = config.outputs_dir / f"{symbol.replace('/', '_')}_report.html"
            notifier.notify_phase("Reporting", symbol)
            report = generate_full_report(
                observed_returns,
                monte_carlo,
                destination,
                title=f"{symbol} {args.profile.title()} Backtest",
                symbol=symbol,
            )
            notifier.notify(report["text_report"])
            notifier.send_photo(report["graphics"], f"{symbol} backtest graphics")
            notifier.send_document(report["html"], f"{symbol} full QuantStats report")
            notifier.send_document(report["latex"], f"{symbol} LaTeX tables")
            print(
                json.dumps(
                    {
                        "symbol": symbol,
                        "profile": args.profile,
                        "eligible": manifest["eligible"],
                        "report": str(report["html"]),
                        "advanced_metrics": report["metrics"],
                    },
                    sort_keys=True,
                )
            )
        notifier.stop_updates(f"Pipeline complete | {time.monotonic() - started:.1f}s")
    except Exception:
        notifier.stop_updates(f"Pipeline failed | {time.monotonic() - started:.1f}s")
        raise
    finally:
        notifier.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
