from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal CPCV research for Alpaca BTC/USD and ETH/USD data.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--symbol", choices=("BTC/USD", "ETH/USD"))
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 is required")

    import numpy as np
    from dotenv import load_dotenv

    from config import AppConfig, ValidationConfig
    from data import load_alpaca_ohlcv, resample_ohlcv
    from rl import CandidateRun, SB3CandidateRunner, TrainingEngine
    from telegram_bot import generate_report
    from validation import moving_block_monte_carlo

    load_dotenv()
    validation = ValidationConfig()
    if args.profile == "smoke":
        validation = replace(validation, temporal_groups=3, test_groups=1, embargo_bars=24, monte_carlo_paths=100)
    config = AppConfig(profile=args.profile, validation=validation)

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

    runner = SB3CandidateRunner() if args.profile == "full" else SmokeRunner()
    symbols = (args.symbol,) if args.symbol else config.research_symbols
    for symbol in symbols:
        m1 = load_alpaca_ohlcv(
            symbol,
            "1min",
            max_days_for_demo=None if args.profile == "full" else 60,
            cache_dir=config.data_dir,
            timestamp_is_bar_open=False,
        )
        decision = resample_ohlcv(m1, "1h")
        manifest = TrainingEngine(config, runner=runner).run_symbol(symbol, decision, m1)
        observed_returns = decision["Close"].pct_change().dropna()
        monte_carlo = moving_block_monte_carlo(
            observed_returns.to_numpy(),
            paths=validation.monte_carlo_paths,
            block_size=min(24, len(observed_returns)),
        )
        generate_report(observed_returns, monte_carlo, config.outputs_dir / f"{symbol.replace('/', '_')}_report.png")
        print(json.dumps({"symbol": symbol, "profile": args.profile, "eligible": manifest["eligible"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
