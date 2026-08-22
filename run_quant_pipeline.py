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

    import pandas as pd
    from dotenv import load_dotenv
    from tqdm import tqdm

    from config import AppConfig
    from data import load_alpaca_ohlcv, load_binance_context, resample_ohlcv
    from quant import HAR_RV_COLUMNS, add_realized_volatility_features, align_m1_features_to_decisions
    from rl import CandidateRun, SB3CandidateRunner, TrainingEngine
    from telegram_bot import PipelineNotifier
    from telegram_bot.reports import generate_full_report
    from validation import moving_block_monte_carlo

    load_dotenv(override=False)
    config = AppConfig.from_env(profile=args.profile)
    if args.cache_only is not None:
        config = replace(config, cache_only=args.cache_only)
    notifier = PipelineNotifier()
    tqdm.write(f"Telegram progress: {'ON' if notifier.enabled else 'OFF'}")

    class SmokeRunner:
        def __call__(self, dataset) -> CandidateRun:
            def buy_and_hold(segments):
                series = []
                cost = 2 * dataset.commission_rate + dataset.base_spread_bps / 10_000
                for segment in segments:
                    returns = segment["Close"].pct_change().dropna().astype(float)
                    if len(returns):
                        returns.iloc[0] -= cost / 2
                        returns.iloc[-1] -= cost / 2
                    series.append(returns)
                return pd.concat(series).sort_index()

            def deterministic(segments):
                series = []
                cost = 2 * dataset.commission_rate + dataset.base_spread_bps / 10_000
                for segment in segments:
                    exposure = segment[dataset.deterministic_column].astype(float)
                    market_return = segment["Close"].pct_change().astype(float)
                    turnover = exposure.diff().abs().fillna(exposure.abs())
                    returns = exposure.shift() * market_return - turnover.shift() * cost / 2
                    if len(returns):
                        returns.iloc[-1] -= exposure.shift().iloc[-1] * cost / 2
                    series.append(returns.iloc[1:])
                return pd.concat(series).sort_index()

            evaluation = buy_and_hold(dataset.test_segments)
            training = buy_and_hold(dataset.training_segments)
            evaluation_deterministic = deterministic(dataset.test_segments)
            training_deterministic = deterministic(dataset.training_segments)
            artifact = dataset.output_dir / "smoke-verification.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps({"profile": "smoke", "promotable": False}), encoding="utf-8")
            validation_return = float(dataset.validation_segment["Close"].pct_change().dropna().sum())
            return CandidateRun(
                tuple(evaluation.to_numpy()),
                str(artifact),
                validation_return,
                tuple(str(value) for value in evaluation.index),
                tuple(training.to_numpy()),
                tuple(str(value) for value in training.index),
                tuple(evaluation.to_numpy()),
                tuple(training.to_numpy()),
                (validation_return,),
                deterministic_returns=tuple(evaluation_deterministic.to_numpy()),
                training_deterministic_returns=tuple(training_deterministic.to_numpy()),
            )

    runner = (
        SB3CandidateRunner(
            config.rl.timesteps,
            config.rl.evaluations,
            notifier,
            config.rl.recurrent_ppo_envs,
            config.rl.off_policy_envs,
        )
        if args.profile == "full"
        else SmokeRunner()
    )
    symbols = (args.symbol,) if args.symbol else config.research_symbols
    started = time.monotonic()
    notifier.start_updates(f"Pipeline starting | {args.profile} | {', '.join(symbols)}")
    try:
        with tqdm(symbols, desc=f"Pipeline {args.profile}", unit="symbol", dynamic_ncols=True) as symbol_bar:
            for symbol in symbol_bar:
                symbol_bar.set_postfix_str(symbol)
                with tqdm(total=5, desc=f"{symbol} phases", leave=False, dynamic_ncols=True) as phase_bar:
                    phase_bar.set_postfix_str("data loading")
                    notifier.notify_phase("Data loading", symbol, f"cache_only={config.cache_only}")
                    m1 = load_alpaca_ohlcv(
                        symbol,
                        "1min",
                        max_days_for_demo=None if args.profile == "full" else 60,
                        cache_dir=config.data_dir,
                        timestamp_is_bar_open=True,
                        cache_only=config.cache_only,
                    )
                    phase_bar.update()
                    phase_bar.set_postfix_str("resampling")
                    notifier.notify_phase("Resampling", symbol, f"{len(m1):,} M1 bars")
                    decision = resample_ohlcv(m1, "1h")
                    decision = decision.loc[decision.index <= m1.index.max().floor("h")]
                    realized = add_realized_volatility_features(m1, include_moments=False)
                    decision = decision.join(
                        align_m1_features_to_decisions(realized[list(HAR_RV_COLUMNS)], decision.index)
                    )
                    if config.binance_context_enabled:
                        decision = decision.join(
                            load_binance_context(
                                symbol,
                                decision.index,
                                cache_dir=config.data_dir,
                                cache_only=config.cache_only,
                            )
                        )
                    phase_bar.update()
                    phase_bar.set_postfix_str("training")
                    notifier.notify_phase("Training", symbol, f"{len(decision):,} H1 bars")
                    manifest = TrainingEngine(config, runner=runner, notifier=notifier).run_symbol(symbol, decision, m1)
                    phase_bar.update()
                    reporting = manifest.pop("_reporting")
                    algorithm = str(reporting["algorithm"])
                    agent_name = {
                        "recurrent_ppo": "Recurrent PPO",
                        "sac": "SAC",
                        "tqc": "TQC",
                        "cvar_qrdqn": "CVaR QR-DQN",
                        "pufferl": "PuffeRL-LSTM",
                    }.get(algorithm, algorithm.replace("_", " ").title())
                    training_returns = reporting["training"]
                    evaluation_returns = reporting["evaluation"]
                    monte_carlo_paths = (
                        config.validation.monte_carlo_paths
                        if args.profile == "full"
                        else min(100, config.validation.monte_carlo_paths)
                    )
                    phase_bar.set_postfix_str("monte carlo")
                    notifier.notify_phase(
                        "Monte Carlo",
                        symbol,
                        f"train + evaluation | {monte_carlo_paths:,} paths each",
                    )
                    training_monte_carlo = moving_block_monte_carlo(
                        training_returns.to_numpy(),
                        paths=monte_carlo_paths,
                        block_size=min(24, len(training_returns)),
                    )
                    evaluation_monte_carlo = moving_block_monte_carlo(
                        evaluation_returns.to_numpy(),
                        paths=monte_carlo_paths,
                        block_size=min(24, len(evaluation_returns)),
                    )
                    phase_bar.update()
                    phase_bar.set_postfix_str("reporting")
                    notifier.notify_phase("Reporting", symbol)
                    stem = symbol.replace("/", "_")
                    training_report = generate_full_report(
                        training_returns,
                        training_monte_carlo,
                        config.outputs_dir / f"{stem}_training_report.html",
                        title=f"{symbol} {args.profile.title()} Training Backtest",
                        symbol=symbol,
                        agent_name=agent_name,
                        benchmark=reporting["training_benchmark"],
                        comparators={"Alpha": reporting["training_deterministic"]},
                    )
                    evaluation_report = generate_full_report(
                        evaluation_returns,
                        evaluation_monte_carlo,
                        config.outputs_dir / f"{stem}_evaluation_report.html",
                        title=f"{symbol} {args.profile.title()} Evaluation Backtest",
                        symbol=symbol,
                        agent_name=agent_name,
                        benchmark=reporting["evaluation_benchmark"],
                        comparators={
                            "Alpha": reporting["evaluation_deterministic"],
                            "Vol B&H": reporting["evaluation_volatility_benchmark"],
                        },
                    )
                    for split, report in (("training", training_report), ("evaluation", evaluation_report)):
                        notifier.notify_html(report["telegram_report"])
                        notifier.send_photo(report["graphics"], f"{symbol} {split} backtest graphics")
                        notifier.send_document(report["html"], f"{symbol} {split} QuantStats report")
                        if report["quantstats_error"]:
                            notifier.notify(f"{symbol} {split} report warning | {report['quantstats_error']}")
                    phase_bar.update()
                print(
                    json.dumps(
                        {
                            "symbol": symbol,
                            "profile": args.profile,
                            "eligible": manifest["eligible"],
                            "training_report": str(training_report["html"]),
                            "evaluation_report": str(evaluation_report["html"]),
                            "training_metrics": training_report["metrics"],
                            "evaluation_metrics": evaluation_report["metrics"],
                            "evaluation_buy_and_hold_metrics": evaluation_report["benchmark_metrics"],
                        },
                        sort_keys=True,
                    )
                )
        elapsed = time.monotonic() - started
        tqdm.write(f"Pipeline complete | {elapsed:.1f}s")
        notifier.stop_updates(f"Pipeline complete | {elapsed:.1f}s")
    except Exception as exc:
        elapsed = time.monotonic() - started
        tqdm.write(f"Pipeline failed | {elapsed:.1f}s")
        notifier.notify_failure(exc, elapsed)
        raise
    finally:
        notifier.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
