from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import CFG
from data_loader import (
    describe_data,
    load_ohlcv,
    make_walk_forward_folds,
    resample_ohlcv,
    split_train_val_test,
)
from features import prepare_feature_frame
from leakage_checks import assert_feature_stability_when_future_appended
from env_bracket import BracketTradingEnv
from baselines import (
    ema_atr_trend_policy,
    evaluate_policy,
    make_trend_hold_policy,
    optimize_trend_hold_policy,
)
from visualize import (
    plot_candles_with_indicators,
    plot_equity_and_drawdown,
    plot_feature_correlation,
    plot_sessions_by_hour,
    plot_trades_on_chart,
    save_html,
    save_html_combined,
)


def _print_cuda_status() -> None:
    """Report whether an NVIDIA GPU is available for PPO training."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed - RL training (train_ppo.py) unavailable.")
        return
    if torch.cuda.is_available():
        print(
            f"CUDA detected: {torch.cuda.get_device_name(0)} "
            f"(torch {torch.__version__}) - RL training will use the GPU."
        )
    else:
        print(
            f"CUDA not detected (torch {torch.__version__}). "
            "PPO training would fall back to CPU."
        )


def _slice_m1_for_decision_window(m1_df, decision_df):
    if decision_df.empty:
        return m1_df.iloc[0:0].copy()
    start = decision_df.index.min()
    end = decision_df.index.max()
    return m1_df.loc[(m1_df.index > start) & (m1_df.index <= end)].copy()


def _make_env(decision_df, m1_df, feature_cols):
    return BracketTradingEnv(
        decision_df,
        m1_df,
        feature_cols,
        sl_atr_multipliers=CFG.sl_atr_multipliers,
        tp_r_multipliers=CFG.tp_r_multipliers,
        initial_equity=CFG.initial_equity,
        risk_fraction=CFG.risk_fraction,
        spread_price=CFG.spread_price,
        slippage_price=CFG.slippage_price,
        commission_per_trade=CFG.commission_per_trade,
        holding_penalty=CFG.holding_penalty,
        reward_mtm_weight=CFG.reward_mtm_weight,
    )


def _scenario_rows(scenario, report_df):
    return [
        {"scenario": scenario, "metric": metric, "value": value}
        for metric, value in report_df["value"].items()
    ]


def _walk_forward_baseline(feat, m1, feature_cols, policy_fn, label):
    """Evaluate a fixed (untrained) baseline policy on every walk-forward
    validation window and return a per-fold metrics frame.  Baselines have no
    fitted parameters, so this measures regime-to-regime *stability* across the
    rolling out-of-sample windows - the same sealed test is excluded throughout.
    """
    folds, _test = make_walk_forward_folds(
        feat,
        n_folds=CFG.n_walk_forward_folds,
        test_frac=CFG.test_frac,
        embargo_bars=CFG.split_embargo_bars,
        anchored=CFG.walk_forward_anchored,
    )
    rows = []
    with tqdm(total=len(folds), desc=f"Walk-forward {label}", unit="fold",
              leave=False, ncols=100) as pbar:
        for k, (_train_df, val_df) in enumerate(folds, start=1):
            val_m1 = _slice_m1_for_decision_window(m1, val_df)
            _eq, _trades, report = evaluate_policy(
                _make_env(val_df, val_m1, feature_cols),
                policy_fn,
                initial_equity=CFG.initial_equity,
                periods_per_year=CFG.periods_per_year,
            )
            metrics = report["value"].to_dict()
            rows.append({
                "policy": label,
                "fold": k,
                "val_start": val_df.index.min().date(),
                "val_end": val_df.index.max().date(),
                "val_return_pct": metrics.get("total_return_pct"),
                "val_profit_factor": metrics.get("profit_factor"),
                "val_win_rate_pct": metrics.get("win_rate_pct"),
                "val_max_dd_pct": metrics.get("max_drawdown_pct"),
                "val_n_trades": metrics.get("n_trades"),
            })
            pbar.update(1)
    return pd.DataFrame(rows)


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    _print_cuda_status()

    total_steps = 8
    pbar = tqdm(total=total_steps, desc="Pipeline", unit="step", ncols=100)
    pbar.write("Pipeline started - data source: %s (%s)" % (
        CFG.data_source, CFG.alpaca_symbol if CFG.data_source == "alpaca" else CFG.csv_path))

    # 1/8 Load data
    pbar.set_description("1/8 Load data")
    print("Loading M1 data...")
    try:
        m1 = load_ohlcv(CFG)
    except Exception as exc:
        pbar.close()
        raise
    print(describe_data(m1))
    pbar.update(1)

    # 2/8 Resample + features
    pbar.set_description("2/8 Resample + features")
    print(f"Resampling to decision timeframe: {CFG.decision_timeframe}  ({CFG.pandas_tf})")
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    print("Building causal stationary features...")
    feat, feature_cols = prepare_feature_frame(
        decision,
        warmup_bars=CFG.warmup_bars,
        atr_period=CFG.atr_period,
        rsi_period=CFG.rsi_period,
    )
    print(f"Decision bars after warmup: {len(feat):,}")
    print(f"Feature count: {len(feature_cols)}")
    pbar.update(1)

    # 3/8 Split + leakage check
    pbar.set_description("3/8 Split + leakage check")
    train_feat, val_feat, test_feat = split_train_val_test(
        feat,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        embargo_bars=CFG.split_embargo_bars,
    )
    print(
        "Temporal split "
        f"train={len(train_feat):,}, val={len(val_feat):,}, test={len(test_feat):,} "
        f"(embargo={CFG.split_embargo_bars} bars)"
    )

    # Lightweight causality check on the first part of the dataframe.
    cut = min(len(decision) - 50, max(CFG.warmup_bars + 200, 300))
    if cut > CFG.warmup_bars + 50:
        assert_feature_stability_when_future_appended(
            decision.iloc[:cut + 50], feature_cols, cut_index=cut,
            atr_period=CFG.atr_period, rsi_period=CFG.rsi_period,
        )
        print("Leakage check passed: appending future bars did not change past features.")
    pbar.update(1)

    # 4/8 Baseline search
    pbar.set_description("4/8 Baseline search")
    print("Searching for a more stable trend baseline on the train/validation splits...")
    train_m1 = _slice_m1_for_decision_window(m1, train_feat)
    val_m1 = _slice_m1_for_decision_window(m1, val_feat)
    best_params, search_df = optimize_trend_hold_policy(
        lambda: _make_env(train_feat, train_m1, feature_cols),
        threshold_grid=CFG.baseline_threshold_grid,
        sl_idx_grid=CFG.baseline_sl_idx_grid,
        tp_idx_grid=CFG.baseline_tp_idx_grid,
        initial_equity=CFG.initial_equity,
        val_env_factory=lambda: _make_env(val_feat, val_m1, feature_cols),
        periods_per_year=CFG.periods_per_year,
    )
    print(f"Selected params: {best_params}")
    search_df.to_csv(out_dir / "policy_search.csv", index=False)
    pbar.update(1)

    print("Test split remains sealed during the default pipeline run.")
    print("Use final_holdout_eval.py after the model/policy is frozen.")

    # 5/8 Evaluate policies
    pbar.set_description("5/8 Evaluate policies")
    selected_policy = make_trend_hold_policy(best_params)
    baseline_train_eq, baseline_train_trades, baseline_train_report = evaluate_policy(
        _make_env(train_feat, train_m1, feature_cols),
        ema_atr_trend_policy,
        initial_equity=CFG.initial_equity,
        periods_per_year=CFG.periods_per_year,
    )
    baseline_val_eq, baseline_val_trades, baseline_val_report = evaluate_policy(
        _make_env(val_feat, val_m1, feature_cols),
        ema_atr_trend_policy,
        initial_equity=CFG.initial_equity,
        periods_per_year=CFG.periods_per_year,
    )
    selected_train_eq, selected_train_trades, selected_train_report = evaluate_policy(
        _make_env(train_feat, train_m1, feature_cols),
        selected_policy,
        initial_equity=CFG.initial_equity,
        periods_per_year=CFG.periods_per_year,
    )
    selected_val_eq, selected_val_trades, selected_val_report = evaluate_policy(
        _make_env(val_feat, val_m1, feature_cols),
        selected_policy,
        initial_equity=CFG.initial_equity,
        periods_per_year=CFG.periods_per_year,
    )
    pbar.update(1)

    # 6/8 Save artifacts (CSV)
    pbar.set_description("6/8 Save artifacts")
    selected_params = asdict(best_params)
    selected_params["sl_atr_multiplier"] = CFG.sl_atr_multipliers[best_params.sl_idx]
    selected_params["tp_r_multiplier"] = CFG.tp_r_multipliers[best_params.tp_idx]
    selected_params_df = pd.DataFrame.from_dict(selected_params, orient="index", columns=["value"])
    selected_params_df.to_csv(out_dir / "selected_policy.csv")

    report_rows = []
    report_rows.extend(_scenario_rows("baseline_train", baseline_train_report))
    report_rows.extend(_scenario_rows("baseline_val", baseline_val_report))
    report_rows.extend(_scenario_rows("selected_train", selected_train_report))
    report_rows.extend(_scenario_rows("selected_val", selected_val_report))
    performance_df = pd.DataFrame(report_rows)

    print("\nPerformance report")
    print(performance_df.pivot(index="metric", columns="scenario", values="value"))
    performance_df.to_csv(out_dir / "performance_report.csv", index=False)
    selected_val_trades.to_csv(out_dir / "selected_val_trade_log.csv", index=False)
    selected_val_eq.to_csv(out_dir / "selected_val_equity_curve.csv")
    pbar.update(1)

    # 7/8 Walk-forward baseline robustness (validation windows only; test sealed)
    pbar.set_description("7/8 Walk-forward robustness")
    print(
        f"\nWalk-forward baseline robustness "
        f"({CFG.n_walk_forward_folds} folds, "
        f"{'anchored' if CFG.walk_forward_anchored else 'rolling'} train, test sealed)..."
    )
    wf_df = pd.concat(
        [
            _walk_forward_baseline(feat, m1, feature_cols, ema_atr_trend_policy, "ema_atr_trend"),
            _walk_forward_baseline(feat, m1, feature_cols, selected_policy, "selected_trend_hold"),
        ],
        ignore_index=True,
    )
    wf_df.to_csv(out_dir / "walk_forward_baseline.csv", index=False)
    print(wf_df.to_string(index=False))
    for label, grp in wf_df.groupby("policy"):
        pos = int((grp["val_return_pct"] > 0).sum())
        print(
            f"  {label}: {pos}/{len(grp)} folds positive, "
            f"mean val_return={grp['val_return_pct'].mean():+.1f}%, "
            f"mean PF={grp['val_profit_factor'].mean():.2f}"
        )
    pbar.update(1)

    # 8/8 Build plots + single combined HTML report
    pbar.set_description("8/8 Build report")
    print("Saving Plotly visuals...")
    pretest_feat = pd.concat([train_feat, val_feat]).sort_index()
    pretest_decision = decision.loc[:val_feat.index.max()]

    figures = [
        ("1. Validation candles + causal indicators",
         plot_candles_with_indicators(val_feat, n=CFG.plot_bars, title="Validation candles with causal indicators")),
        ("2. Price/volume behaviour by hour (train+val only)",
         plot_sessions_by_hour(pretest_decision, title="Price/volume behaviour by hour (train+val only)")),
        ("3. Feature correlation heatmap",
         plot_feature_correlation(pretest_feat, feature_cols)),
        ("4. Selected trend-hold validation trades with TP/SL brackets",
         plot_trades_on_chart(val_feat, selected_val_trades, n=CFG.plot_bars,
                              title="Selected trend-hold validation trades with TP/SL brackets")),
        ("5. Selected trend-hold validation equity & drawdown",
         plot_equity_and_drawdown(selected_val_eq, title="Selected trend-hold validation equity & drawdown")),
    ]
    chart_paths = [
        out_dir / "01_val_candles_indicators.html",
        out_dir / "02_pretest_sessions_by_hour.html",
        out_dir / "03_pretest_feature_correlation.html",
        out_dir / "04_val_trades_on_chart.html",
        out_dir / "05_val_equity_drawdown.html",
    ]
    for fig, path in tqdm(zip([f for _, f in figures], chart_paths),
                           desc="Writing chart files", total=len(figures), unit="chart",
                           leave=False, ncols=100):
        save_html(fig, path)

    combined_path = out_dir / "00_combined_report.html"
    save_html_combined(figures, combined_path,
                       title="RL Trading - Pipeline Report (GLD / Alpaca)")
    pbar.update(1)
    pbar.close()

    print(f"Done. Report and charts in: {out_dir.resolve()}")
    print(f"Combined (single file): {combined_path.resolve()}")


if __name__ == "__main__":
    main()
