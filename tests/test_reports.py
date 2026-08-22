from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from telegram_bot.reports import _keep_matplotlib_log, generate_full_report
from validation import moving_block_monte_carlo


class ReportTests(unittest.TestCase):
    def test_quantstats_missing_arial_warning_is_filtered(self) -> None:
        missing_arial = logging.makeLogRecord(
            {"msg": "findfont: Font family 'Arial' not found."}
        )
        other_warning = logging.makeLogRecord({"msg": "different font warning"})

        self.assertFalse(_keep_matplotlib_log(missing_arial))
        self.assertTrue(_keep_matplotlib_log(other_warning))

    def test_singular_quantstats_kde_falls_back_to_local_html(self) -> None:
        index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        returns = pd.Series(np.zeros(len(index)), index=index)
        monte_carlo = moving_block_monte_carlo(
            returns.to_numpy(), paths=20, block_size=4, batch_size=5, seed=3
        )

        def quantstats_html(*args, **kwargs) -> None:
            raise np.linalg.LinAlgError("singular covariance")

        quantstats = SimpleNamespace(reports=SimpleNamespace(html=quantstats_html))
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            sys.modules, {"quantstats": quantstats}
        ):
            report = generate_full_report(
                returns,
                monte_carlo,
                Path(folder) / "smoke.html",
                title="Smoke",
            )

            self.assertIn(
                "return covariance is singular",
                report["html"].read_text(encoding="utf-8"),
            )
            self.assertEqual(
                report["quantstats_error"],
                "QuantStats plots unavailable: return covariance is singular.",
            )

    def test_quantstats_report_compares_model_with_buy_and_hold(self) -> None:
        index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        strategy = pd.Series(np.tile([0.002, -0.001], 24), index=index)
        benchmark = pd.Series(np.tile([0.001, -0.0008], 24), index=index)
        monte_carlo = moving_block_monte_carlo(
            strategy.to_numpy(), paths=20, block_size=4, batch_size=5, seed=3
        )
        captured = {}

        def quantstats_html(returns, *, benchmark, output, **kwargs) -> None:
            captured["returns"] = returns
            captured["benchmark"] = benchmark
            captured["strategy_title"] = kwargs["strategy_title"]
            captured["benchmark_title"] = kwargs["benchmark_title"]
            Path(output).write_text("<html><body></body></html>", encoding="utf-8")

        quantstats = SimpleNamespace(reports=SimpleNamespace(html=quantstats_html))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"quantstats": quantstats}):
            destination = Path(folder) / "evaluation.html"
            destination.with_suffix(".tex").write_text("stale", encoding="utf-8")
            report = generate_full_report(
                strategy,
                monte_carlo,
                destination,
                title="Evaluation",
                symbol="BTC/USD",
                agent_name="PuffeRL-LSTM",
                benchmark=benchmark,
            )
            self.assertIn("PuffeRL-LSTM", report["text_report"])
            self.assertNotIn("RL model", report["text_report"])
            self.assertIn("Buy & Hold", report["text_report"])
            self.assertNotIn("Strategy", report["text_report"])
            self.assertNotIn("Alpha", report["text_report"])
            self.assertIn("<pre>", report["telegram_report"])
            self.assertIn("PuffeRL-LSTM", report["telegram_report"])
            self.assertIn("B&amp;H", report["telegram_report"])
            self.assertIn("Buy &amp; Hold", report["html"].read_text(encoding="utf-8"))
            self.assertNotIn("latex", report)
            self.assertFalse(destination.with_suffix(".tex").exists())
            expected_index = index.tz_convert("UTC").tz_localize(None)
            pd.testing.assert_index_equal(captured["returns"].index, expected_index)
            pd.testing.assert_index_equal(captured["benchmark"].index, expected_index)
            np.testing.assert_array_equal(captured["returns"].to_numpy(), strategy.to_numpy())
            np.testing.assert_array_equal(captured["benchmark"].to_numpy(), benchmark.to_numpy())
            self.assertEqual(captured["strategy_title"], "PuffeRL-LSTM")
            self.assertEqual(captured["benchmark_title"], "Buy & Hold")


if __name__ == "__main__":
    unittest.main()
