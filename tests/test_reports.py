from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from telegram_bot.reports import generate_full_report
from validation import moving_block_monte_carlo


class ReportTests(unittest.TestCase):
    def test_quantstats_report_compares_model_with_buy_and_hold(self) -> None:
        index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        strategy = pd.Series(np.tile([0.002, -0.001], 24), index=index)
        benchmark = pd.Series(np.tile([0.001, -0.0008], 24), index=index)
        alpha = pd.Series(np.tile([0.0015, -0.0009], 24), index=index)
        monte_carlo = moving_block_monte_carlo(
            strategy.to_numpy(), paths=20, block_size=4, batch_size=5, seed=3
        )
        captured = {}

        def quantstats_html(returns, *, benchmark, output, **kwargs) -> None:
            captured["returns"] = returns
            captured["benchmark"] = benchmark
            Path(output).write_text("<html><body></body></html>", encoding="utf-8")

        quantstats = SimpleNamespace(reports=SimpleNamespace(html=quantstats_html))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"quantstats": quantstats}):
            report = generate_full_report(
                strategy,
                monte_carlo,
                Path(folder) / "evaluation.html",
                title="Evaluation",
                symbol="BTC/USD",
                benchmark=benchmark,
                comparators={"Alpha": alpha},
            )
            self.assertIn("RL model", report["text_report"])
            self.assertIn("Buy & Hold", report["text_report"])
            self.assertIn("Alpha", report["text_report"])
            self.assertIn("<pre>", report["telegram_report"])
            self.assertIn("B&amp;H", report["telegram_report"])
            self.assertIn("Buy &amp; Hold", report["html"].read_text(encoding="utf-8"))
            latex = report["latex"].read_text(encoding="utf-8")
            self.assertIn(r"\documentclass{article}", latex)
            self.assertIn(r"\resizebox{\textwidth}{!}", latex)
            self.assertIn(r"Buy \& Hold", latex)
            self.assertIn("Alpha", latex)
            pd.testing.assert_series_equal(captured["returns"], strategy.rename("RL model"))
            pd.testing.assert_series_equal(captured["benchmark"], benchmark.rename("Buy & Hold"))


if __name__ == "__main__":
    unittest.main()
