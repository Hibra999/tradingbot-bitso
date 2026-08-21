from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant import (
    add_realized_volatility_features,
    align_m1_features_to_decisions,
    fixed_width_fracdiff,
    fracdiff_weights,
    garman_klass_volatility,
    micro_price,
    parkinson_volatility,
    select_adf_d,
    yang_zhang_volatility,
)


class QuantFeatureTests(unittest.TestCase):
    def test_fracdiff_and_volatility_are_causal(self) -> None:
        index = pd.date_range("2025-01-01", periods=120, freq="min", tz="UTC")
        close = pd.Series(np.exp(np.arange(120) * 0.001), index=index)
        diff = fixed_width_fracdiff(close, 1.0)
        np.testing.assert_allclose(diff.iloc[1:], close.diff().iloc[1:])
        np.testing.assert_allclose(fracdiff_weights(1.0), [1.0, -1.0])

        bars = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close / 1.01, "Close": close, "Volume": 1})
        past = add_realized_volatility_features(bars.iloc[:100])
        full = add_realized_volatility_features(bars)
        pd.testing.assert_frame_equal(past, full.iloc[:100])

        expected = np.log(1.01**2) / np.sqrt(4 * np.log(2))
        self.assertAlmostEqual(float(parkinson_volatility(bars, 5).iloc[-1]), expected)
        log_hl = np.log(bars["High"] / bars["Low"])
        log_co = np.log(bars["Close"] / bars["Open"])
        expected_gk = np.sqrt((0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)).iloc[-5:].mean())
        self.assertAlmostEqual(float(garman_klass_volatility(bars, 5).iloc[-1]), expected_gk)
        realized = np.sqrt(np.log(close).diff().pow(2).iloc[-5:].sum())
        self.assertAlmostEqual(float(full["rv_5m"].iloc[-1]), realized)

        decisions = pd.DatetimeIndex([bars.index[59] + pd.Timedelta(seconds=30), bars.index[89]])
        aligned = align_m1_features_to_decisions(full[["rv_5m"]], decisions)
        self.assertEqual(float(aligned.iloc[0, 0]), float(full["rv_5m"].iloc[59]))
        self.assertEqual(float(aligned.iloc[1, 0]), float(full["rv_5m"].iloc[89]))

        varied = bars.iloc[:8].copy()
        varied["Open"] = varied["Close"].shift().fillna(varied["Close"].iloc[0]) * np.linspace(1, 1.002, 8)
        overnight = np.log(varied["Open"] / varied["Close"].shift())
        open_close = np.log(varied["Close"] / varied["Open"])
        rogers_satchell = np.log(varied["High"] / varied["Open"]) * np.log(varied["High"] / varied["Close"]) + np.log(varied["Low"] / varied["Open"]) * np.log(varied["Low"] / varied["Close"])
        k = 0.34 / (1.34 + 4 / 2)
        expected_yz = np.sqrt(overnight.rolling(3).var() + k * open_close.rolling(3).var() + (1 - k) * rogers_satchell.rolling(3).mean())
        self.assertAlmostEqual(float(yang_zhang_volatility(varied, 3).iloc[-1]), float(expected_yz.iloc[-1]))

    def test_micro_price_weights_opposite_quote(self) -> None:
        value = micro_price(
            pd.Series([99.0]), pd.Series([3.0]), pd.Series([101.0]), pd.Series([1.0])
        ).iloc[0]
        self.assertEqual(value, 100.5)

    def test_adf_selection_uses_smallest_passing_d_and_fails_explicitly(self) -> None:
        stationary = pd.Series(np.random.default_rng(7).normal(size=300))
        self.assertEqual(select_adf_d(stationary).d, 0.0)
        with self.assertRaisesRegex(ValueError, "No fractional-difference candidate"):
            select_adf_d(pd.Series(np.random.default_rng(9).normal(size=300).cumsum()), grid=(0.0,))


if __name__ == "__main__":
    unittest.main()
