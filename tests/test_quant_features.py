from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant import (
    add_realized_volatility_features,
    fixed_width_fracdiff,
    fracdiff_weights,
    micro_price,
    parkinson_volatility,
    select_adf_d,
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
