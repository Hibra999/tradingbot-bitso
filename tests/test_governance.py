from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from rl import file_sha256, internal_purged_validation_tail, load_approved_manifest, promotion_gate


class GovernanceTests(unittest.TestCase):
    def test_internal_validation_tail_is_purged(self) -> None:
        index = pd.date_range("2025-01-01", periods=1_000, freq="h", tz="UTC")
        ends = pd.DatetimeIndex(index[np.minimum(np.arange(1_000) + 24, 999)])
        training, validation = internal_purged_validation_tail(
            np.arange(1_000), index, ends, validation_fraction=0.1, embargo_bars=20
        )
        self.assertLess(index[training].max(), index[validation].min())
        for item in training:
            self.assertLess(ends[item], index[validation].min())

    def test_live_loading_needs_full_gate_operator_selection_and_two_flags(self) -> None:
        metrics = {
            "dsr_p_value": 0.01,
            "max_drawdown": -0.1,
            "ruin_probability_20": 0.01,
            "pbo_probability": 0.0,
            "excess_return_vs_buy_and_hold": 0.05,
            "excess_return_vs_deterministic_alpha": 0.02,
            "deterministic_alpha_return": 0.03,
            "deterministic_alpha_ci95_low": 0.0001,
            "alpha_diagnostic_pass": True,
            "alpha_ic_mean": 0.05,
            "alpha_ic_positive_fraction": 0.8,
            "stress_return": 0.01,
            "bootstrap_mean_ci95_low": 0.0001,
            "paired_alpha_ci95_low": 0.0001,
            "paired_volatility_bh_ci95_low": 0.0001,
            "mcs_90_pass": True,
            "seed_iqm_return_ci95_low": 0.001,
            "seed_stability_pass": True,
            "profitable_fold_fraction": 0.8,
            "max_fold_profit_share": 0.4,
        }
        self.assertFalse(promotion_gate(metrics, profile="smoke")[0])
        self.assertTrue(promotion_gate(metrics, profile="full")[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            artifact_path = Path(folder) / "model.zip"
            feature_path = Path(folder) / "features.pkl"
            alpha_path = Path(folder) / "alpha.pkl"
            artifact_path.write_bytes(b"model")
            feature_path.write_bytes(b"features")
            alpha_path.write_bytes(b"alpha")
            artifact = str(artifact_path)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "profile": "full",
                        "eligible": True,
                        "selected_artifact": artifact,
                        "artifact_paths": [artifact],
                        "eligible_artifacts": [artifact],
                        "artifact_bundle": {
                            "model_path": artifact,
                            "feature_pipeline_path": str(feature_path),
                            "alpha_pipeline_path": str(alpha_path),
                            "action_contract": "target_exposure_long_cash_v1",
                            "sha256": {
                                "model": file_sha256(artifact_path),
                                "feature_pipeline": file_sha256(feature_path),
                                "alpha_pipeline": file_sha256(alpha_path),
                            },
                        },
                    }
                )
            )
            with patch.dict(
                os.environ,
                {
                    "MODEL_APPROVED": "true",
                    "BITSO_LIVE_ENABLED": "true",
                    "APPROVED_MODEL_MANIFEST": str(path),
                },
                clear=False,
            ):
                self.assertEqual(load_approved_manifest(path)["selected_artifact"], artifact)
            with self.assertRaises(PermissionError):
                load_approved_manifest(path)


if __name__ == "__main__":
    unittest.main()
