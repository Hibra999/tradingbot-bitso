from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from config import AppConfig
from data import BAR_SCHEMA_VERSION, read_parquet_metadata
from data.alpaca import _read_cached_parquet
from data.binance import _validate_hourly
from execution import ExecutionJournal


class CoreTests(unittest.TestCase):
    def test_binance_context_rejects_missing_hours(self) -> None:
        frame = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2025-01-01 01:00", tz="UTC"), pd.Timestamp("2025-01-01 03:00", tz="UTC")]
            ),
        )
        with self.assertRaisesRegex(ValueError, "contain gaps"):
            _validate_hourly(frame, "test")

    def test_legacy_crypto_cache_migrates_open_times_exactly_once(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="min", tz="UTC")
        bars = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
            index=index,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.parquet"
            bars.to_parquet(path, engine="pyarrow")
            migrated = _read_cached_parquet(
                path,
                is_crypto=True,
                bar_delta=pd.Timedelta(minutes=1),
                pandas_rule="1min",
            )
            reloaded = _read_cached_parquet(
                path,
                is_crypto=True,
                bar_delta=pd.Timedelta(minutes=1),
                pandas_rule="1min",
            )
            self.assertIsNotNone(migrated)
            self.assertIsNotNone(reloaded)
            pd.testing.assert_index_equal(migrated.index, index + pd.Timedelta(minutes=1))
            pd.testing.assert_frame_equal(migrated, reloaded)
            metadata = read_parquet_metadata(path)
            self.assertEqual(metadata["bar_schema_version"], BAR_SCHEMA_VERSION)
            self.assertEqual(metadata["bar_timestamp"], "close")

    def test_public_config_and_journal_redact_secrets(self) -> None:
        config = AppConfig()
        self.assertNotIn("secret", str(config.public_dict()).lower())
        self.assertEqual(len(config.config_hash), 64)

        with tempfile.TemporaryDirectory() as folder, ExecutionJournal(Path(folder) / "journal.sqlite3") as journal:
            event_id = journal.append("order", {"api_key": "do-not-store", "price": "10.25"})
            journal.set_state("engine", {"mode": "paper"})
            self.assertEqual(journal.events()[0]["id"], event_id)
            self.assertEqual(journal.events()[0]["payload"]["api_key"], "[redacted]")
            self.assertEqual(journal.get_state("engine"), {"mode": "paper"})

    def test_every_declared_dependency_is_exactly_pinned(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename in ("requirements.in", "requirements.txt"):
            packages = [
                line.strip()
                for line in (root / filename).read_text(encoding="utf-8").splitlines()
                if line and not line[0].isspace() and not line.startswith("#")
            ]
            self.assertTrue(packages)
            self.assertTrue(all("==" in package for package in packages), packages)


if __name__ == "__main__":
    unittest.main()
