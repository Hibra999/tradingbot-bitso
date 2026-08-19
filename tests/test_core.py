from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from execution import ExecutionJournal


class CoreTests(unittest.TestCase):
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
