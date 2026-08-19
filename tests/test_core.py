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


if __name__ == "__main__":
    unittest.main()
