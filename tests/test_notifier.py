from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from telegram_bot.notifier import PipelineNotifier


class PipelineNotifierTests(unittest.TestCase):
    def test_html_message_requests_telegram_html_parsing(self) -> None:
        notifier = PipelineNotifier.__new__(PipelineNotifier)
        notifier._chat_ids = (123,)
        notifier._post = Mock()

        notifier.notify_html("<b>Report</b>")

        notifier._post.assert_called_once_with(
            "sendMessage",
            json={"chat_id": 123, "text": "<b>Report</b>", "parse_mode": "HTML"},
        )

    def test_failure_alert_includes_error_without_environment_secret(self) -> None:
        notifier = PipelineNotifier.__new__(PipelineNotifier)
        notifier.stop_updates = Mock()
        notifier.notify_html = Mock()

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "do-not-send"}):
            notifier.notify_failure(ValueError("bad token do-not-send"), 12.34)

        status = notifier.stop_updates.call_args.args[0]
        alert = notifier.notify_html.call_args.args[0]
        self.assertIn("ValueError: bad token [redacted]", status)
        self.assertIn("ValueError: bad token [redacted]", alert)
        self.assertNotIn("do-not-send", status + alert)


if __name__ == "__main__":
    unittest.main()
