from __future__ import annotations

import unittest
from unittest.mock import Mock

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


if __name__ == "__main__":
    unittest.main()
