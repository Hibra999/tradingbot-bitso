from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

from telegram_bot.notifier import PipelineNotifier


class PipelineNotifierTests(unittest.TestCase):
    def test_html_message_requests_telegram_html_parsing(self) -> None:
        notifier = PipelineNotifier.__new__(PipelineNotifier)
        notifier._chat_ids = (123,)
        notifier._state_lock = threading.Lock()
        notifier._sent_message_ids = {}
        notifier._post = Mock(return_value={"message_id": 1})

        notifier.notify_html("<b>Report</b>")

        notifier._post.assert_called_once_with(
            "sendMessage",
            json={"chat_id": 123, "text": "<b>Report</b>", "parse_mode": "HTML"},
        )

    def test_commands_enforce_allowlist_report_progress_and_clear_current_run(self) -> None:
        notifier = PipelineNotifier.__new__(PipelineNotifier)
        notifier._chat_ids = (123,)
        notifier._state_lock = threading.Lock()
        notifier._status = "Evaluation"
        notifier._query_status = "Training 5/10"
        notifier._started = time.monotonic()
        notifier._message_ids = {123: 10}
        notifier._sent_message_ids = {123: {10}}
        notifier._post = Mock(return_value={"message_id": 11})
        notifier._publish_status = Mock()

        notifier._handle_command(999, 20, "/progress")
        notifier._handle_command(123, 20, "/progress@quant_bot")

        self.assertEqual(notifier._post.call_count, 1)
        self.assertIn("Training 5/10", notifier._post.call_args.kwargs["json"]["text"])

        notifier._post.reset_mock()
        notifier._handle_command(123, 20, "/clear")

        deleted = {
            call.kwargs["json"]["message_id"] for call in notifier._post.call_args_list
        }
        self.assertEqual(deleted, {10, 11, 20})
        notifier._publish_status.assert_called_once_with((123,))

    def test_status_buttons_route_allowlisted_callbacks(self) -> None:
        notifier = PipelineNotifier.__new__(PipelineNotifier)
        notifier._chat_ids = (123,)
        notifier._state_lock = threading.Lock()
        notifier._status = "Evaluation"
        notifier._query_status = "Training 5/10"
        notifier._started = time.monotonic()
        notifier._message_ids = {}
        notifier._sent_message_ids = {}
        notifier._post = Mock(return_value={"message_id": 12})

        notifier._publish_status((123,))
        buttons = notifier._post.call_args.kwargs["json"]["reply_markup"]["inline_keyboard"]
        self.assertEqual(
            [button["callback_data"] for row in buttons for button in row],
            ["pipeline:progress", "pipeline:status", "pipeline:help", "pipeline:clear"],
        )
        notifier._command_offset = 5
        notifier._post = Mock(
            side_effect=[
                [
                    {
                        "update_id": 5,
                        "callback_query": {
                            "id": "callback",
                            "data": "pipeline:progress",
                            "message": {"message_id": 10, "chat": {"id": 123}},
                        },
                    }
                ],
                True,
                {"message_id": 12},
            ]
        )

        notifier._poll_commands(timeout=0)

        self.assertEqual(notifier._post.call_count, 3)
        self.assertEqual(
            notifier._post.call_args_list[1].kwargs["json"],
            {"callback_query_id": "callback"},
        )
        self.assertIn("Training 5/10", notifier._post.call_args_list[2].kwargs["json"]["text"])

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
