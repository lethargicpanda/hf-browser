import curses
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hf_browser import testing_ui as ui
from hf_browser.inference import APIError


class Screen:
    def __init__(self, keys=(), edits=()):
        self.keys = iter(keys)
        self.edits = iter(edits)
        self.text = []
        self.timeouts = []

    def timeout(self, value):
        self.timeouts.append(value)

    def getmaxyx(self):
        return (24, 100)

    def erase(self):
        pass

    def addstr(self, y, x, text, attr=0):
        self.text.append(text)

    def refresh(self):
        pass

    def move(self, *args):
        pass

    def getch(self):
        key = next(self.keys)
        return key() if callable(key) else key

    def get_wch(self):
        return next(self.edits)


class UITests(unittest.TestCase):
    def setUp(self):
        self.colors = patch.object(curses, "color_pair", return_value=0)
        self.cursor = patch.object(curses, "curs_set")
        self.colors.start()
        self.cursor.start()
        self.addCleanup(self.colors.stop)
        self.addCleanup(self.cursor.stop)

    def test_multiline_editor_saves_unicode_and_restores_cursor(self):
        screen = Screen(edits=["你", "好", "\n", "x", "\x04"])
        app = SimpleNamespace(scr=screen)
        self.assertEqual(ui.edit(app, "Prompt", "", True), "你好\nx")
        self.assertEqual(screen.timeouts[-1], -1)
        curses.curs_set.assert_called_with(0)

    def test_editor_escape_preserves_original(self):
        app = SimpleNamespace(scr=Screen(edits=["x", "\x1b"]))
        self.assertEqual(ui.edit(app, "Prompt", "original", True), "original")

    def test_workers_keep_output_and_errors_in_their_own_queues(self):
        finished = threading.Event()

        def work(cancel, emit):
            emit("partial")
            finished.set()
            raise APIError("provider failed")

        job = ui.Job(work)
        self.assertTrue(finished.wait(1))
        # Synchronize on the terminal event, without any live network calls.
        events = [job.events.get(timeout=1), job.events.get(timeout=1)]
        for item in events:
            job.events.put(item)
        self.assertEqual(job.drain(), ["partial"])
        self.assertTrue(job.done)
        self.assertEqual(job.error, "provider failed")

    def test_streaming_screen_edits_runs_and_cancels_without_blocking_input(self):
        started = threading.Event()
        cancelled = threading.Event()

        def stream(url, payload, cancel, emit, headers=None):
            self.assertEqual(payload["messages"][-1]["content"], "Hello")
            emit("Hello back")
            started.set()
            cancel.wait(2)
            if cancel.is_set():
                cancelled.set()

        def cancel_after_start():
            self.assertTrue(started.wait(1))
            return 27

        screen = Screen(keys=[ord("p"), 10, cancel_after_start, ord("q")], edits=list("Hello") + ["\x04"])
        app = SimpleNamespace(scr=screen, deployments={})
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}), patch.object(
                ui, "wait_job", return_value=["provider"]), patch.object(ui, "stream_chat", side_effect=stream):
            ui.test_screen(app, {"id": "owner/model", "pipeline_tag": "text-generation"})
        self.assertTrue(cancelled.wait(1))
        self.assertIn(100, screen.timeouts)
        self.assertEqual(screen.timeouts[-1], -1)
        self.assertTrue(any("Cancelled locally" in line for line in screen.text))

    def test_pause_or_build_error_is_explained(self):
        app = SimpleNamespace(scr=Screen())
        with patch.object(ui, "space_info", return_value={"runtime": {"stage": "BUILD_ERROR"}}):
            def run_job(app, title, job, **kwargs):
                while True:
                    kind, data = job.events.get(timeout=1)
                    if kind == "error":
                        raise APIError(data)
            with patch.object(ui, "wait_job", side_effect=run_job), self.assertRaisesRegex(APIError, "BUILD_ERROR"):
                ui.monitor_space(app, "user/space")


if __name__ == "__main__":
    unittest.main()
