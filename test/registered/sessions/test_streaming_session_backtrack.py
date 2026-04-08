"""
Streaming session backtracking tests: basic backtrack, drop_previous_output,
offset, invalid rid rejection, and memory leak after repeated backtracks.

All tests share a single server (DEFAULT_SMALL_MODEL) with streaming sessions
and chunked prefill enabled.

Usage:
    python -m pytest test_streaming_backtrack.py -xvs
    python -m unittest test_streaming_backtrack.TestStreamingBacktrack
"""

import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-large")


class TestStreamingSessionBacktrack(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-streaming-session",
                "--chunked-prefill-size",
                "512",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_session(self):
        requests.post(self.base_url + "/flush_cache")
        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 4096, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _close_session(self, session_id):
        resp = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(resp.status_code, 200)

    def _generate(self, text, session_params=None, max_new_tokens=16):
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "no_stop_trim": True,
            },
        }
        if session_params is not None:
            payload["session_params"] = session_params
        resp = requests.post(self.base_url + "/generate", json=payload, timeout=120)
        self.assertEqual(resp.status_code, 200, f"Generate failed: {resp.text}")
        return resp.json()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_basic_backtrack(self):
        """Backtrack to an earlier turn and branch; KV should be reused."""
        sid = self._open_session()

        r1 = self._generate("Hello world", session_params={"id": sid})
        rid1 = r1["meta_info"]["id"]

        r2 = self._generate(" How are you?", session_params={"id": sid, "rid": rid1})
        rid2 = r2["meta_info"]["id"]
        self.assertGreater(
            r2["meta_info"]["cached_tokens"], 0, "Turn 2 should reuse KV"
        )

        # Backtrack to turn 1 — should still have cached KV
        r3 = self._generate(
            " What about France?", session_params={"id": sid, "rid": rid1}
        )
        self.assertGreater(
            r3["meta_info"]["cached_tokens"],
            0,
            "Backtrack should reuse KV from turn 1",
        )

        self._close_session(sid)

    def test_drop_previous_output(self):
        """Backtrack with drop_previous_output reduces prompt tokens."""
        sid = self._open_session()

        r1 = self._generate("Hello world", session_params={"id": sid})
        rid1 = r1["meta_info"]["id"]

        self._generate(" Continue.", session_params={"id": sid, "rid": rid1})

        # Backtrack with output kept
        r_with = self._generate(" Summarize.", session_params={"id": sid, "rid": rid1})
        # Backtrack with output dropped
        r_drop = self._generate(
            " Summarize.",
            session_params={
                "id": sid,
                "rid": rid1,
                "drop_previous_output": True,
            },
        )
        self.assertLess(
            r_drop["meta_info"]["prompt_tokens"],
            r_with["meta_info"]["prompt_tokens"],
            "drop_previous_output should produce fewer prompt tokens",
        )

        self._close_session(sid)

    def test_offset(self):
        """Backtrack with offset truncates the base context."""
        sid = self._open_session()

        r1 = self._generate("Hello world", session_params={"id": sid})
        rid1 = r1["meta_info"]["id"]

        r2 = self._generate(" Continue.", session_params={"id": sid, "rid": rid1})

        # Backtrack with a small offset — fewer prompt tokens than full context
        r_off = self._generate(
            " Continue.",
            session_params={"id": sid, "rid": rid1, "offset": 5},
        )
        continue_tokens = (
            r2["meta_info"]["prompt_tokens"]
            - r1["meta_info"]["prompt_tokens"]
            - r1["meta_info"]["completion_tokens"]
        )
        self.assertEqual(
            r_off["meta_info"]["prompt_tokens"],
            5 + continue_tokens,
            "prompt_tokens should be exactly offset + new input tokens",
        )

        self._close_session(sid)

    def test_invalid_rid_after_backtrack(self):
        """After backtrack, the superseded rid should be rejected."""
        sid = self._open_session()

        r1 = self._generate("Hello world", session_params={"id": sid})
        rid1 = r1["meta_info"]["id"]

        r2 = self._generate(" How are you?", session_params={"id": sid, "rid": rid1})
        rid2 = r2["meta_info"]["id"]

        # Backtrack to rid1 — this should invalidate rid2
        self._generate(" What about France?", session_params={"id": sid, "rid": rid1})

        # rid2 should now be rejected
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": " Fail.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "session_params": {"id": sid, "rid": rid2},
            },
            timeout=120,
        )
        self.assertNotEqual(
            resp.status_code,
            200,
            "Request with invalidated rid should be rejected",
        )

        self._close_session(sid)

    def test_no_memory_leak(self):
        """Multiple backtracks should not leak KV memory."""
        sid = self._open_session()

        r1 = self._generate("Hello world", session_params={"id": sid})
        rid1 = r1["meta_info"]["id"]

        # Several forward-then-backtrack cycles
        for i in range(5):
            self._generate(f" Turn {i}.", session_params={"id": sid, "rid": rid1})

        self._close_session(sid)
        requests.post(self.base_url + "/flush_cache")
        time.sleep(3)
        health = requests.get(self.base_url + "/health")
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after repeated backtracks — likely a KV memory leak.",
        )


if __name__ == "__main__":
    unittest.main()
