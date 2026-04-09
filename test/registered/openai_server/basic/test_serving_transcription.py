"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Whisper.

Covers non-streaming and normal (token-level) streaming transcription.

Usage:
    python3 test_serving_transcription.py -v
"""

import io
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    download_audio_bytes,
    parse_sse_stream,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")

WHISPER_MODEL = "openai/whisper-large-v3"
AUDIO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3"


class TestServingTranscription(CustomTestCase):
    """Test Whisper transcription via /v1/audio/transcriptions endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.model = WHISPER_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                "whisper",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _transcribe(self, language="en"):
        """Send a non-streaming transcription request and return the JSON response."""
        audio_bytes = download_audio_bytes(AUDIO_URL)
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data={
                "model": "whisper",
                "language": language,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _transcribe_stream(self, language="en"):
        """Send a streaming transcription request and return parsed SSE events."""
        audio_bytes = download_audio_bytes(AUDIO_URL)
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data={
                "model": "whisper",
                "language": language,
                "stream": "true",
            },
            timeout=120,
            stream=True,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return parse_sse_stream(response)

    # ---- Non-streaming tests ----

    def test_basic_transcription(self):
        """Test that transcription returns a valid non-empty response."""
        result = self._transcribe()
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0, "Transcription should not be empty")

    def test_transcription_content_quality(self):
        """Test that transcription captures key content from the audio."""
        result = self._transcribe()
        text = result["text"].lower()
        keywords = ["privilege", "leader", "science", "art"]
        matches = [kw for kw in keywords if kw in text]
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 of {keywords} in transcription, "
            f"found {matches}. Full text: {text}",
        )

    def test_multiple_sequential_requests(self):
        """Test that sequential requests produce consistent results."""
        results = []
        for _ in range(3):
            result = self._transcribe()
            self.assertIn("text", result)
            self.assertTrue(len(result["text"]) > 0)
            results.append(result["text"])

        for i in range(1, len(results)):
            self.assertEqual(
                results[0],
                results[i],
                f"Transcription {i + 1} differs from first transcription",
            )

    # ---- Streaming tests ----

    def test_streaming_returns_events(self):
        """Streaming transcription should return SSE events with non-empty text."""
        events, full_text = self._transcribe_stream()
        self.assertGreater(len(events), 0, "Should receive at least one SSE event")
        self.assertGreater(len(full_text), 0, "Assembled text should not be empty")

        last_event = events[-1]
        self.assertEqual(
            last_event["choices"][0].get("finish_reason"),
            "stop",
            "Last event should have finish_reason='stop'",
        )

    def test_streaming_event_format(self):
        """Verify each SSE event has the expected structure."""
        events, _ = self._transcribe_stream()
        for i, event in enumerate(events):
            self.assertIn("id", event, f"Event {i} missing 'id'")
            self.assertIn("choices", event, f"Event {i} missing 'choices'")
            self.assertEqual(
                len(event["choices"]), 1, f"Event {i} should have exactly 1 choice"
            )
            self.assertIn(
                "delta", event["choices"][0], f"Event {i} choice missing 'delta'"
            )

    def test_streaming_vs_nonstreaming_consistency(self):
        """Streaming and non-streaming should produce the same transcription."""
        non_stream = self._transcribe()
        _, stream_text = self._transcribe_stream()

        non_stream_text = non_stream["text"].strip()
        stream_text = stream_text.strip()

        self.assertEqual(
            non_stream_text,
            stream_text,
            f"Streaming text differs from non-streaming.\n"
            f"  non-stream: {non_stream_text[:200]}\n"
            f"  stream:     {stream_text[:200]}",
        )


if __name__ == "__main__":
    unittest.main()
