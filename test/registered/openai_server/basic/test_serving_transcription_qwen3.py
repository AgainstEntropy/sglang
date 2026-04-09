"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Qwen3-ASR.

Covers non-streaming and chunk-based streaming transcription.

Usage:
    python3 test_serving_transcription_qwen3.py -v
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

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small", nightly=True)

MODEL = "Qwen/Qwen3-ASR-0.6B"
TEST_AUDIO_EN_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
)
TEST_AUDIO_ZH_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
)
TEST_AUDIO_EN_LOCAL = "/tmp/test_qwen3_asr_en.wav"
TEST_AUDIO_ZH_LOCAL = "/tmp/test_qwen3_asr_zh.wav"


class TestServingTranscriptionQwen3(CustomTestCase):
    """Test Qwen3-ASR transcription via /v1/audio/transcriptions endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                MODEL,
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _transcribe(self, audio_url, local_path, language=None):
        """Send a non-streaming transcription request."""
        audio_bytes = download_audio_bytes(audio_url, local_path)
        data = {"model": MODEL}
        if language:
            data["language"] = language
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data=data,
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _transcribe_stream(self, audio_url, local_path, language=None):
        """Send a chunk-based streaming transcription request."""
        audio_bytes = download_audio_bytes(audio_url, local_path)
        data = {"model": MODEL, "stream": "true"}
        if language:
            data["language"] = language
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data=data,
            timeout=180,
            stream=True,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return parse_sse_stream(response)

    # ---- Non-streaming tests ----

    def test_basic_transcription_english(self):
        """Non-streaming transcription returns valid non-empty text for English."""
        result = self._transcribe(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0, "Transcription should not be empty")
        text = result["text"].lower()
        keywords = ["listening", "solo", "music", "writing"]
        matches = [kw for kw in keywords if kw in text]
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 of {keywords} in transcription, "
            f"found {matches}. Full text: {text}",
        )

    def test_basic_transcription_chinese(self):
        """Non-streaming transcription returns valid non-empty text for Chinese."""
        result = self._transcribe(TEST_AUDIO_ZH_URL, TEST_AUDIO_ZH_LOCAL)
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0, "Transcription should not be empty")
        keywords = ["交易", "停滞"]
        matches = [kw for kw in keywords if kw in result["text"]]
        self.assertGreaterEqual(
            len(matches),
            1,
            f"Expected at least 1 of {keywords} in transcription, "
            f"found {matches}. Full text: {result['text']}",
        )

    # ---- Chunk-based streaming tests ----

    def test_streaming_english(self):
        """Chunk-based streaming should return SSE events with non-empty text."""
        events, full_text = self._transcribe_stream(
            TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL
        )
        self.assertGreater(len(events), 0, "Should receive at least one SSE event")
        self.assertGreater(len(full_text), 0, "Assembled text should not be empty")

        last_event = events[-1]
        self.assertEqual(
            last_event["choices"][0].get("finish_reason"),
            "stop",
            "Last event should have finish_reason='stop'",
        )

        text = full_text.lower()
        keywords = ["listening", "solo", "music", "writing"]
        matches = [kw for kw in keywords if kw in text]
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 of {keywords} in streaming transcription, "
            f"found {matches}. Full text: {text}",
        )

    def test_streaming_chinese(self):
        """Chunk-based streaming for Chinese audio."""
        events, full_text = self._transcribe_stream(
            TEST_AUDIO_ZH_URL, TEST_AUDIO_ZH_LOCAL
        )
        self.assertGreater(len(events), 0, "Should receive at least one SSE event")
        self.assertGreater(len(full_text), 0, "Assembled text should not be empty")
        keywords = ["交易", "停滞"]
        matches = [kw for kw in keywords if kw in full_text]
        self.assertGreaterEqual(
            len(matches),
            1,
            f"Expected at least 1 of {keywords} in streaming transcription, "
            f"found {matches}. Full text: {full_text}",
        )

    def test_streaming_event_format(self):
        """Verify each SSE event has the expected TranscriptionStreamResponse structure."""
        events, _ = self._transcribe_stream(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        for i, event in enumerate(events):
            self.assertIn("id", event, f"Event {i} missing 'id'")
            self.assertEqual(
                event.get("object"),
                "transcription.chunk",
                f"Event {i} has wrong object type",
            )
            self.assertIn("model", event, f"Event {i} missing 'model'")
            self.assertIn("choices", event, f"Event {i} missing 'choices'")
            self.assertEqual(
                len(event["choices"]), 1, f"Event {i} should have exactly 1 choice"
            )
            self.assertIn(
                "delta", event["choices"][0], f"Event {i} choice missing 'delta'"
            )

    def test_streaming_vs_nonstreaming_similarity(self):
        """Streaming and non-streaming should produce similar transcriptions.

        Chunk-based streaming may produce slightly different text at chunk
        boundaries, so we compare word overlap instead of exact equality.
        """
        non_stream = self._transcribe(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        _, stream_text = self._transcribe_stream(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)

        non_stream_text = non_stream["text"].strip()
        stream_text = stream_text.strip()

        self.assertGreater(len(non_stream_text), 0)
        self.assertGreater(len(stream_text), 0)

        ns_words = set(non_stream_text.lower().split())
        st_words = set(stream_text.lower().split())
        if ns_words:
            overlap = len(ns_words & st_words) / len(ns_words)
            self.assertGreater(
                overlap,
                0.8,
                f"Word overlap too low ({overlap:.0%}). "
                f"Non-stream: {non_stream_text[:100]}, "
                f"Stream: {stream_text[:100]}",
            )


if __name__ == "__main__":
    unittest.main()
