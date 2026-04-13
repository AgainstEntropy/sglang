"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Whisper.

Covers non-streaming and normal (token-level) streaming transcription.

Usage:
    python3 test_serving_transcription.py -v
"""

import unittest

from sglang.test.asr_utils import ASRTestBase, AudioTestCase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")


class TestServingTranscription(ASRTestBase):
    """Test Whisper transcription via /v1/audio/transcriptions endpoint."""

    model = "openai/whisper-large-v3"
    served_model_name = "whisper"
    streaming_exact_match = True
    audio_cases = [
        AudioTestCase(
            url="https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3",
            keywords=["privilege", "leader", "science", "art"],
            min_keyword_matches=2,
            mime_type="audio/mpeg",
            filename="audio.mp3",
            language="en",
        ),
    ]

    def test_multiple_sequential_requests(self):
        """Test that sequential requests produce consistent results."""
        case = self.audio_cases[0]
        results = []
        for _ in range(3):
            text = self._transcribe(case)
            self.assertGreater(len(text), 0)
            results.append(text)

        for i in range(1, len(results)):
            self.assertEqual(
                results[0],
                results[i],
                f"Transcription {i + 1} differs from first transcription",
            )


if __name__ == "__main__":
    unittest.main()
