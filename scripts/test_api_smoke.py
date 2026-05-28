"""Smoke test the API endpoints without loading real model weights."""
from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api  # noqa: E402


class _FakeResult:
    transcript = "Merhaba dunya"
    sentiment_label = "POSITIVE"
    positivity_score = 0.91
    acoustic_label = "calm"
    stress_score = 0.12
    anomaly_detected = False
    discordance_score = 0.79
    verdict = "Belirgin bir duygu celiskisi saptanmadi."
    acoustic_mode = "classifier"
    metadata = {"smoke_test": True}


def _fake_analyze_audio_file(*args, **kwargs):
    return _FakeResult()


def main() -> int:
    api.analyze_audio_file = _fake_analyze_audio_file  # type: ignore[assignment]
    client = TestClient(api.app)

    health_response = client.get("/health")
    print("/health ->", health_response.status_code, health_response.json())

    fake_audio = io.BytesIO(b"RIFF0000WAVEfmt ")
    analyze_response = client.post(
        "/analyze",
        data={
            "asr_model": "Whisper Small",
            "sentiment_model": "Savasy Turkish Sentiment",
            "acoustic_model": "DynAnn Speech Emotion",
        },
        files={"file": ("sample.wav", fake_audio, "audio/wav")},
    )
    print("/analyze ->", analyze_response.status_code)
    print(analyze_response.json())

    if health_response.status_code != 200 or analyze_response.status_code != 200:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())