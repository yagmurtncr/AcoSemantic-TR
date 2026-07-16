from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prosody
from .audio_utils import load_audio
from .decision import decide
from .models import analyze_acoustic_emotion, analyze_sentiment, transcribe_speech


# -----------------------------------------------------------------------------
# Shared result structure
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class AnalysisResult:
    transcript: str
    sentiment_label: str
    positivity_score: float
    acoustic_label: str
    stress_score: float
    anomaly_detected: bool
    discordance_score: float
    verdict: str
    acoustic_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)
    prosodic_stress: float = 0.0
    prosodic_features: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# End-to-end analysis pipeline
# -----------------------------------------------------------------------------
def analyze_audio_file(
    audio_path: str | Path,
    asr_model: str,
    sentiment_model: str,
    acoustic_model: str,
    *,
    semantic_prompt: str | None = None,
    stress_override: float | None = None,
) -> AnalysisResult:
    audio, sample_rate = load_audio(audio_path)
    transcript_result = transcribe_speech(audio, sample_rate, asr_model)
    transcript = transcript_result["text"]
    semantic_text = semantic_prompt or transcript

    sentiment_result = analyze_sentiment(semantic_text, sentiment_model) if semantic_text else {
        "label": "unknown",
        "score": 0.0,
        "positivity": 0.0,
        "raw": None,
    }
    # Model-free prosodic descriptors (pitch, energy, tempo, pauses).
    prosodic = prosody.extract(audio, sample_rate)
    prosodic_stress = prosody.stress_index(audio, sample_rate, prosodic)

    try:
        acoustic_result = analyze_acoustic_emotion(audio, sample_rate, acoustic_model)
    except Exception as exc:  # acoustic model unavailable -> degrade to prosody
        acoustic_result = {
            "label": "prosodic-fallback",
            "score": prosodic_stress,
            "mode": "prosody-only",
            "raw": {"error": str(exc)},
            "model": None,
        }

    if stress_override is not None:
        acoustic_result = {
            **acoustic_result,
            "label": "demo-overridden",
            "score": float(stress_override),
            "mode": f"{acoustic_result['mode']}-override",
        }

    anomaly_detected, discordance_score, verdict = decide(
        sentiment_result["positivity"],
        acoustic_result["score"],
    )

    return AnalysisResult(
        transcript=transcript,
        sentiment_label=sentiment_result["label"],
        positivity_score=float(sentiment_result["positivity"]),
        acoustic_label=acoustic_result["label"],
        stress_score=float(acoustic_result["score"]),
        anomaly_detected=anomaly_detected,
        discordance_score=float(discordance_score),
        verdict=verdict,
        acoustic_mode=str(acoustic_result["mode"]),
        prosodic_stress=float(prosodic_stress),
        prosodic_features=prosodic.as_dict(),
        metadata={
            "sample_rate": sample_rate,
            "transcription_raw": transcript_result["raw"],
            "semantic_prompt": semantic_prompt,
            "sentiment_raw": sentiment_result["raw"],
            "sentiment_model": sentiment_result.get("model"),
            "acoustic_raw": acoustic_result["raw"],
            "acoustic_model": acoustic_result.get("model"),
            "stress_override": stress_override,
            "prosodic_stress": float(prosodic_stress),
            "prosodic_features": prosodic.as_dict(),
        },
    )
