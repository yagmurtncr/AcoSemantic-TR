from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import numpy as np
import torch
from transformers import pipeline

from .audio_utils import compute_prosody_heuristics
from .config import DEFAULT_SENTIMENT_MODELS


def _device_index() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=16)
def _load_pipeline(task: str, model_name: str):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    pipeline_kwargs: dict[str, Any] = {"task": task, "model": model_name, "device": _device_index()}
    if token:
        pipeline_kwargs["token"] = token

    try:
        return pipeline(**pipeline_kwargs)
    except TypeError:
        if token:
            pipeline_kwargs.pop("token", None)
            pipeline_kwargs["use_auth_token"] = token
            return pipeline(**pipeline_kwargs)
        raise


def _load_pipeline_with_fallback(task: str, model_names: list[str]):
    last_error: Exception | None = None
    for model_name in model_names:
        try:
            return _load_pipeline(task, model_name), model_name
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No model names provided for {task}")


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def _is_positive_label(label: str) -> bool:
    normalized = _normalize_label(label)
    return any(token in normalized for token in ("positive", "pos", "joy", "happy", "1"))


def _is_negative_label(label: str) -> bool:
    normalized = _normalize_label(label)
    return any(token in normalized for token in ("negative", "neg", "sad", "anger", "angry", "fear", "0"))


def transcribe_speech(audio: np.ndarray, sample_rate: int, model_name: str) -> dict[str, Any]:
    asr = _load_pipeline("automatic-speech-recognition", model_name)
    payload = {"array": audio, "sampling_rate": sample_rate}

    kwargs: dict[str, Any] = {}
    if "whisper" in model_name.lower():
        kwargs["generate_kwargs"] = {"task": "transcribe", "language": "turkish"}

    result = asr(payload, **kwargs)
    text = result["text"] if isinstance(result, dict) and "text" in result else str(result)
    return {"text": text.strip(), "raw": result}


def analyze_sentiment(text: str, model_name: str) -> dict[str, Any]:
    candidate_models = [model_name]
    for fallback_model in DEFAULT_SENTIMENT_MODELS.values():
        if fallback_model not in candidate_models:
            candidate_models.append(fallback_model)

    try:
        sentiment, loaded_model_name = _load_pipeline_with_fallback("text-classification", candidate_models)
        result = sentiment(text, truncation=True)
        top = result[0] if isinstance(result, list) else result
        label = str(top.get("label", "unknown"))
        score = float(top.get("score", 0.0))

        if _is_positive_label(label):
            positivity = score
        elif _is_negative_label(label):
            positivity = max(0.0, 1.0 - score)
        else:
            positivity = score if score >= 0.5 else 1.0 - score

        return {
            "label": label,
            "score": score,
            "positivity": min(max(positivity, 0.0), 1.0),
            "model": loaded_model_name,
            "raw": result,
        }
    except Exception:
        return _heuristic_sentiment(text)


_STRESS_LABEL_KEYWORDS = ("angry", "anger", "fear", "frustrat", "stress", "tense", "disgust")
_RELAXED_LABEL_KEYWORDS = ("neutral", "calm", "happy", "joy", "relax", "peace", "sad")
_POSITIVE_TEXT_HINTS = (
    "happy",
    "good",
    "great",
    "love",
    "calm",
    "peace",
    "joy",
    "thank",
    "sevin",
    "mutlu",
    "iyi",
    "harika",
)
_NEGATIVE_TEXT_HINTS = (
    "sad",
    "bad",
    "angry",
    "fear",
    "stress",
    "tense",
    "worried",
    "kotu",
    "kötü",
    "uzgun",
    "üzgün",
    "kork",
)


def _heuristic_sentiment(text: str) -> dict[str, Any]:
    normalized = " ".join(text.lower().replace("!", " ! ").replace("?", " ? ").split())
    words = normalized.split()
    positive_hits = sum(1 for hint in _POSITIVE_TEXT_HINTS if hint in normalized)
    negative_hits = sum(1 for hint in _NEGATIVE_TEXT_HINTS if hint in normalized)

    # Short utterances need a slightly stronger prior because there are fewer lexical cues.
    short_text_bonus = 0.0
    if len(words) <= 3:
        short_text_bonus = 0.08

    punctuation_bonus = 0.0
    if "!" in words and positive_hits > 0:
        punctuation_bonus += 0.06
    if "?" in words and negative_hits > 0:
        punctuation_bonus += 0.04

    positivity = 0.5 + 0.18 * positive_hits - 0.18 * negative_hits + short_text_bonus + punctuation_bonus
    if len(words) == 1:
        # Single-word utterances like "Güneş" or "Simit" should not collapse to neutral.
        if positive_hits > negative_hits:
            positivity += 0.10
        elif negative_hits > positive_hits:
            positivity -= 0.10
    positivity = min(max(positivity, 0.0), 1.0)
    label = "POSITIVE" if positivity >= 0.5 else "NEGATIVE"
    return {
        "label": label,
        "score": positivity,
        "positivity": positivity,
        "model": "heuristic-fallback",
        "raw": {
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
        },
    }


def analyze_acoustic_emotion(audio: np.ndarray, sample_rate: int, model_name: str) -> dict[str, Any]:
    try:
        # Prefer a model-backed classifier when available.
        classifier = _load_pipeline("audio-classification", model_name)
        predictions = classifier({"array": audio, "sampling_rate": sample_rate}, top_k=5)
        if not isinstance(predictions, list):
            predictions = [predictions]

        label_scores = {str(item.get("label", "unknown")).lower(): float(item.get("score", 0.0)) for item in predictions}
        stress_score = 0.0
        relaxed_score = 0.0
        for label, score in label_scores.items():
            if any(keyword in label for keyword in _STRESS_LABEL_KEYWORDS):
                stress_score += score
            elif any(keyword in label for keyword in _RELAXED_LABEL_KEYWORDS):
                relaxed_score = max(relaxed_score, score)

        if stress_score <= 0.0 and relaxed_score > 0.0:
            stress_score = 1.0 - relaxed_score
        elif stress_score <= 0.0:
            stress_score = float(max(score for score in label_scores.values())) if label_scores else 0.0

        stress_score = min(max(stress_score, 0.0), 1.0)
        top_label = max(label_scores, key=label_scores.get) if label_scores else "unknown"
        return {
            "label": top_label,
            "score": stress_score,
            "raw": predictions,
            "mode": "classifier",
            "model": model_name,
        }
    except Exception as exc:
        # Log the classifier failure to help debugging and fall back to prosody heuristics.
        try:
            import logging

            logging.getLogger(__name__).warning("Acoustic classifier failed (%s): %s", model_name, exc)
        except Exception:
            pass
        heuristics = compute_prosody_heuristics(audio, sample_rate)
        return {
            "label": "heuristic-fallback",
            "score": heuristics["stress_score"],
            "raw": heuristics,
            "mode": "heuristic",
            "model": f"{model_name}-heuristic-fallback",
        }
