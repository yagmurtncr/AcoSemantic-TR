from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import numpy as np
import torch
from transformers import pipeline


def _device_index() -> int:
    return 0 if torch.cuda.is_available() else -1


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            return st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_HUB_TOKEN")
    except Exception:
        pass

    return None


@lru_cache(maxsize=16)
def _load_pipeline(task: str, model_name: str):
    token = _hf_token()
    pipeline_kwargs: dict[str, Any] = {"task": task, "model": model_name, "device": _device_index()}
    if token:
        pipeline_kwargs["token"] = token

    return pipeline(**pipeline_kwargs)


def transcribe_speech(audio: np.ndarray, sample_rate: int, model_name: str) -> dict[str, Any]:
    asr = _load_pipeline("automatic-speech-recognition", model_name)
    payload = {"array": audio, "sampling_rate": sample_rate}

    kwargs: dict[str, Any] = {}
    if "whisper" in model_name.lower():
        kwargs["generate_kwargs"] = {"task": "transcribe", "language": "turkish"}

    result = asr(payload, **kwargs)
    text = result["text"] if isinstance(result, dict) and "text" in result else str(result)
    return {"text": text.strip(), "raw": result}


def _label_is_positive(label: str) -> bool:
    normalized = label.strip().lower().replace("_", " ")
    return any(token in normalized for token in ("positive", "pos", "joy", "happy", "1"))


def _label_is_negative(label: str) -> bool:
    normalized = label.strip().lower().replace("_", " ")
    return any(token in normalized for token in ("negative", "neg", "sad", "anger", "angry", "fear", "0"))


def analyze_sentiment(text: str, model_name: str) -> dict[str, Any]:
    sentiment = _load_pipeline("text-classification", model_name)
    result = sentiment(text, truncation=True)
    top = result[0] if isinstance(result, list) else result
    label = str(top.get("label", "unknown"))
    score = float(top.get("score", 0.0))

    if _label_is_positive(label):
        positivity = score
    elif _label_is_negative(label):
        positivity = max(0.0, 1.0 - score)
    else:
        positivity = score if score >= 0.5 else 1.0 - score

    return {
        "label": label,
        "score": score,
        "positivity": min(max(positivity, 0.0), 1.0),
        "model": model_name,
        "raw": result,
    }


def analyze_acoustic_emotion(audio: np.ndarray, sample_rate: int, model_name: str) -> dict[str, Any]:
    classifier = _load_pipeline("audio-classification", model_name)
    predictions = classifier({"array": audio, "sampling_rate": sample_rate}, top_k=5)
    if not isinstance(predictions, list):
        predictions = [predictions]

    label_scores = {str(item.get("label", "unknown")).lower(): float(item.get("score", 0.0)) for item in predictions}
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
