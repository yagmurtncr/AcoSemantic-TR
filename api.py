from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.analysis import analyze_audio_file
from src.config import DEFAULT_ACOUSTIC_MODELS, DEFAULT_ASR_MODELS, DEFAULT_SENTIMENT_MODELS

app = FastAPI(title="AcoSemantic-TR API", version="0.1.0")


# -----------------------------------------------------------------------------
# Response models
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = Field(default="AcoSemantic-TR API")


class ModelChoice(BaseModel):
    label: str
    model: str


class AnalyzeResponse(BaseModel):
    transcript: str
    sentiment_label: str
    positivity_score: float
    acoustic_label: str
    stress_score: float
    anomaly_detected: bool
    discordance_score: float
    verdict: str
    acoustic_mode: str
    metadata: dict[str, object]


class ModelGroupResponse(BaseModel):
    asr: ModelChoice
    sentiment: ModelChoice
    acoustic: ModelChoice


# -----------------------------------------------------------------------------
# Model resolution helpers
# -----------------------------------------------------------------------------
def _resolve_model(model_map: dict[str, str], model_name: str) -> str:
    if model_name not in model_map:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    return model_map[model_name]


def _make_model_group(asr_label: str, sentiment_label: str, acoustic_label: str) -> ModelGroupResponse:
    return ModelGroupResponse(
        asr=ModelChoice(label=asr_label, model=_resolve_model(DEFAULT_ASR_MODELS, asr_label)),
        sentiment=ModelChoice(label=sentiment_label, model=_resolve_model(DEFAULT_SENTIMENT_MODELS, sentiment_label)),
        acoustic=ModelChoice(label=acoustic_label, model=_resolve_model(DEFAULT_ACOUSTIC_MODELS, acoustic_label)),
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    asr_model: str = Form("Whisper Small"),
    sentiment_model: str = Form("Savasy Turkish Sentiment"),
    acoustic_model: str = Form("DynAnn Speech Emotion"),
) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = Path(file.filename).suffix or ".wav"
    model_group = _make_model_group(asr_model, sentiment_model, acoustic_model)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"upload{suffix}"
        temp_path.write_bytes(file_bytes)

        result = analyze_audio_file(
            temp_path,
            model_group.asr.model,
            model_group.sentiment.model,
            model_group.acoustic.model,
        )

    return AnalyzeResponse(
        transcript=result.transcript,
        sentiment_label=result.sentiment_label,
        positivity_score=result.positivity_score,
        acoustic_label=result.acoustic_label,
        stress_score=result.stress_score,
        anomaly_detected=result.anomaly_detected,
        discordance_score=result.discordance_score,
        verdict=result.verdict,
        acoustic_mode=result.acoustic_mode,
        metadata=result.metadata,
    )