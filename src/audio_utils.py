from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .config import TARGET_SAMPLE_RATE


def load_audio(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sample_rate


def save_uploaded_audio(uploaded_bytes: bytes, destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(uploaded_bytes)
    return destination_path


def build_mel_spectrogram_figure(audio: np.ndarray, sample_rate: int) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12, 4))
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8_000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    image = librosa.display.specshow(mel_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=axis, cmap="magma")
    axis.set_title("Mel-Spektrogram")
    axis.set_xlabel("Zaman")
    axis.set_ylabel("Mel")
    figure.colorbar(image, ax=axis, format="%+2.0f dB")
    figure.tight_layout()
    return figure
