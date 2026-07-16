"""Model-free prosodic feature extraction (pure NumPy).

The project's premise is that *how* something is said (pitch, energy, tempo)
can contradict *what* is said. Until now the acoustic side relied entirely on a
single audio-classification model; this module adds interpretable, from-scratch
prosodic descriptors so the "acoustic" analysis actually measures pitch / energy
/ tempo — and can run even when the heavy model is unavailable.

Everything here is deterministic NumPy DSP (framing, RMS, zero-crossings,
autocorrelation pitch), so it is fast and fully unit-testable with synthetic
signals — no model downloads required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True)
class ProsodicFeatures:
    pitch_mean_hz: float       # mean fundamental frequency (voiced frames)
    pitch_std_hz: float        # F0 variability -> arousal / stress
    voiced_ratio: float        # fraction of frames that are voiced
    rms_mean: float            # loudness
    rms_std: float             # loudness variability
    zcr_mean: float            # zero-crossing rate proxy for speech rate/noisiness
    pause_ratio: float         # fraction of low-energy (silence) frames

    def as_dict(self) -> dict:
        return asdict(self)


def _frame(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    n = 1 + (len(audio) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return audio[idx]


def _estimate_f0(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    """Autocorrelation pitch estimate for a single frame (0.0 if unvoiced)."""
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < 1e-4:            # silence
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    min_lag = int(sr / fmax)
    max_lag = min(int(sr / fmin), len(corr) - 1)
    if max_lag <= min_lag:
        return 0.0
    segment = corr[min_lag:max_lag]
    lag = int(np.argmax(segment)) + min_lag
    # voiced only if the periodic peak is prominent vs. the zero-lag energy
    if corr[0] <= 0 or corr[lag] / corr[0] < 0.3:
        return 0.0
    return float(sr / lag)


def extract(audio: np.ndarray, sample_rate: int,
            frame_ms: float = 32.0, hop_ms: float = 16.0,
            fmin: float = 70.0, fmax: float = 400.0) -> ProsodicFeatures:
    audio = np.asarray(audio, dtype=np.float64).ravel()
    if audio.size == 0:
        return ProsodicFeatures(0, 0, 0, 0, 0, 0, 1.0)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak                            # normalise amplitude

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    frames = _frame(audio, frame_len, hop)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    silence_gate = 0.1 * (rms.max() if rms.max() > 0 else 1.0)
    voiced_mask = rms > silence_gate

    f0s = np.array([_estimate_f0(f, sample_rate, fmin, fmax) for f in frames])
    voiced_f0 = f0s[(f0s > 0) & voiced_mask]

    return ProsodicFeatures(
        pitch_mean_hz=float(voiced_f0.mean()) if voiced_f0.size else 0.0,
        pitch_std_hz=float(voiced_f0.std()) if voiced_f0.size else 0.0,
        voiced_ratio=float(voiced_mask.mean()),
        rms_mean=float(rms.mean()),
        rms_std=float(rms.std()),
        zcr_mean=float(zcr.mean()),
        pause_ratio=float(1.0 - voiced_mask.mean()),
    )


def _norm(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def stress_index(audio: np.ndarray, sample_rate: int,
                 features: ProsodicFeatures | None = None) -> float:
    """Composite prosodic stress/arousal score in [0, 1].

    High vocal arousal (stress) tends to show up as **higher pitch variability**,
    **higher/energetic loudness**, a **faster rate** (more zero-crossings) and
    **fewer long pauses**. Each cue is normalised against a speech-typical range
    and combined with interpretable weights.
    """
    f = features or extract(audio, sample_rate)
    pitch_var = _norm(f.pitch_std_hz, 5.0, 80.0)       # steady vs. jittery pitch
    loudness = _norm(f.rms_mean, 0.02, 0.35)
    rate = _norm(f.zcr_mean, 0.02, 0.25)
    fluency = 1.0 - f.pause_ratio                       # less silence -> more pressured
    score = 0.40 * pitch_var + 0.25 * loudness + 0.20 * rate + 0.15 * fluency
    return float(np.clip(score, 0.0, 1.0))
