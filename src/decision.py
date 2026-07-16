"""Emotion-discordance decision logic.

Isolated from audio I/O and the ML models so it is pure, fast and unit-testable.
The rule is unchanged from the original pipeline: when the *words* are positive
but the *voice* is stressed, the speaker may be masking stress → flag an anomaly.

Thresholds default to the project config but are injectable for testing/tuning.
"""
from __future__ import annotations

from .config import POSITIVE_THRESHOLD, STRESS_THRESHOLD

_MASKING_VERDICT = "Anomali Tespit Edildi: Kullanici kelimeleriyle stresini gizliyor!"
_NO_ANOMALY_VERDICT = "Belirgin bir duygu celiskisi saptanmadi."


def decide(
    positivity_score: float,
    stress_score: float,
    *,
    positive_threshold: float = POSITIVE_THRESHOLD,
    stress_threshold: float = STRESS_THRESHOLD,
) -> tuple[bool, float, str]:
    """Return (anomaly_detected, discordance_score, verdict).

    * Positive text **and** high vocal stress → masking anomaly; the discordance
      score is the mean of the two signals (both high = strongly discordant).
    * Otherwise no anomaly; the score is the gap between the two signals.
    """
    if positivity_score > positive_threshold and stress_score > stress_threshold:
        discordance_score = (positivity_score + stress_score) / 2.0
        return True, discordance_score, _MASKING_VERDICT

    discordance_score = abs(positivity_score - stress_score)
    return False, discordance_score, _NO_ANOMALY_VERDICT
