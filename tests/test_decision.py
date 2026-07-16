from src.config import POSITIVE_THRESHOLD, STRESS_THRESHOLD
from src.decision import decide


def test_masking_anomaly_when_positive_words_but_high_stress():
    anomaly, score, verdict = decide(0.9, 0.8)
    assert anomaly is True
    assert score == (0.9 + 0.8) / 2.0
    assert "Anomali" in verdict


def test_no_anomaly_when_calm_voice():
    anomaly, score, verdict = decide(0.9, 0.1)   # positive words, calm voice
    assert anomaly is False
    assert score == abs(0.9 - 0.1)
    assert "saptanmadi" in verdict


def test_no_anomaly_when_negative_words():
    anomaly, _, _ = decide(0.2, 0.9)             # negative words, high stress -> consistent
    assert anomaly is False


def test_boundary_is_strict_greater_than():
    # exactly at thresholds must NOT trigger (rule uses strict >)
    anomaly, _, _ = decide(POSITIVE_THRESHOLD, STRESS_THRESHOLD)
    assert anomaly is False


def test_thresholds_are_injectable():
    # with permissive thresholds the same input flips to an anomaly
    anomaly, _, _ = decide(0.4, 0.2, positive_threshold=0.3, stress_threshold=0.1)
    assert anomaly is True
