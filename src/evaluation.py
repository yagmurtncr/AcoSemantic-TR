"""Evaluation & threshold calibration for the discordance detector.

The README's top "production-readiness" task is *"recalibrate the thresholds with
real data."* This module makes that data-driven: given labelled
(positivity, stress, is_discordant) samples it grid-searches the decision
thresholds and reports precision / recall / F1 — so the defaults in
``config.py`` can be justified with numbers instead of guessed.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .decision import decide


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "accuracy": round(accuracy, 4)}


@dataclass(slots=True)
class CalibrationResult:
    positive_threshold: float
    stress_threshold: float
    metrics: dict[str, float]


def calibrate_thresholds(
    samples: Sequence[tuple[float, float, bool]],
    positive_grid: Sequence[float] | None = None,
    stress_grid: Sequence[float] | None = None,
) -> CalibrationResult:
    """Grid-search (positive_threshold, stress_threshold) to maximise F1.

    ``samples`` are ``(positivity_score, stress_score, is_discordant_label)``.
    """
    positive_grid = positive_grid or [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30..0.80
    stress_grid = stress_grid or [round(0.20 + 0.05 * i, 2) for i in range(12)]      # 0.20..0.75

    y_true = [bool(lbl) for _, _, lbl in samples]
    best = CalibrationResult(0.5, 0.3, {"f1": -1.0})
    for pt in positive_grid:
        for st in stress_grid:
            y_pred = [decide(pos, stress, positive_threshold=pt, stress_threshold=st)[0]
                      for pos, stress, _ in samples]
            m = binary_metrics(y_true, y_pred)
            if m["f1"] > best.metrics["f1"]:
                best = CalibrationResult(pt, st, m)
    return best
