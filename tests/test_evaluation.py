from src.evaluation import binary_metrics, calibrate_thresholds


def test_binary_metrics_perfect():
    m = binary_metrics([True, False, True, False], [True, False, True, False])
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_binary_metrics_mixed():
    # 1 tp, 1 fp, 1 fn, 1 tn
    m = binary_metrics([True, True, False, False], [True, False, True, False])
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_calibration_finds_separating_thresholds():
    # discordant samples: high positivity AND high stress -> label True
    # everything else -> False. A good threshold pair should separate them.
    samples = [
        (0.9, 0.8, True), (0.8, 0.7, True), (0.95, 0.6, True),   # masking stress
        (0.9, 0.1, False), (0.2, 0.9, False), (0.3, 0.2, False), # consistent
    ]
    result = calibrate_thresholds(samples)
    assert result.metrics["f1"] >= 0.99          # perfectly separable
    assert 0.3 <= result.positive_threshold <= 0.8
    assert 0.2 <= result.stress_threshold <= 0.75
