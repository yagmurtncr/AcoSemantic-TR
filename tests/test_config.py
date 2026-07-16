from src import config


def test_thresholds_in_unit_range():
    assert 0.0 <= config.POSITIVE_THRESHOLD <= 1.0
    assert 0.0 <= config.STRESS_THRESHOLD <= 1.0


def test_sample_rate_and_upload_limits():
    assert config.TARGET_SAMPLE_RATE == 16_000
    assert config.MAX_UPLOAD_MB > 0


def test_model_registries_non_empty_and_stringy():
    for registry in (config.DEFAULT_ASR_MODELS,
                     config.DEFAULT_SENTIMENT_MODELS,
                     config.DEFAULT_ACOUSTIC_MODELS):
        assert registry, "model registry must not be empty"
        assert all(isinstance(k, str) and "/" in v for k, v in registry.items())
