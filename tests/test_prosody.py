import numpy as np

from src import prosody


def _sine(freq, sr=16000, dur=1.0, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pitch_recovers_known_sine_frequency():
    sr = 16000
    feats = prosody.extract(_sine(150, sr=sr), sr)
    assert abs(feats.pitch_mean_hz - 150) < 15   # autocorrelation pitch within ~10%


def test_stress_index_in_unit_range():
    s = prosody.stress_index(_sine(150), 16000)
    assert 0.0 <= s <= 1.0


def test_aroused_signal_scores_higher_than_calm():
    sr = 16000
    rng = np.random.default_rng(0)
    # calm: quiet, steady low tone
    calm = _sine(120, sr=sr, amp=0.05)
    # aroused: loud, higher + wobbling pitch + noise (variable pitch/energy)
    t = np.arange(sr) / sr
    wobble = 260 + 60 * np.sin(2 * np.pi * 5 * t)          # vibrato -> pitch variability
    aroused = 0.6 * np.sin(2 * np.pi * wobble * t) + 0.05 * rng.standard_normal(sr)
    assert prosody.stress_index(aroused.astype(np.float32), sr) > prosody.stress_index(calm, sr)


def test_pause_ratio_detects_silence():
    sr = 16000
    speech = _sine(150, sr=sr, dur=0.5, amp=0.5)
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    signal = np.concatenate([speech, silence])
    feats = prosody.extract(signal, sr)
    assert feats.pause_ratio > 0.3    # ~half is silence


def test_empty_audio_is_safe():
    feats = prosody.extract(np.array([], dtype=np.float32), 16000)
    assert feats.pitch_mean_hz == 0.0
    assert feats.pause_ratio == 1.0
