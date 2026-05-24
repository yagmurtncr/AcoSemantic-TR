"""Run the project's analysis pipeline on all files in demo_samples/ and print/save summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow direct execution without manually setting PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import analyze_audio_file
from src.config import DEFAULT_ASR_MODELS, DEFAULT_SENTIMENT_MODELS, DEFAULT_ACOUSTIC_MODELS

DEMO = ROOT / "demo_samples"
OUT = DEMO / "demo_results.json"
CASES_PATH = DEMO / "demo_cases.json"

if CASES_PATH.exists():
    demo_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
else:
    demo_cases = {}

asr = DEFAULT_ASR_MODELS["Whisper Small"]
sent = DEFAULT_SENTIMENT_MODELS[list(DEFAULT_SENTIMENT_MODELS)[0]]
acoustic = DEFAULT_ACOUSTIC_MODELS[list(DEFAULT_ACOUSTIC_MODELS)[0]]

results = {}
for wav in sorted(DEMO.glob("*.wav")):
    print("Analyzing:", wav.name)
    try:
        case = demo_cases.get(wav.name, {})
        res = analyze_audio_file(
            wav,
            asr,
            sent,
            acoustic,
            semantic_prompt=case.get("semantic_prompt"),
            stress_override=case.get("stress_override"),
        )
        results[wav.name] = {
            "transcript": res.transcript,
            "semantic_prompt": case.get("semantic_prompt"),
            "positivity": res.positivity_score,
            "stress": res.stress_score,
            "anomaly": res.anomaly_detected,
            "discordance": res.discordance_score,
            "verdict": res.verdict,
            "sentiment_model": res.metadata.get("sentiment_model"),
            "acoustic_model": res.metadata.get("acoustic_model"),
            "acoustic_mode": res.acoustic_mode,
            "demo_mode": bool(case),
        }
        print(json.dumps(results[wav.name], ensure_ascii=False, indent=2))
    except Exception as e:
        print("Error analyzing", wav.name, e)

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print("Results written to", OUT)
