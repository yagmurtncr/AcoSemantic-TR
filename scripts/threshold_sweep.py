"""Sweep POSITIVE and STRESS thresholds over a grid and recommend a pair.

Logic:
- Load all WAVs from demo_samples and compute (positivity, stress) using
  existing `analyze_audio_file` from `src.analysis`.
- Use demo_cases.json to identify 'ground truth' demo anomalies (cases where
  demo_case has values and previous results marked anomaly true).
- For each threshold pair, compute which files would be marked anomalous and
  score the pair by: true_positives - false_positives (higher is better).
- Recommend the pair with highest score, and print top candidates.
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import analyze_audio_file
from src.config import DEFAULT_ACOUSTIC_MODELS, DEFAULT_ASR_MODELS, DEFAULT_SENTIMENT_MODELS

DEMO = ROOT / "demo_samples"
CASES_PATH = DEMO / "demo_cases.json"

cases = json.loads(CASES_PATH.read_text(encoding="utf-8")) if CASES_PATH.exists() else {}

asr = DEFAULT_ASR_MODELS["Whisper Small"]
sent = DEFAULT_SENTIMENT_MODELS[list(DEFAULT_SENTIMENT_MODELS)[0]]
acoustic = DEFAULT_ACOUSTIC_MODELS[list(DEFAULT_ACOUSTIC_MODELS)[0]]

files = sorted(DEMO.glob("*.wav"))
scores = {}
print(f"Computing positivity/stress for {len(files)} files...")
for wav in files:
    try:
        case = cases.get(wav.name, {})
        res = analyze_audio_file(wav, asr, sent, acoustic, semantic_prompt=case.get("semantic_prompt"), stress_override=case.get("stress_override"))
        scores[wav.name] = (res.positivity_score, res.stress_score)
    except Exception as e:
        print("Error on", wav.name, e)

# Identify demo positives: earlier demo_mode True and anomaly True in results
results_path = DEMO / "demo_results.json"
results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
demo_positives = {name for name, r in results.items() if r.get("demo_mode") and r.get("anomaly")}

print(f"Demo positive (expected anomalies): {sorted(demo_positives)}")

# Grid search
pos_range = [round(x,2) for x in [0.5,0.55,0.6,0.65,0.7,0.75,0.8]]
stress_range = [round(x,2) for x in [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]]

def score_threshold(pthr, sthr):
    preds = {name for name,(p,s) in scores.items() if p>pthr and s>sthr}
    tp = len(preds & demo_positives)
    fp = len(preds - demo_positives)
    return tp - fp, tp, fp, preds

results_grid = []
for p,s in itertools.product(pos_range, stress_range):
    sc,tp,fp,preds = score_threshold(p,s)
    results_grid.append((sc,tp,fp,p, s, preds))

results_grid.sort(reverse=True, key=lambda x: (x[0], x[1], -x[2]))

print("Top candidates (score, tp, fp, pos_thr, stress_thr):")
for row in results_grid[:6]:
    sc,tp,fp,p,s,preds = row
    print(sc, tp, fp, p, s, sorted(preds)[:5])

best = results_grid[0]
_, best_tp, best_fp, best_p, best_s, best_preds = best
print("\nRecommended thresholds:")
print(f"POSITIVE_THRESHOLD = {best_p}\nSTRESS_THRESHOLD = {best_s}")
print(f"True positives: {best_tp}, false positives: {best_fp}")
