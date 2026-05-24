import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external_datasets" / "TurEV-DB"
DEST = ROOT / "demo_samples"

DEST.mkdir(parents=True, exist_ok=True)

wav_files = [p for p in SRC.rglob("*.wav") if p.is_file()]
if not wav_files:
    print("No wav files found in TurEV-DB")
    raise SystemExit(1)

# Prefer files in 'sound' subfolders if available
sound_files = [p for p in wav_files if '/sound/' in str(p).replace('\\', '/')]
candidates = sound_files if sound_files else wav_files

count = min(20, len(candidates))
selected = random.sample(candidates, count)

copied = []
for i, p in enumerate(selected, start=1):
    dest_name = f"turev_{i}_{p.name}"
    dest_path = DEST / dest_name
    shutil.copy2(p, dest_path)
    copied.append(dest_path.name)

print("Copied files:")
for n in copied:
    print(n)
