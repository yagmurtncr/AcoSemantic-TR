"""Download a small set of RAVDESS sample files into demo_samples/.

This script attempts to download a few known filenames from the RAVDESS Zenodo record.
If downloads fail, it prints which URLs failed and exits with non-zero code.

Note: If Zenodo layout changes or the record requires manual access, provide direct links
or download the dataset manually and place three files into demo_samples/ with names:
- calm_words_high_stress.wav
- positive_words_angry_tone.wav
- neutral_words_controlled_tone.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

OUT_DIR = Path("../demo_samples").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate URLs constructed for RAVDESS (may change); tries several actors
BASE = "https://zenodo.org/record/1188976/files"
# Example filenames from RAVDESS naming pattern: 03-01-01-01-01-01-01.wav
FILENAMES = [
    "03-01-01-01-01-01-01.wav",
    "03-01-02-01-01-01-01.wav",
    "03-01-01-02-01-01-01.wav",
]

urls = []
for actor in range(1, 6):
    actor_dir = f"Actor_{actor:02d}"
    for name in FILENAMES:
        urls.append(f"{BASE}/{actor_dir}/{name}?download=1")

success = []
failed = []
for url in urls:
    if len(success) >= 3:
        break
    try:
        print("Downloading:", url)
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.content:
            out_name = OUT_DIR / f"demo_{len(success)+1}.wav"
            out_name.write_bytes(r.content)
            print("Saved:", out_name)
            success.append(str(out_name))
        else:
            print("Failed (status):", url, r.status_code)
            failed.append((url, r.status_code))
    except Exception as e:
        print("Error downloading", url, e)
        failed.append((url, str(e)))

print("\nSummary:")
print("Succeeded:", success)
print("Failed:", failed)

if not success:
    print("No files downloaded. Please download sample files manually into demo_samples/.")
    sys.exit(2)

print("Done.")
