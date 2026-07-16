"""Download the RAVDESS speech zip from Zenodo and extract 3 example wav files into demo_samples/.

Targets chosen to provide contrasting emotions:
- calm (02) strong intensity actor 01
- angry (05) normal intensity actor 02
- neutral (01) normal intensity actor 03

If download fails, script exits with non-zero code.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

OUT_DIR = Path("../demo_samples").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ZIP_PATH = OUT_DIR / "Audio_Speech_Actors_01-24.zip"

TARGETS = {
    "calm_words_high_stress.wav": "03-01-02-02-01-01-01.wav",
    "positive_words_angry_tone.wav": "03-01-05-01-01-01-02.wav",
    "neutral_words_controlled_tone.wav": "03-01-01-01-01-01-03.wav",
}

print("Downloading", ZIP_URL)
with requests.get(ZIP_URL, stream=True, timeout=120) as r:
    if r.status_code != 200:
        print("Failed to download zip, status:", r.status_code)
        sys.exit(2)
    total = int(r.headers.get("content-length", 0))
    with open(ZIP_PATH, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
print("Saved zip to", ZIP_PATH)

print("Extracting targets...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    namelist = z.namelist()
    # search for files inside archive that end with the target filename
    for out_name, target_suffix in TARGETS.items():
        matches = [n for n in namelist if n.endswith(target_suffix)]
        if not matches:
            print("Target not found in zip:", target_suffix)
            continue
        # take first match
        member = matches[0]
        dest = OUT_DIR / out_name
        with z.open(member) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        print("Extracted", dest)

print("Done. Check demo_samples/")
