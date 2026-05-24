"""Attempt to load RAVDESS (or similar) from Hugging Face `datasets` and save up to 3 wavs.

This script tries a list of dataset identifiers and saves audio files it finds to demo_samples/.
"""
from __future__ import annotations

from pathlib import Path
import sys

from datasets import load_dataset

OUT = Path("../demo_samples").resolve()
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    "ravdess",
    "ravdess-emotion",
    "patrickvonplaten/ravdess",
    "emotion_ravdess",
]

saved = []
for name in CANDIDATES:
    if len(saved) >= 3:
        break
    try:
        print("Trying dataset:", name)
        ds = load_dataset(name, split="train")
        for ex in ds:
            if len(saved) >= 3:
                break
            # try common audio fields
            audio_field = None
            for key in ("audio", "speech", "file", "path", "wav"):
                if key in ex:
                    audio_field = key
                    break
            if audio_field is None:
                # sometimes nested dict fields
                for k, v in ex.items():
                    if isinstance(v, dict) and "array" in v and "sampling_rate" in v:
                        audio_field = k
                        break
            if audio_field is None:
                continue

            val = ex[audio_field]
            if isinstance(val, dict) and "array" in val:
                arr = val["array"]
                sr = val.get("sampling_rate", 16000)
                # save using soundfile
                import soundfile as sf
                out = OUT / f"demo_hf_{len(saved)+1}.wav"
                sf.write(out, arr, sr)
                print("Saved array->", out)
                saved.append(str(out))
            elif isinstance(val, str):
                # val is a path or url
                try:
                    # if it's a local path accessible by datasets, load bytes
                    with open(val, "rb") as fh:
                        out = OUT / f"demo_hf_{len(saved)+1}.wav"
                        out.write_bytes(fh.read())
                        print("Saved path->", out)
                        saved.append(str(out))
                except Exception:
                    # skip
                    continue
        if saved:
            print("Found audio files with dataset", name)
    except Exception as e:
        print("Failed to load", name, e)

print("Saved files:", saved)
if not saved:
    print("No audio files found via datasets. Consider manual download or provide URLs.")
    sys.exit(2)

print("Done.")
