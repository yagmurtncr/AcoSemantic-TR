"""Generate a simple human-readable summary from demo_results.json."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "demo_samples"
SOURCE = DEMO / "demo_results.json"
JSON_OUT = DEMO / "batch_summary.json"
MD_OUT = DEMO / "batch_summary.md"


def main() -> int:
    if not SOURCE.exists():
        print(f"Source file not found: {SOURCE}")
        return 1

    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = list(data.items())
    total = len(items)
    anomalies = sum(1 for _, item in items if item.get("anomaly"))
    normal = total - anomalies
    avg_positivity = sum(float(item.get("positivity", 0.0)) for _, item in items) / total if total else 0.0
    avg_stress = sum(float(item.get("stress", 0.0)) for _, item in items) / total if total else 0.0
    acoustic_modes = Counter(str(item.get("acoustic_mode", "unknown")) for _, item in items)
    top_anomalies = sorted(
        ((name, item) for name, item in items if item.get("anomaly")),
        key=lambda pair: float(pair[1].get("discordance", 0.0)),
        reverse=True,
    )[:5]

    summary = {
        "total_files": total,
        "anomalies": anomalies,
        "normal": normal,
        "anomaly_rate": round((anomalies / total) if total else 0.0, 4),
        "average_positivity": round(avg_positivity, 4),
        "average_stress": round(avg_stress, 4),
        "acoustic_modes": dict(acoustic_modes),
        "top_anomalies": [
            {
                "file": name,
                "discordance": item.get("discordance"),
                "verdict": item.get("verdict"),
            }
            for name, item in top_anomalies
        ],
    }

    md_lines = [
        "# Batch Ozeti",
        "",
        f"- Toplam dosya: {summary['total_files']}",
        f"- Anomali sayisi: {summary['anomalies']}",
        f"- Normal sayisi: {summary['normal']}",
        f"- Anomali orani: {summary['anomaly_rate']}",
        f"- Ortalama pozitiflik: {summary['average_positivity']}",
        f"- Ortalama stres: {summary['average_stress']}",
        "",
        "## Akustik Mod Dagilimi",
    ]
    for mode, count in summary["acoustic_modes"].items():
        md_lines.append(f"- {mode}: {count}")

    md_lines.extend(["", "## En Yuksek Cakisma Skorlu Anomaliler"])
    if top_anomalies:
        for name, item in top_anomalies:
            md_lines.append(f"- {name}: {item.get('discordance')} | {item.get('verdict')}")
    else:
        md_lines.append("- Anomali bulunamadi.")

    JSON_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    MD_OUT.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())