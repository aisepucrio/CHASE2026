"""
Convert resultado_manual.csv into JSON in the format:
{
  "output_v1": {
    "segment_000": {"principal_emocao_detectada": "Happiness"},
    ...
  },
  "output_v2": { ... }
}

Usage:
  python tools/convert_manual_csv_to_json.py resultado_manual.csv resultado_manual.json
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any


def convert(csv_path: Path) -> Dict[str, Any]:
    outputs: Dict[str, Dict[str, Dict[str, str]]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("TITLE") or "").strip()
            segment = (row.get("SEGMENT") or "").strip()
            label = (row.get("LABLE") or "").strip()

            if not title or not segment or not label:
                # skip incomplete lines
                continue

            output_key = f"output_{title}"
            segment_key = Path(segment).stem  # e.g., segment_000 from segment_000.m4a

            outputs.setdefault(output_key, {})[segment_key] = {
                "principal_emocao_detectada": label
            }

    return outputs


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/convert_manual_csv_to_json.py <input_csv> <output_json>")
        sys.exit(1)

    csv_path = Path(sys.argv[1]).resolve()
    json_path = Path(sys.argv[2]).resolve()

    if not csv_path.exists():
        print(f"Input CSV not found: {csv_path}")
        sys.exit(2)

    outputs = convert(csv_path)

    json_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    # Simple report
    total_segments = sum(len(v) for v in outputs.values())
    print(f"Wrote {total_segments} segments into {json_path}")
    print("Top-level keys:", ", ".join(outputs.keys()))


if __name__ == "__main__":
    main()
