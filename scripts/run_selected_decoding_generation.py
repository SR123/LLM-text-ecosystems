#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.selected_decoding import selected_decoding_step


def main() -> int:
    parser = argparse.ArgumentParser(description="Run selected-decoding toy generation")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/selected_decoding.yaml")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    base_probs = {"A": 0.5, "B": 0.3, "C": 0.2}
    viability = {"A": 0.2, "B": 0.8, "C": 0.9}
    selected = selected_decoding_step(base_probs, viability, horizon_weight=1.5)

    out = root / "GitHub" / "data" / "outputs" / "json" / "selected_decoding_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base": base_probs, "viability": viability, "selected": selected}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
