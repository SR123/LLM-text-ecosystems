#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.markov import doob_h_transform, survival_probabilities


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Markov absorbing-state demo")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/markov_absorbing.yaml")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    P = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    alive = [True, True, False]
    h = survival_probabilities(P, alive, horizon=8)
    transformed = doob_h_transform(P, h[-1], alive)

    out = {
        "base_transition": P.tolist(),
        "alive": alive,
        "horizon": 8,
        "survival": h.tolist(),
        "doob_transformed": transformed.tolist(),
    }
    out_path = root / "GitHub" / "data" / "outputs" / "json" / "markov_absorbing_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
