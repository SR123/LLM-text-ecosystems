#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.conan_doyle_figure3 import Figure3Config, run_figure3_reproduction  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the Conan Doyle Figure 3 n-gram experiment.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="GitHub/data/outputs/conan_doyle_figure3")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--transition-budget", type=int, default=120000)
    parser.add_argument("--one-shot-budget", type=int, default=20000)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--evaluation-trajectories", type=int, default=1000)
    parser.add_argument("--evaluation-horizon", type=int, default=100)
    parser.add_argument("--lookahead-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260309)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    cfg = Figure3Config(
        generations=args.generations,
        transition_budget=args.transition_budget,
        one_shot_budget=args.one_shot_budget,
        replicates=args.replicates,
        evaluation_trajectories=args.evaluation_trajectories,
        evaluation_horizon=args.evaluation_horizon,
        lookahead_horizon=args.lookahead_horizon,
        seed=args.seed,
    )
    outputs = run_figure3_reproduction(root=root, output_dir=output_dir, cfg=cfg)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
