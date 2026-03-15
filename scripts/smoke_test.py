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

from drift_selection.database import ExperimentDB
from drift_selection.ngram import NgramModel


def assert_exists(path: Path, msg: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{msg}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight end-to-end smoke test")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    required_dirs = [
        root / "Nat_Paper",
        root / "GitHub" / "src" / "drift_selection",
        root / "GitHub" / "scripts",
        root / "GitHub" / "data" / "databases",
    ]
    for d in required_dirs:
        assert_exists(d, "Missing required directory")

    # Verify figure paths for paper compile
    required_figs = [
        root / "Nat_Paper" / "figures" / "figure_theorem2_one_shot_bars.pdf",
        root / "Nat_Paper" / "figures" / "figure_metrics_plain_vs_plain.pdf",
        root / "Nat_Paper" / "figures" / "figure_metrics_with_explicit_lookahead.pdf",
        root / "Nat_Paper" / "figures" / "figure_perplexity_supplement.pdf",
    ]
    for f in required_figs:
        assert_exists(f, "Missing paper figure")

    # Initialize SQLite registry
    db_path = root / "GitHub" / "data" / "databases" / "experiment_index.sqlite"
    db = ExperimentDB(db_path)
    db.init_schema()

    # Tiny n-gram run
    text = "sherlock holmes investigated the curious case with doctor watson " * 40
    tokens = text.split()
    model = NgramModel(order=3)
    model.fit(tokens)
    generated = model.generate(tokens[:2], length=60)
    if len(generated) != 60:
        raise RuntimeError("n-gram generation failed")

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
