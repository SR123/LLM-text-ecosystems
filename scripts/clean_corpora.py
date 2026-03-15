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

from drift_selection.cleaning import clean_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean raw corpora into normalized UTF-8 text")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    corpora_root = root / "GitHub" / "corpora"

    raw_root = corpora_root / "raw"
    cleaned_root = corpora_root / "cleaned"

    for corpus_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        out_dir = cleaned_root / corpus_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for txt in sorted(corpus_dir.glob("*.txt")):
            dst = out_dir / txt.name
            clean_file(txt, dst)
        print(f"Cleaned corpus: {corpus_dir.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
