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

from drift_selection.corpus import train_val_test_split_words, write_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/val/test splits from cleaned corpora")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    corpora_root = root / "GitHub" / "corpora"
    cleaned_root = corpora_root / "cleaned"
    split_root = corpora_root / "splits"

    for corpus_dir in sorted([p for p in cleaned_root.iterdir() if p.is_dir()]):
        text_chunks = []
        for txt in sorted(corpus_dir.glob("*.txt")):
            text_chunks.append(txt.read_text(encoding="utf-8", errors="ignore"))
        if not text_chunks:
            continue
        merged = "\n\n".join(text_chunks)
        train, val, test = train_val_test_split_words(merged, seed=args.seed)
        write_splits(split_root / corpus_dir.name, train, val, test)
        print(f"Built splits for {corpus_dir.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
