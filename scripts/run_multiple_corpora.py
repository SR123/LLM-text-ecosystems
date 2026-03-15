#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight multiple-corpora sweep")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/multiple_corpora.yaml")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    split_root = root / "GitHub" / "corpora" / "splits"
    rows = []
    for corpus_dir in sorted([p for p in split_root.iterdir() if p.is_dir()]):
        train = corpus_dir / "train.txt"
        if not train.exists():
            continue
        text = train.read_text(encoding="utf-8", errors="ignore")
        words = text.split()
        rows.append({
            "corpus": corpus_dir.name,
            "tokens": len(words),
            "unique_tokens": len(set(words)),
            "type_token_ratio": (len(set(words)) / len(words)) if words else 0.0,
        })

    out_csv = root / "GitHub" / "data" / "outputs" / "csv" / "multiple_corpora_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["corpus", "tokens", "unique_tokens", "type_token_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
