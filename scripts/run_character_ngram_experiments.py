#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.metrics import distinct_n
from drift_selection.ngram import NgramModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run character-level n-gram demo")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/ngram_character_level.yaml")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rng = random.Random(args.seed)

    source = root / "GitHub" / "corpora" / "splits" / "conan_doyle" / "train.txt"
    text = source.read_text(encoding="utf-8", errors="ignore") if source.exists() else "The game is afoot." * 100
    chars = list(text[:60000])

    model = NgramModel(order=4)
    model.fit(chars)
    seed_tokens = chars[:3] if len(chars) >= 3 else list("Hol")
    generated = model.generate(seed_tokens, length=1000, rng=rng)

    out_json = root / "GitHub" / "data" / "outputs" / "json" / "character_ngram_run.json"
    out_text = root / "GitHub" / "data" / "outputs" / "text_samples" / "character_ngram_sample.txt"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "distinct_3": distinct_n(generated, 3),
        "length": len(generated),
    }
    out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    out_text.write_text("".join(generated), encoding="utf-8")
    print("Character n-gram run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
