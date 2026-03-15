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

from drift_selection.database import ExperimentDB
from drift_selection.metrics import distinct_n
from drift_selection.ngram import NgramModel
from drift_selection.utils import utc_now_iso


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight Conan Doyle n-gram experiment")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/ngram_conan_doyle.yaml")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rng = random.Random(args.seed)

    split_file = root / "GitHub" / "corpora" / "splits" / "conan_doyle" / "train.txt"
    if split_file.exists():
        text = split_file.read_text(encoding="utf-8", errors="ignore")
    else:
        text = "Mr Holmes was seated at the breakfast table with Dr Watson." * 50

    tokens = text.split()
    model = NgramModel(order=3)
    model.fit(tokens[: min(len(tokens), 50000)])

    seed_tokens = tokens[:2] if len(tokens) >= 2 else ["Sherlock", "Holmes"]
    generated = model.generate(seed_tokens, length=120, rng=rng)
    d2 = distinct_n(generated, 2)

    out_dir = root / "GitHub" / "data" / "outputs"
    text_out = out_dir / "text_samples" / "ngram_sample.txt"
    json_out = out_dir / "json" / "ngram_run.json"
    csv_out = out_dir / "csv" / "ngram_metrics.csv"
    text_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    text_out.write_text(" ".join(generated), encoding="utf-8")
    payload = {
        "run_name": "ngram_conan_doyle_smoke",
        "timestamp": utc_now_iso(),
        "distinct_2": d2,
        "sample_path": str(text_out.relative_to(root)),
    }
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_out.write_text("metric,value\ndistinct_2,{:.6f}\n".format(d2), encoding="utf-8")

    db = ExperimentDB(root / "GitHub" / "data" / "databases" / "experiment_index.sqlite")
    db.init_schema()
    run_id = db.start_run(
        {
            "run_name": payload["run_name"],
            "timestamp_start": payload["timestamp"],
            "git_version": "local-workspace",
            "script_name": "run_ngram_experiments.py",
            "corpus": "conan_doyle",
            "model_family": "ngram",
            "selection_rule": "neutral",
            "seed": args.seed,
            "config_path": args.config,
            "output_dir": str(out_dir.relative_to(root)),
            "status": "running",
        }
    )
    db.add_metric(run_id, "distinct_2", float(d2))
    db.add_artifact(run_id, "text_sample", str(text_out.relative_to(root)))
    db.end_run(run_id, utc_now_iso(), status="completed", metrics_summary=json.dumps({"distinct_2": d2}))
    db.export_registry_csv(root / "GitHub" / "data" / "outputs" / "csv" / "experiment_registry.csv")

    print(f"Completed n-gram experiment run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
