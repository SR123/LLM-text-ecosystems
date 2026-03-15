#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build appendix table exports")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    db_path = root / "GitHub" / "data" / "databases" / "experiment_index.sqlite"
    out_path = root / "GitHub" / "data" / "outputs" / "csv" / "appendix_table_runs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if db_path.exists():
        with sqlite3.connect(db_path) as con:
            rows = con.execute(
                "SELECT run_id,run_name,corpus,model_family,selection_rule,status FROM runs ORDER BY run_id"
            ).fetchall()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "run_name", "corpus", "model_family", "selection_rule", "status"])
        for row in rows:
            w.writerow(row)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
