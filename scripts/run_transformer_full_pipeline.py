#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full tiny-transformer Conan Doyle pipeline")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_teacher_student.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    py = sys.executable
    scripts_dir = root / "GitHub" / "scripts"

    force_flag = ["--force"] if args.force else []

    _run([py, str(scripts_dir / "run_transformer_teacher_train.py"), "--root", str(root), "--config", args.config, *force_flag])
    _run([py, str(scripts_dir / "run_transformer_generate_environments.py"), "--root", str(root), "--config", args.config, "--mode", args.mode, *force_flag])
    _run([py, str(scripts_dir / "run_transformer_student_train.py"), "--root", str(root), "--config", args.config, "--mode", args.mode, *force_flag])
    _run([py, str(scripts_dir / "run_transformer_evaluation.py"), "--root", str(root), "--config", args.config, "--mode", args.mode])

    print("Full transformer pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
