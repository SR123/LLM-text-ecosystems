#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Backward-compatible entrypoint for transformer teacher/student pipeline")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_teacher_student.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    script = root / "GitHub" / "scripts" / "run_transformer_full_pipeline.py"
    cmd = [sys.executable, str(script), "--root", str(root), "--config", args.config, "--mode", args.mode]
    if args.force:
        cmd.append("--force")

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
