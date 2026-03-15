#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def simulate(alpha: float, mu0: float, M: int, steps: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    mu = mu0
    out = [mu]
    for _ in range(steps):
        k = sum(1 for _ in range(M) if rng.random() < mu)
        mu = (1 - alpha) * mu + alpha * (k / M)
        out.append(mu)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mixed-environment alpha demo")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/alpha_mixing.yaml")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    alphas = [0.2, 0.5, 0.8, 1.0]
    series = {str(a): simulate(a, mu0=0.1, M=500, steps=30, seed=args.seed) for a in alphas}

    out = root / "GitHub" / "data" / "outputs" / "json" / "alpha_mixing_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"alphas": alphas, "series": series}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
