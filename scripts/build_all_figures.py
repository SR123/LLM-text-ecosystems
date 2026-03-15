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

from drift_selection.plots import save_bar_plot, save_line_plot


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lightweight canonical figures")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    fig_root = root / "GitHub" / "figures"
    csv_root = root / "GitHub" / "data" / "outputs" / "csv"

    x = list(range(0, 21))
    y = [0.1 + 0.02 * i for i in x]
    save_line_plot(
        x,
        y,
        fig_root / "appendix" / "alpha_sweep_demo.pdf",
        fig_root / "appendix" / "alpha_sweep_demo.png",
        title="Alpha Sweep Demo",
        xlabel="generation",
        ylabel="minority_mass",
        csv_path=csv_root / "alpha_sweep_demo.csv",
    )

    labels = ["neutral", "viability", "anti_viability"]
    vals = [0.32, 0.51, 0.24]
    save_bar_plot(
        labels,
        vals,
        fig_root / "appendix" / "selection_rule_demo.pdf",
        fig_root / "appendix" / "selection_rule_demo.png",
        title="Selection Rule Comparison",
        ylabel="support_rate",
        csv_path=csv_root / "selection_rule_demo.csv",
    )

    print("Built figure demos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
