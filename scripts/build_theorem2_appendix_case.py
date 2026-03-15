#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.utils import ensure_dir, timestamp  # noqa: E402


ROOT = Path("/Users/sorenriis/Documents/Collaboration")
MIDM_RUNS = ROOT / "GitHub" / "data" / "outputs" / "theorem2_soft_policy_midM_probe" / "runs"
OUTPUT_DIR = ROOT / "GitHub" / "data" / "outputs" / "theorem2_appendix_case"
APPENDIX_FIG_DIR = ROOT / "GitHub" / "appendix" / "figures" / "generated"

RUNS = {
    "hard_uniform_good": MIDM_RUNS / "v0_01_theorem2_softfollow_hard_uniform_good_prevref_iid_uniform_v100_m10000_o3_g30_a0p1",
    "soft_prob_beta_3": MIDM_RUNS / "v0_01_theorem2_softfollow_soft_prob_beta_3_prevref_iid_uniform_v100_m10000_o3_g30_a0p1",
    "soft_prob_beta_6": MIDM_RUNS / "v0_01_theorem2_softfollow_soft_prob_beta_6_prevref_iid_uniform_v100_m10000_o3_g30_a0p1",
}

LABELS = {
    "hard_uniform_good": "Hard good-only",
    "soft_prob_beta_3": "Soft beta=3",
    "soft_prob_beta_6": "Soft beta=6",
}

COLORS = {
    "hard_uniform_good": "#ba3d0f",
    "soft_prob_beta_3": "#1d6f42",
    "soft_prob_beta_6": "#2157a5",
}


def _load_run(path: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(path / "metrics_final.csv")
    return manifest, metrics


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, float):
                vals.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(APPENDIX_FIG_DIR)

    trajectories: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for policy_name, run_dir in RUNS.items():
        manifest, metrics = _load_run(run_dir)
        metrics = metrics.copy()
        metrics["policy_name"] = policy_name
        metrics["policy_label"] = LABELS[policy_name]
        trajectories.append(metrics)
        first = metrics.iloc[0]
        final = metrics.iloc[-1]
        min_row = metrics.loc[metrics["lookahead_kl_bits"].idxmin()]
        summary_rows.append(
            {
                "policy_name": policy_name,
                "policy_label": LABELS[policy_name],
                "seed": int(manifest["seed"]),
                "initial_kl_bits": float(first["lookahead_kl_bits"]),
                "min_kl_bits": float(min_row["lookahead_kl_bits"]),
                "generation_of_min_kl": int(min_row["generation"]),
                "final_kl_bits": float(final["lookahead_kl_bits"]),
                "final_top1_match_rate": float(final["lookahead_top1_match_rate"]),
                "final_desirable_window_share": float(final["desirable_window_share"]),
                "final_agent_desirable_rate": float(final["agent_desirable_rate"]),
            }
        )

    traj_df = pd.concat(trajectories, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("policy_name").reset_index(drop=True)
    traj_df.to_csv(OUTPUT_DIR / "trajectory_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for policy_name, group in traj_df.groupby("policy_name"):
        group = group.sort_values("generation")
        label = LABELS[policy_name]
        color = COLORS[policy_name]
        axes[0].plot(group["generation"], group["lookahead_kl_bits"], marker="o", linewidth=2.0, markersize=3.2, label=label, color=color)
        axes[1].plot(group["generation"], group["lookahead_top1_match_rate"], marker="o", linewidth=2.0, markersize=3.2, label=label, color=color)
    axes[0].set_title("Exact KL(bits) to one-step law")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("KL(bits)")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Top-1 agreement")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Match rate")
    axes[1].grid(alpha=0.25)
    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Exact 5-lookahead vs one-step law | iid urtext | V=100 | M=10000 | alpha=0.10")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    pdf_path = OUTPUT_DIR / "figure_theorem2_soft_policy_case.pdf"
    png_path = OUTPUT_DIR / "figure_theorem2_soft_policy_case.png"
    fig.savefig(pdf_path, dpi=180, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    appendix_pdf = APPENDIX_FIG_DIR / "figure_appx_theorem2_soft_policy_case.pdf"
    appendix_png = APPENDIX_FIG_DIR / "figure_appx_theorem2_soft_policy_case.png"
    appendix_pdf.write_bytes(pdf_path.read_bytes())
    appendix_png.write_bytes(png_path.read_bytes())

    report_lines = [
        "# theorem2 appendix case",
        "",
        f"- created_at: {timestamp()}",
        "- urtext preset: iid_uniform",
        "- utility basis: previous_generation_reference",
        "- V: 100",
        "- M: 10000",
        "- alpha: 0.10",
        "- generations: 30",
        "- exact lookahead span: 5",
        "",
        "## Summary",
        _df_to_markdown(summary_df),
        "",
        "## Interpretation",
        "The hard policy keeps publishing only desirable 5-grams, but its first-token law stays noticeably different from the one-step n-gram law.",
        "The soft policies still favor desirable continuations, but because they remain probability-tilted rather than hard-truncated, the KL gap collapses by roughly two orders of magnitude.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "summary_path": str(OUTPUT_DIR / "summary.csv"),
                "trajectory_path": str(OUTPUT_DIR / "trajectory_metrics.csv"),
                "figure_pdf": str(pdf_path),
                "appendix_figure_pdf": str(appendix_pdf),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
