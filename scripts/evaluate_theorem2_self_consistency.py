#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.ngram_regeneration_lab import (  # noqa: E402
    AgentConfig,
    RegenerationLabConfig,
    SelectionConfig,
    build_empirical_model,
    build_selection_utility,
    choose_agent_extension,
    load_generation_snapshot,
)
from drift_selection.utils import ensure_dir, stable_slug, timestamp  # noqa: E402


@dataclass(frozen=True)
class EvalConfig:
    contexts_per_generation: int = 16
    rollouts_per_context: int = 12
    max_generations: int | None = None
    progress_bar: bool = True


def _kl_bits(q: dict[int, float], p: dict[int, float]) -> float:
    total = 0.0
    for token, q_prob in q.items():
        if q_prob <= 0.0:
            continue
        p_prob = float(p.get(token, 0.0))
        if p_prob <= 0.0:
            return float("inf")
        total += float(q_prob) * math.log2(float(q_prob) / p_prob)
    return float(total)


def _js_bits(q: dict[int, float], p: dict[int, float]) -> float:
    keys = set(q) | set(p)
    m = {token: 0.5 * float(q.get(token, 0.0) + p.get(token, 0.0)) for token in keys}
    return 0.5 * _kl_bits(q, m) + 0.5 * _kl_bits(p, m)


def _tv_distance(q: dict[int, float], p: dict[int, float]) -> float:
    keys = set(q) | set(p)
    return 0.5 * sum(abs(float(q.get(token, 0.0)) - float(p.get(token, 0.0))) for token in keys)


def _sample_contexts(
    tokens: list[int],
    *,
    context_len: int,
    sample_size: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    if context_len <= 0:
        return [tuple()]
    positions = list(range(0, max(1, len(tokens) - context_len)))
    if not positions:
        return [tuple(tokens[-context_len:])] if tokens else [tuple()]
    if len(positions) <= sample_size:
        picks = positions
    else:
        picks = sorted(rng.sample(positions, sample_size))
    return [tuple(int(x) for x in tokens[pos:pos + context_len]) for pos in picks]


def _estimate_agent_first_token_distribution(
    model: Any,
    context: tuple[int, ...],
    utility: Any,
    agent_config: AgentConfig,
    *,
    rollouts: int,
    rng: random.Random,
) -> dict[int, float]:
    counts: dict[int, int] = {}
    for _ in range(max(1, rollouts)):
        candidate = choose_agent_extension(
            model,
            list(context),
            utility,
            agent_config=agent_config,
            rng=rng,
        )
        first_token = int(candidate.extension[0])
        counts[first_token] = counts.get(first_token, 0) + 1
    total = float(sum(counts.values()) or 1.0)
    return {token: count / total for token, count in counts.items()}


def _evaluate_generation(
    *,
    tokens: list[int],
    generation: int,
    lab_config: RegenerationLabConfig,
    utility: Any,
    agent_config: AgentConfig,
    seed: int,
    contexts_per_generation: int,
    rollouts_per_context: int,
) -> dict[str, Any]:
    context_len = max(0, int(lab_config.max_order) - 1)
    rng = random.Random(seed)
    contexts = _sample_contexts(
        tokens,
        context_len=context_len,
        sample_size=contexts_per_generation,
        rng=rng,
    )
    model = build_empirical_model(tokens, int(lab_config.max_order))

    kl_values: list[float] = []
    js_values: list[float] = []
    tv_values: list[float] = []
    order_used_values: list[int] = []
    support_sizes: list[int] = []
    agent_top1_match: list[float] = []

    for ctx_index, context in enumerate(contexts):
        base_dist, order_used = model.distribution(context)
        agent_rng = random.Random(seed + 1009 * (generation + 1) + 9176 * (ctx_index + 1))
        agent_dist = _estimate_agent_first_token_distribution(
            model,
            context,
            utility,
            agent_config,
            rollouts=rollouts_per_context,
            rng=agent_rng,
        )
        base_top = max(base_dist.items(), key=lambda item: item[1])[0]
        agent_top = max(agent_dist.items(), key=lambda item: item[1])[0]
        kl_values.append(_kl_bits(agent_dist, base_dist))
        js_values.append(_js_bits(agent_dist, base_dist))
        tv_values.append(_tv_distance(agent_dist, base_dist))
        order_used_values.append(int(order_used))
        support_sizes.append(len(base_dist))
        agent_top1_match.append(1.0 if int(base_top) == int(agent_top) else 0.0)

    return {
        "generation": generation,
        "contexts_sampled": len(contexts),
        "rollouts_per_context": int(rollouts_per_context),
        "lookahead_kl_bits": float(sum(kl_values) / len(kl_values)),
        "lookahead_js_bits": float(sum(js_values) / len(js_values)),
        "lookahead_tv_distance": float(sum(tv_values) / len(tv_values)),
        "base_avg_support_size": float(sum(support_sizes) / len(support_sizes)),
        "base_avg_order_used": float(sum(order_used_values) / len(order_used_values)),
        "top1_match_rate": float(sum(agent_top1_match) / len(agent_top1_match)),
    }


def _plot_trajectory(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    scales: list[str],
    presets: list[str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()
    colors = plt.get_cmap("tab10")
    for axis, scale in zip(axes_flat, scales):
        scale_df = df[df["scale_label"] == scale]
        for idx, preset in enumerate(presets):
            preset_df = scale_df[scale_df["preset_name"] == preset].sort_values("generation")
            if preset_df.empty:
                continue
            axis.plot(
                preset_df["generation"],
                preset_df[metric],
                marker="o",
                linewidth=2.0,
                markersize=3.5,
                label=preset,
                color=colors(idx),
            )
        axis.set_title(scale.replace("_", ", "))
        axis.set_xlabel("Generation")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    value_format: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iat[row_idx, col_idx]
            label = "nan" if pd.isna(value) else format(float(value), value_format)
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines: list[str] = []
    for _, row in df.iterrows():
        values: list[str] = []
        for column in df.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body_lines])


def _evaluate_run(
    run_dir: Path,
    *,
    eval_config: EvalConfig,
) -> pd.DataFrame:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    selection_config = manifest.get("selection_config")
    agent_config_raw = manifest.get("agent_config")
    if not selection_config or selection_config.get("mode") == "none" or not agent_config_raw:
        return pd.DataFrame()

    lab_config = RegenerationLabConfig(**manifest["lab_config"])
    utility, _ = build_selection_utility(
        SelectionConfig(**selection_config),
        urtext_tokens=load_generation_snapshot(run_dir / "state", 0),
        seed=int(manifest["seed"]),
    )
    agent_config = AgentConfig(**agent_config_raw)

    last_generation = int(json.loads((run_dir / "checkpoint_state.json").read_text(encoding="utf-8"))["last_completed_generation"])
    if eval_config.max_generations is not None:
        last_generation = min(last_generation, int(eval_config.max_generations))

    rows: list[dict[str, Any]] = []
    iterator = range(0, last_generation + 1)
    for generation in iterator:
        tokens = load_generation_snapshot(run_dir / "state", generation)
        metrics = _evaluate_generation(
            tokens=tokens,
            generation=generation,
            lab_config=lab_config,
            utility=utility,
            agent_config=agent_config,
            seed=int(manifest["seed"]) + 7919 * generation,
            contexts_per_generation=eval_config.contexts_per_generation,
            rollouts_per_context=eval_config.rollouts_per_context,
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate theorem-2 self-consistency via KL divergence.")
    parser.add_argument("--sweep-registry", required=True, help="Path to theorem-2 sweep experiment_registry.csv")
    parser.add_argument("--contexts-per-generation", type=int, default=16)
    parser.add_argument("--rollouts-per-context", type=int, default=12)
    parser.add_argument("--max-generations", type=int, default=None)
    args = parser.parse_args()

    registry_path = Path(args.sweep_registry).resolve()
    sweep_dir = registry_path.parent
    sweep_name = sweep_dir.name
    root_dir = ensure_dir(sweep_dir / "self_consistency")
    figures_dir = ensure_dir(root_dir / "figures")
    tables_dir = ensure_dir(root_dir / "tables")
    runs_dir = sweep_dir.parent.parent / "runs"

    registry_df = pd.read_csv(registry_path)
    eval_config = EvalConfig(
        contexts_per_generation=args.contexts_per_generation,
        rollouts_per_context=args.rollouts_per_context,
        max_generations=args.max_generations,
    )

    per_generation_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    iterator = registry_df.itertuples(index=False)
    for row in tqdm(list(iterator), desc="Self-consistency", unit="run"):
        run_dir = Path(getattr(row, "run_dir"))
        eval_df = _evaluate_run(run_dir, eval_config=eval_config)
        if eval_df.empty:
            continue
        eval_df["run_name"] = getattr(row, "run_name")
        eval_df["preset_name"] = getattr(row, "preset_name")
        eval_df["scale_label"] = getattr(row, "scale_label")
        eval_df["urtext_mode"] = getattr(row, "urtext_mode")
        per_generation_rows.extend(eval_df.to_dict(orient="records"))

        initial = eval_df.iloc[0]
        final = eval_df.iloc[-1]
        final_rows.append(
            {
                "run_name": getattr(row, "run_name"),
                "preset_name": getattr(row, "preset_name"),
                "scale_label": getattr(row, "scale_label"),
                "urtext_mode": getattr(row, "urtext_mode"),
                "initial_lookahead_kl_bits": float(initial["lookahead_kl_bits"]),
                "final_lookahead_kl_bits": float(final["lookahead_kl_bits"]),
                "delta_lookahead_kl_bits": float(final["lookahead_kl_bits"]) - float(initial["lookahead_kl_bits"]),
                "initial_lookahead_js_bits": float(initial["lookahead_js_bits"]),
                "final_lookahead_js_bits": float(final["lookahead_js_bits"]),
                "initial_lookahead_tv_distance": float(initial["lookahead_tv_distance"]),
                "final_lookahead_tv_distance": float(final["lookahead_tv_distance"]),
                "final_top1_match_rate": float(final["top1_match_rate"]),
                "contexts_per_generation": int(final["contexts_sampled"]),
                "rollouts_per_context": int(final["rollouts_per_context"]),
            }
        )

    per_generation_df = pd.DataFrame(per_generation_rows).sort_values(["run_name", "generation"]).reset_index(drop=True)
    final_df = pd.DataFrame(final_rows).sort_values(["scale_label", "preset_name"]).reset_index(drop=True)

    per_generation_path = tables_dir / "self_consistency_by_generation.csv"
    final_path = tables_dir / "self_consistency_final_summary.csv"
    per_generation_df.to_csv(per_generation_path, index=False)
    final_df.to_csv(final_path, index=False)

    scales = sorted(final_df["scale_label"].unique())
    presets = sorted(final_df["preset_name"].unique())
    _plot_trajectory(
        per_generation_df,
        metric="lookahead_kl_bits",
        ylabel="KL(agent || base) [bits]",
        title="Theorem 2 self-consistency: KL trajectories",
        path=figures_dir / "trajectory_lookahead_kl_bits.png",
        scales=scales,
        presets=presets,
    )
    _plot_trajectory(
        per_generation_df,
        metric="lookahead_js_bits",
        ylabel="JS(agent, base) [bits]",
        title="Theorem 2 self-consistency: JS trajectories",
        path=figures_dir / "trajectory_lookahead_js_bits.png",
        scales=scales,
        presets=presets,
    )
    _plot_trajectory(
        per_generation_df,
        metric="lookahead_tv_distance",
        ylabel="TV(agent, base)",
        title="Theorem 2 self-consistency: TV trajectories",
        path=figures_dir / "trajectory_lookahead_tv_distance.png",
        scales=scales,
        presets=presets,
    )

    kl_pivot = final_df.pivot(index="preset_name", columns="scale_label", values="final_lookahead_kl_bits")
    js_pivot = final_df.pivot(index="preset_name", columns="scale_label", values="final_lookahead_js_bits")
    tv_pivot = final_df.pivot(index="preset_name", columns="scale_label", values="final_lookahead_tv_distance")
    kl_pivot.to_csv(tables_dir / "final_lookahead_kl_bits_pivot.csv")
    js_pivot.to_csv(tables_dir / "final_lookahead_js_bits_pivot.csv")
    tv_pivot.to_csv(tables_dir / "final_lookahead_tv_distance_pivot.csv")

    _plot_heatmap(
        kl_pivot,
        title="Final KL(agent || base) [bits]",
        cmap="YlGnBu",
        value_format=".3f",
        path=figures_dir / "final_lookahead_kl_bits_heatmap.png",
    )
    _plot_heatmap(
        js_pivot,
        title="Final JS(agent, base) [bits]",
        cmap="YlOrRd",
        value_format=".3f",
        path=figures_dir / "final_lookahead_js_bits_heatmap.png",
    )
    _plot_heatmap(
        tv_pivot,
        title="Final TV(agent, base)",
        cmap="coolwarm",
        value_format=".3f",
        path=figures_dir / "final_lookahead_tv_distance_heatmap.png",
    )

    mean_by_scale = (
        final_df.groupby("scale_label")[
            ["initial_lookahead_kl_bits", "final_lookahead_kl_bits", "delta_lookahead_kl_bits", "final_top1_match_rate"]
        ]
        .mean()
        .reset_index()
    )
    mean_by_preset = (
        final_df.groupby("preset_name")[
            ["initial_lookahead_kl_bits", "final_lookahead_kl_bits", "delta_lookahead_kl_bits", "final_top1_match_rate"]
        ]
        .mean()
        .reset_index()
    )
    mean_by_scale.to_csv(tables_dir / "self_consistency_mean_by_scale.csv", index=False)
    mean_by_preset.to_csv(tables_dir / "self_consistency_mean_by_preset.csv", index=False)

    best = final_df.nsmallest(5, "final_lookahead_kl_bits")[
        ["run_name", "preset_name", "scale_label", "final_lookahead_kl_bits", "delta_lookahead_kl_bits", "final_top1_match_rate"]
    ]
    worst = final_df.nlargest(5, "final_lookahead_kl_bits")[
        ["run_name", "preset_name", "scale_label", "final_lookahead_kl_bits", "delta_lookahead_kl_bits", "final_top1_match_rate"]
    ]
    report_lines = [
        f"# Self-consistency evaluation for {sweep_name}",
        "",
        f"- created_at: {timestamp()}",
        f"- contexts_per_generation: {eval_config.contexts_per_generation}",
        f"- rollouts_per_context: {eval_config.rollouts_per_context}",
        "- metric: empirical context-average KL(agent first-token law || base next-token law)",
        "",
        "## Smallest final KL",
        _df_to_markdown(best),
        "",
        "## Largest final KL",
        _df_to_markdown(worst),
        "",
        "## Mean by scale",
        _df_to_markdown(mean_by_scale),
        "",
        "## Mean by preset",
        _df_to_markdown(mean_by_preset),
        "",
    ]
    report_path = root_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at": timestamp(),
        "sweep_registry": str(registry_path),
        "evaluation_root": str(root_dir),
        "contexts_per_generation": eval_config.contexts_per_generation,
        "rollouts_per_context": eval_config.rollouts_per_context,
        "per_generation_path": str(per_generation_path),
        "final_summary_path": str(final_path),
        "figure_paths": [
            str(figures_dir / "trajectory_lookahead_kl_bits.png"),
            str(figures_dir / "trajectory_lookahead_js_bits.png"),
            str(figures_dir / "trajectory_lookahead_tv_distance.png"),
            str(figures_dir / "final_lookahead_kl_bits_heatmap.png"),
            str(figures_dir / "final_lookahead_js_bits_heatmap.png"),
            str(figures_dir / "final_lookahead_tv_distance_heatmap.png"),
        ],
    }
    (root_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
