#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
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
    LatentGrammarConfig,
    RegenerationLabConfig,
    SelectionConfig,
    UrtextConfig,
    run_ngram_regeneration_lab,
)
from drift_selection.utils import ensure_dir, stable_slug, timestamp  # noqa: E402


@dataclass(frozen=True)
class UrtextPreset:
    name: str
    description: str
    mode: str
    max_order: int
    order_keep_probs: dict[int, float] | None = None


@dataclass(frozen=True)
class ScalePreset:
    vocab_size: int
    text_length: int

    @property
    def label(self) -> str:
        return f"v{self.vocab_size}_m{self.text_length}"


@dataclass(frozen=True)
class ExperimentSpec:
    index: int
    preset: UrtextPreset
    scale: ScalePreset
    alpha: float
    generations: int
    version: str

    @property
    def run_name(self) -> str:
        alpha_tag = str(self.alpha).replace(".", "p")
        return (
            f"theorem2_ref5_{self.preset.name}_{self.scale.label}_"
            f"o{self.preset.max_order}_g{self.generations}_a{alpha_tag}"
        )

    @property
    def seed(self) -> int:
        raw = f"{self.version}:{self.index}:{self.preset.name}:{self.scale.label}:{self.alpha}:{self.generations}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _build_presets() -> list[UrtextPreset]:
    return [
        UrtextPreset(
            name="iid_uniform",
            description="Urtext sampled i.i.d. uniformly from the vocabulary.",
            mode="synthetic_iid",
            max_order=3,
            order_keep_probs=None,
        ),
        UrtextPreset(
            name="latent_tri_dense",
            description="Synthetic latent trigram world with relatively dense higher-order support.",
            mode="synthetic_latent",
            max_order=3,
            order_keep_probs={2: 0.80, 3: 0.60},
        ),
        UrtextPreset(
            name="latent_tri_balanced",
            description="Synthetic latent trigram world with balanced higher-order support.",
            mode="synthetic_latent",
            max_order=3,
            order_keep_probs={2: 0.50, 3: 0.25},
        ),
        UrtextPreset(
            name="latent_tri_sparse",
            description="Synthetic latent trigram world with sparse higher-order support.",
            mode="synthetic_latent",
            max_order=3,
            order_keep_probs={2: 0.30, 3: 0.08},
        ),
        UrtextPreset(
            name="latent_four_cascade",
            description="Synthetic latent 4-gram world with cascading support probabilities.",
            mode="synthetic_latent",
            max_order=4,
            order_keep_probs={2: 0.50, 3: 0.25, 4: 0.125},
        ),
    ]


def _build_scales() -> list[ScalePreset]:
    return [
        ScalePreset(vocab_size=100, text_length=1000),
        ScalePreset(vocab_size=100, text_length=10000),
        ScalePreset(vocab_size=200, text_length=1000),
        ScalePreset(vocab_size=200, text_length=10000),
    ]


def _build_experiments(*, version: str, alpha: float, generations: int) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    index = 1
    for preset in _build_presets():
        for scale in _build_scales():
            experiments.append(
                ExperimentSpec(
                    index=index,
                    preset=preset,
                    scale=scale,
                    alpha=alpha,
                    generations=generations,
                    version=version,
                )
            )
            index += 1
    return experiments


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _plot_metric_grid(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    scales: list[ScalePreset],
    presets: list[UrtextPreset],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()
    colors = plt.get_cmap("tab10")
    for axis, scale in zip(axes_flat, scales):
        scale_df = df[df["scale_label"] == scale.label]
        for idx, preset in enumerate(presets):
            preset_df = scale_df[scale_df["preset_name"] == preset.name].sort_values("generation")
            if preset_df.empty or metric not in preset_df.columns:
                continue
            axis.plot(
                preset_df["generation"],
                preset_df[metric],
                marker="o",
                linewidth=2.0,
                markersize=3.5,
                label=preset.name,
                color=colors(idx),
            )
        axis.set_title(f"{scale.label.replace('_', ', ')}")
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


def _summarise_run(spec: ExperimentSpec, run: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [dict(row) for row in run.metrics_rows]
    first = rows[0]
    final = rows[-1]
    max_order = int(spec.preset.max_order)
    registry_row: dict[str, Any] = {
        "experiment_index": spec.index,
        "version": spec.version,
        "run_name": spec.run_name,
        "preset_name": spec.preset.name,
        "preset_description": spec.preset.description,
        "urtext_mode": spec.preset.mode,
        "max_order": max_order,
        "vocab_size": spec.scale.vocab_size,
        "text_length": spec.scale.text_length,
        "scale_label": spec.scale.label,
        "alpha": spec.alpha,
        "generations": spec.generations,
        "seed": spec.seed,
        "run_dir": str(run.paths.run_dir),
        "metrics_final_path": str(run.paths.metrics_final_path),
        "summary_path": str(run.paths.summary_path),
        "sample_text_path": str(run.paths.sample_text_path),
        "final_generation": int(final["generation"]),
        "final_vocab_ratio": float(final["vocab_ratio_vs_gen0"]),
        "final_max_order_ratio": float(final[f"distinct_{max_order}grams_ratio_vs_gen0"]),
        "final_desirable_window_share": float(final.get("desirable_window_share", 0.0)),
        "final_undesirable_window_share": float(final.get("undesirable_window_share", 0.0)),
        "final_agent_desirable_rate": float(final.get("agent_desirable_rate", 0.0)),
        "final_agent_undesirable_rate": float(final.get("agent_undesirable_rate", 0.0)),
        "final_agent_avg_search_cost": float(final.get("agent_avg_search_cost", 0.0)),
        "initial_entropy_bits": float(first["token_entropy_bits"]),
        "final_entropy_bits": float(final["token_entropy_bits"]),
        "entropy_delta_bits": float(final["token_entropy_bits"]) - float(first["token_entropy_bits"]),
    }
    trajectory_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched.update(
            {
                "experiment_index": spec.index,
                "run_name": spec.run_name,
                "preset_name": spec.preset.name,
                "preset_description": spec.preset.description,
                "urtext_mode": spec.preset.mode,
                "max_order": max_order,
                "vocab_size_target": spec.scale.vocab_size,
                "scale_label": spec.scale.label,
                "seed": spec.seed,
                "max_order_support_ratio_vs_gen0": float(row[f"distinct_{max_order}grams_ratio_vs_gen0"]),
                "max_order_distinct": int(row[f"distinct_{max_order}grams"]),
            }
        )
        trajectory_rows.append(enriched)
    return registry_row, trajectory_rows


def _load_neutral_registry(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    keep = [
        "preset_name",
        "scale_label",
        "final_vocab_ratio",
        "final_max_order_ratio",
        "entropy_delta_bits",
    ]
    out = df[keep].copy()
    out = out.rename(
        columns={
            "final_vocab_ratio": "neutral_final_vocab_ratio",
            "final_max_order_ratio": "neutral_final_max_order_ratio",
            "entropy_delta_bits": "neutral_entropy_delta_bits",
        }
    )
    return out


def _write_report(
    *,
    path: Path,
    registry_df: pd.DataFrame,
    sweep_name: str,
    alpha: float,
    generations: int,
) -> None:
    best_desirable = registry_df.nlargest(5, "final_desirable_window_share")[
        ["run_name", "preset_name", "scale_label", "final_desirable_window_share", "final_agent_desirable_rate"]
    ]
    best_delta = registry_df.nlargest(5, "delta_vs_neutral_max_order_ratio")[
        ["run_name", "preset_name", "scale_label", "delta_vs_neutral_max_order_ratio", "final_desirable_window_share"]
    ]
    worst_delta = registry_df.nsmallest(5, "delta_vs_neutral_max_order_ratio")[
        ["run_name", "preset_name", "scale_label", "delta_vs_neutral_max_order_ratio", "final_desirable_window_share"]
    ]
    mean_by_scale = (
        registry_df.groupby("scale_label")[
            [
                "final_desirable_window_share",
                "final_agent_desirable_rate",
                "final_max_order_ratio",
                "delta_vs_neutral_max_order_ratio",
            ]
        ]
        .mean()
        .reset_index()
    )
    mean_by_preset = (
        registry_df.groupby("preset_name")[
            [
                "final_desirable_window_share",
                "final_agent_desirable_rate",
                "final_max_order_ratio",
                "delta_vs_neutral_max_order_ratio",
            ]
        ]
        .mean()
        .reset_index()
    )
    lines = [
        f"# {sweep_name}",
        "",
        f"- created_at: {timestamp()}",
        f"- experiments: {len(registry_df)}",
        f"- alpha: {alpha}",
        f"- generations: {generations}",
        "- utility: all urtext 5-grams are desirable; unseen 5-grams are treated as undesirable",
        "- agent: sampled 5-step lookahead, publish a random desirable 5-token extension when found",
        "",
        "## Top final desirable-window share",
        _df_to_markdown(best_desirable),
        "",
        "## Largest max-order retention gains vs neutral recursion",
        _df_to_markdown(best_delta),
        "",
        "## Smallest max-order retention gains vs neutral recursion",
        _df_to_markdown(worst_delta),
        "",
        "## Mean final metrics by scale",
        _df_to_markdown(mean_by_scale),
        "",
        "## Mean final metrics by urtext preset",
        _df_to_markdown(mean_by_preset),
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_outputs(
    *,
    output_root: Path,
    sweep_name: str,
    version: str,
    alpha: float,
    generations: int,
    presets: list[UrtextPreset],
    scales: list[ScalePreset],
    registry_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    neutral_registry_path: Path | None,
) -> dict[str, str]:
    sweep_dir = ensure_dir(output_root / "sweeps" / stable_slug(f"{version}_{sweep_name}"))
    figures_dir = ensure_dir(sweep_dir / "figures")
    tables_dir = ensure_dir(sweep_dir / "tables")

    registry_df = pd.DataFrame(registry_rows).sort_values("experiment_index").reset_index(drop=True)
    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["experiment_index", "generation"]).reset_index(drop=True)

    neutral_df = _load_neutral_registry(neutral_registry_path)
    if neutral_df is not None:
        registry_df = registry_df.merge(neutral_df, on=["preset_name", "scale_label"], how="left")
        registry_df["delta_vs_neutral_max_order_ratio"] = (
            registry_df["final_max_order_ratio"] - registry_df["neutral_final_max_order_ratio"]
        )
        registry_df["delta_vs_neutral_vocab_ratio"] = (
            registry_df["final_vocab_ratio"] - registry_df["neutral_final_vocab_ratio"]
        )
        registry_df["delta_vs_neutral_entropy_delta"] = (
            registry_df["entropy_delta_bits"] - registry_df["neutral_entropy_delta_bits"]
        )
    else:
        registry_df["neutral_final_max_order_ratio"] = float("nan")
        registry_df["neutral_final_vocab_ratio"] = float("nan")
        registry_df["neutral_entropy_delta_bits"] = float("nan")
        registry_df["delta_vs_neutral_max_order_ratio"] = float("nan")
        registry_df["delta_vs_neutral_vocab_ratio"] = float("nan")
        registry_df["delta_vs_neutral_entropy_delta"] = float("nan")

    registry_path = sweep_dir / "experiment_registry.csv"
    trajectory_path = sweep_dir / "trajectory_metrics.csv"
    _save_csv(registry_df, registry_path)
    _save_csv(trajectory_df, trajectory_path)

    scale_summary = (
        registry_df.groupby("scale_label")[
            [
                "final_desirable_window_share",
                "final_agent_desirable_rate",
                "final_max_order_ratio",
                "delta_vs_neutral_max_order_ratio",
            ]
        ]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    scale_summary.columns = ["_".join(col).strip("_") for col in scale_summary.columns]
    preset_summary = (
        registry_df.groupby("preset_name")[
            [
                "final_desirable_window_share",
                "final_agent_desirable_rate",
                "final_max_order_ratio",
                "delta_vs_neutral_max_order_ratio",
            ]
        ]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    preset_summary.columns = ["_".join(col).strip("_") for col in preset_summary.columns]
    _save_csv(scale_summary, tables_dir / "summary_by_scale.csv")
    _save_csv(preset_summary, tables_dir / "summary_by_preset.csv")

    desirable_pivot = registry_df.pivot(index="preset_name", columns="scale_label", values="final_desirable_window_share")
    agent_pivot = registry_df.pivot(index="preset_name", columns="scale_label", values="final_agent_desirable_rate")
    delta_support_pivot = registry_df.pivot(index="preset_name", columns="scale_label", values="delta_vs_neutral_max_order_ratio")
    delta_vocab_pivot = registry_df.pivot(index="preset_name", columns="scale_label", values="delta_vs_neutral_vocab_ratio")
    desirable_pivot.to_csv(tables_dir / "final_desirable_window_share_pivot.csv")
    agent_pivot.to_csv(tables_dir / "final_agent_desirable_rate_pivot.csv")
    delta_support_pivot.to_csv(tables_dir / "delta_vs_neutral_max_order_ratio_pivot.csv")
    delta_vocab_pivot.to_csv(tables_dir / "delta_vs_neutral_vocab_ratio_pivot.csv")

    _plot_metric_grid(
        trajectory_df,
        metric="desirable_window_share",
        ylabel="Desirable 5-gram share",
        title="Theorem 2 sweep: desirable-window share trajectories",
        path=figures_dir / "trajectory_desirable_share_by_scale.png",
        scales=scales,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="agent_desirable_rate",
        ylabel="Agent desirable-publication rate",
        title="Theorem 2 sweep: agent desirable-publication trajectories",
        path=figures_dir / "trajectory_agent_desirable_rate_by_scale.png",
        scales=scales,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="max_order_support_ratio_vs_gen0",
        ylabel="Max-order support ratio vs gen0",
        title="Theorem 2 sweep: max-order support trajectories",
        path=figures_dir / "trajectory_max_order_ratio_by_scale.png",
        scales=scales,
        presets=presets,
    )
    _plot_heatmap(
        desirable_pivot,
        title="Final desirable-window share",
        cmap="YlGnBu",
        value_format=".3f",
        path=figures_dir / "final_desirable_window_share_heatmap.png",
    )
    _plot_heatmap(
        agent_pivot,
        title="Final agent desirable-publication rate",
        cmap="YlOrRd",
        value_format=".3f",
        path=figures_dir / "final_agent_desirable_rate_heatmap.png",
    )
    _plot_heatmap(
        delta_support_pivot,
        title="Delta vs neutral: final max-order support ratio",
        cmap="coolwarm",
        value_format=".3f",
        path=figures_dir / "delta_vs_neutral_max_order_ratio_heatmap.png",
    )
    _plot_heatmap(
        delta_vocab_pivot,
        title="Delta vs neutral: final vocabulary ratio",
        cmap="coolwarm",
        value_format=".3f",
        path=figures_dir / "delta_vs_neutral_vocab_ratio_heatmap.png",
    )

    report_path = sweep_dir / "report.md"
    _write_report(
        path=report_path,
        registry_df=registry_df,
        sweep_name=sweep_name,
        alpha=alpha,
        generations=generations,
    )

    manifest = {
        "sweep_name": sweep_name,
        "version": version,
        "created_at": timestamp(),
        "output_root": str(output_root),
        "neutral_registry_path": str(neutral_registry_path) if neutral_registry_path is not None else None,
        "registry_path": str(registry_path),
        "trajectory_path": str(trajectory_path),
        "report_path": str(report_path),
        "figure_paths": [
            str(figures_dir / "trajectory_desirable_share_by_scale.png"),
            str(figures_dir / "trajectory_agent_desirable_rate_by_scale.png"),
            str(figures_dir / "trajectory_max_order_ratio_by_scale.png"),
            str(figures_dir / "final_desirable_window_share_heatmap.png"),
            str(figures_dir / "final_agent_desirable_rate_heatmap.png"),
            str(figures_dir / "delta_vs_neutral_max_order_ratio_heatmap.png"),
            str(figures_dir / "delta_vs_neutral_vocab_ratio_heatmap.png"),
        ],
        "table_paths": [
            str(tables_dir / "summary_by_scale.csv"),
            str(tables_dir / "summary_by_preset.csv"),
            str(tables_dir / "final_desirable_window_share_pivot.csv"),
            str(tables_dir / "final_agent_desirable_rate_pivot.csv"),
            str(tables_dir / "delta_vs_neutral_max_order_ratio_pivot.csv"),
            str(tables_dir / "delta_vs_neutral_vocab_ratio_pivot.csv"),
        ],
    }
    (sweep_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a theorem-2 sweep with urtext 5-grams as the desirable set.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", default="V0_01")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--candidate-trials", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--run-progress", action="store_true")
    parser.add_argument("--neutral-registry", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = ensure_dir(root / "GitHub" / "data" / "outputs" / "theorem2_urtext_reference_sweep")
    presets = _build_presets()
    scales = _build_scales()
    experiments = _build_experiments(version=args.version, alpha=args.alpha, generations=args.generations)
    if args.limit is not None:
        experiments = experiments[: args.limit]

    if args.neutral_registry:
        neutral_registry_path = Path(args.neutral_registry).resolve()
    else:
        neutral_registry_path = (
            root
            / "GitHub"
            / "data"
            / "outputs"
            / "theorem1_urtext_sweep"
            / "sweeps"
            / stable_slug(f"{args.version}_theorem1_urtext_sweep_20runs_a{str(args.alpha).replace('.', 'p')}_g{args.generations}")
            / "experiment_registry.csv"
        )

    registry_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    sweep_name = (
        f"theorem2_ref5_sweep_{len(experiments)}runs_"
        f"a{str(args.alpha).replace('.', 'p')}_g{args.generations}_trials{args.candidate_trials}"
    )

    for spec in tqdm(experiments, desc="Theorem 2 sweep", unit="run"):
        urtext_config = UrtextConfig(
            mode=spec.preset.mode,
            vocab_size=spec.scale.vocab_size,
        )
        latent_config = LatentGrammarConfig(
            max_order=spec.preset.max_order,
            order_keep_probs=spec.preset.order_keep_probs,
            exact_support=False,
            support_cache_limit=8192,
        )
        lab_config = RegenerationLabConfig(
            text_length=spec.scale.text_length,
            generations=spec.generations,
            alpha=spec.alpha,
            max_order=spec.preset.max_order,
            restart_probability=0.0,
            sample_retained_block=True,
        )
        selection_config = SelectionConfig(
            mode="reference_rgram",
            span=args.span,
            lookahead_samples=args.candidate_trials,
            reference_min_count=1,
            reference_unseen_category="undesirable",
        )
        agent_config = AgentConfig(
            mode="sampled_lookahead",
            publish_strategy="desirable_then_random",
            lookahead_depth=args.span,
            candidate_trials=args.candidate_trials,
            publish_horizon=args.span,
            branch_factor=max(6, min(spec.scale.vocab_size, 32)),
            max_expansions=max(4096, args.candidate_trials * args.span * 4),
            utility_weight=1.0,
            logprob_weight=0.0,
            rollback_penalty=0.0,
        )
        run = run_ngram_regeneration_lab(
            version=args.version,
            run_name=spec.run_name,
            urtext_config=urtext_config,
            latent_config=latent_config,
            lab_config=lab_config,
            selection_config=selection_config,
            agent_config=agent_config,
            seed=spec.seed,
            output_root=output_root,
            resume=not args.force_rebuild,
            force_rebuild=args.force_rebuild,
            progress_bar=args.run_progress,
        )
        registry_row, run_trajectory_rows = _summarise_run(spec, run)
        registry_rows.append(registry_row)
        trajectory_rows.extend(run_trajectory_rows)

    manifest = _generate_outputs(
        output_root=output_root,
        sweep_name=sweep_name,
        version=args.version,
        alpha=args.alpha,
        generations=args.generations,
        presets=presets,
        scales=scales,
        registry_rows=registry_rows,
        trajectory_rows=trajectory_rows,
        neutral_registry_path=neutral_registry_path if neutral_registry_path.exists() else None,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
