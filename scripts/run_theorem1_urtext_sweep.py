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
            f"theorem1_{self.preset.name}_{self.scale.label}_"
            f"o{self.preset.max_order}_g{self.generations}_a{alpha_tag}"
        )

    @property
    def seed(self) -> int:
        raw = f"{self.version}:{self.index}:{self.run_name}".encode("utf-8")
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


def _threshold_crossing(rows: list[dict[str, Any]], key: str, threshold: float) -> int | None:
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if int(row.get("generation", 0)) > 0 and numeric <= threshold:
            return int(row["generation"])
    return None


def _row_for_order(row: dict[str, Any], order: int, suffix: str = "") -> float | None:
    key = f"distinct_{order}grams{suffix}"
    value = row.get(key)
    if value is None:
        return None
    return float(value)


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
        "initial_entropy_bits": float(first["token_entropy_bits"]),
        "final_entropy_bits": float(final["token_entropy_bits"]),
        "entropy_delta_bits": float(final["token_entropy_bits"]) - float(first["token_entropy_bits"]),
        "final_vocab_size": int(final["vocab_size"]),
        "final_vocab_ratio": float(final["vocab_ratio_vs_gen0"]),
        "vocab_ratio_leq_0p75_generation": _threshold_crossing(rows, "vocab_ratio_vs_gen0", 0.75),
        "vocab_ratio_leq_0p50_generation": _threshold_crossing(rows, "vocab_ratio_vs_gen0", 0.50),
        "final_max_order_distinct": int(final[f"distinct_{max_order}grams"]),
        "final_max_order_ratio": float(final[f"distinct_{max_order}grams_ratio_vs_gen0"]),
        "max_order_ratio_leq_0p50_generation": _threshold_crossing(
            rows,
            f"distinct_{max_order}grams_ratio_vs_gen0",
            0.50,
        ),
    }
    for order in range(1, max_order + 1):
        registry_row[f"final_distinct_{order}grams"] = int(final[f"distinct_{order}grams"])
        registry_row[f"final_distinct_{order}grams_ratio"] = float(final[f"distinct_{order}grams_ratio_vs_gen0"])

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


def _write_report(
    *,
    path: Path,
    registry_df: pd.DataFrame,
    sweep_name: str,
    alpha: float,
    generations: int,
) -> None:
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

    best_support = registry_df.nlargest(5, "final_max_order_ratio")[
        ["run_name", "preset_name", "scale_label", "final_max_order_ratio", "final_vocab_ratio"]
    ]
    worst_support = registry_df.nsmallest(5, "final_max_order_ratio")[
        ["run_name", "preset_name", "scale_label", "final_max_order_ratio", "final_vocab_ratio"]
    ]
    mean_by_scale = (
        registry_df.groupby("scale_label")[["final_vocab_ratio", "final_max_order_ratio", "entropy_delta_bits"]]
        .mean()
        .reset_index()
    )
    mean_by_preset = (
        registry_df.groupby("preset_name")[["final_vocab_ratio", "final_max_order_ratio", "entropy_delta_bits"]]
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
        "",
        "## Top final max-order support retention",
        _df_to_markdown(best_support),
        "",
        "## Bottom final max-order support retention",
        _df_to_markdown(worst_support),
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
) -> dict[str, str]:
    sweep_dir = ensure_dir(output_root / "sweeps" / stable_slug(f"{version}_{sweep_name}"))
    figures_dir = ensure_dir(sweep_dir / "figures")
    tables_dir = ensure_dir(sweep_dir / "tables")

    registry_df = pd.DataFrame(registry_rows).sort_values("experiment_index").reset_index(drop=True)
    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["experiment_index", "generation"]).reset_index(drop=True)

    registry_path = sweep_dir / "experiment_registry.csv"
    trajectory_path = sweep_dir / "trajectory_metrics.csv"
    _save_csv(registry_df, registry_path)
    _save_csv(trajectory_df, trajectory_path)

    scale_summary = (
        registry_df.groupby("scale_label")[["final_vocab_ratio", "final_max_order_ratio", "entropy_delta_bits"]]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    scale_summary.columns = ["_".join(col).strip("_") for col in scale_summary.columns]
    preset_summary = (
        registry_df.groupby("preset_name")[["final_vocab_ratio", "final_max_order_ratio", "entropy_delta_bits"]]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    preset_summary.columns = ["_".join(col).strip("_") for col in preset_summary.columns]
    _save_csv(scale_summary, tables_dir / "summary_by_scale.csv")
    _save_csv(preset_summary, tables_dir / "summary_by_preset.csv")

    support_pivot = registry_df.pivot(index="preset_name", columns="scale_label", values="final_max_order_ratio")
    vocab_pivot = registry_df.pivot(index="preset_name", columns="scale_label", values="final_vocab_ratio")
    entropy_pivot = registry_df.pivot(index="preset_name", columns="scale_label", values="entropy_delta_bits")
    support_pivot.to_csv(tables_dir / "final_max_order_support_ratio_pivot.csv")
    vocab_pivot.to_csv(tables_dir / "final_vocab_ratio_pivot.csv")
    entropy_pivot.to_csv(tables_dir / "entropy_delta_pivot.csv")

    _plot_metric_grid(
        trajectory_df,
        metric="vocab_ratio_vs_gen0",
        ylabel="Vocabulary ratio vs gen0",
        title="Theorem 1 sweep: vocabulary retention trajectories",
        path=figures_dir / "trajectory_vocab_ratio_by_scale.png",
        scales=scales,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="distinct_3grams_ratio_vs_gen0",
        ylabel="Distinct 3-gram ratio vs gen0",
        title="Theorem 1 sweep: trigram-support retention trajectories",
        path=figures_dir / "trajectory_trigram_ratio_by_scale.png",
        scales=scales,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="max_order_support_ratio_vs_gen0",
        ylabel="Max-order support ratio vs gen0",
        title="Theorem 1 sweep: max-order support retention trajectories",
        path=figures_dir / "trajectory_max_order_ratio_by_scale.png",
        scales=scales,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="token_entropy_bits",
        ylabel="Token entropy (bits)",
        title="Theorem 1 sweep: entropy trajectories",
        path=figures_dir / "trajectory_entropy_by_scale.png",
        scales=scales,
        presets=presets,
    )
    _plot_heatmap(
        support_pivot,
        title="Final max-order support ratio",
        cmap="YlGnBu",
        value_format=".3f",
        path=figures_dir / "final_max_order_support_ratio_heatmap.png",
    )
    _plot_heatmap(
        vocab_pivot,
        title="Final vocabulary ratio",
        cmap="YlOrRd",
        value_format=".3f",
        path=figures_dir / "final_vocab_ratio_heatmap.png",
    )
    _plot_heatmap(
        entropy_pivot,
        title="Entropy delta (final - initial)",
        cmap="coolwarm",
        value_format=".3f",
        path=figures_dir / "entropy_delta_heatmap.png",
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
        "registry_path": str(registry_path),
        "trajectory_path": str(trajectory_path),
        "report_path": str(report_path),
        "figure_paths": [
            str(figures_dir / "trajectory_vocab_ratio_by_scale.png"),
            str(figures_dir / "trajectory_trigram_ratio_by_scale.png"),
            str(figures_dir / "trajectory_max_order_ratio_by_scale.png"),
            str(figures_dir / "trajectory_entropy_by_scale.png"),
            str(figures_dir / "final_max_order_support_ratio_heatmap.png"),
            str(figures_dir / "final_vocab_ratio_heatmap.png"),
            str(figures_dir / "entropy_delta_heatmap.png"),
        ],
        "table_paths": [
            str(tables_dir / "summary_by_scale.csv"),
            str(tables_dir / "summary_by_preset.csv"),
            str(tables_dir / "final_max_order_support_ratio_pivot.csv"),
            str(tables_dir / "final_vocab_ratio_pivot.csv"),
            str(tables_dir / "entropy_delta_pivot.csv"),
        ],
    }
    (sweep_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 20-setting theorem-1 urtext sweep.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", default="V0_01")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--run-progress", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = ensure_dir(root / "GitHub" / "data" / "outputs" / "theorem1_urtext_sweep")
    presets = _build_presets()
    scales = _build_scales()
    experiments = _build_experiments(version=args.version, alpha=args.alpha, generations=args.generations)
    if args.limit is not None:
        experiments = experiments[: args.limit]

    registry_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    sweep_name = (
        f"theorem1_urtext_sweep_{len(experiments)}runs_"
        f"a{str(args.alpha).replace('.', 'p')}_g{args.generations}"
    )

    for spec in tqdm(experiments, desc="Theorem 1 sweep", unit="run"):
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
        selection_config = SelectionConfig(mode="none")
        run = run_ngram_regeneration_lab(
            version=args.version,
            run_name=spec.run_name,
            urtext_config=urtext_config,
            latent_config=latent_config,
            lab_config=lab_config,
            selection_config=selection_config,
            agent_config=None,
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
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
