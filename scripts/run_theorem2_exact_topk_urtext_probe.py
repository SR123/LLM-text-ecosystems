#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import run_theorem2_exact_5gram_reference_sweep as base  # noqa: E402
import run_theorem2_exact_soft_policy_followup as follow  # noqa: E402

from drift_selection.checkpoints import atomic_save_json  # noqa: E402
from drift_selection.ngram_regeneration_lab import (  # noqa: E402
    LatentGrammarConfig,
    RegenerationLabConfig,
    UrtextConfig,
    build_empirical_model,
    build_reference_rgram_utility,
    build_urtext,
    metrics_for_generation,
)
from drift_selection.utils import timestamp  # noqa: E402


SPAN = 5
VERSION = "V0_01"
OUTPUT_ROOT = GH_ROOT / "data" / "outputs" / "theorem2_topk_urtext_probe"


@dataclass(frozen=True)
class TopKSpec:
    top_k: int
    beta: float
    alpha: float = 0.10
    generations: int = 20
    vocab_size: int = 100
    text_length: int = 10000
    max_order: int = 3
    preset_name: str = "iid_uniform"

    @property
    def run_name(self) -> str:
        beta_tag = str(self.beta).replace(".", "p")
        return (
            f"theorem2_exact_topk_urtext_top{self.top_k}_beta{beta_tag}_"
            f"{self.preset_name}_v{self.vocab_size}_m{self.text_length}_"
            f"o{self.max_order}_g{self.generations}_a{str(self.alpha).replace('.', 'p')}"
        )

    @property
    def seed(self) -> int:
        raw = (
            f"{VERSION}:{self.run_name}:{self.top_k}:{self.beta}:{self.alpha}:"
            f"{self.generations}:{self.vocab_size}:{self.text_length}:{self.max_order}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _policy(beta: float) -> follow.PolicyPreset:
    return follow.PolicyPreset(
        name=f"soft_prob_beta_{beta:g}".replace(".", "p"),
        description=(
            "Probability-tilted exact 5-lookahead with fixed top-K urtext utility and "
            f"beta={beta:g}."
        ),
        beta=float(beta),
    )


def _run_single(spec: TopKSpec) -> dict[str, Any]:
    paths = base._ensure_run_paths(output_root=OUTPUT_ROOT, run_name=spec.run_name, version=VERSION)
    urtext_config = UrtextConfig(mode="synthetic_iid", vocab_size=spec.vocab_size)
    latent_config = LatentGrammarConfig(
        max_order=spec.max_order,
        order_keep_probs=None,
        exact_support=False,
        support_cache_limit=8192,
    )
    lab_config = RegenerationLabConfig(
        text_length=spec.text_length,
        generations=spec.generations,
        alpha=spec.alpha,
        max_order=spec.max_order,
        restart_probability=0.0,
        sample_retained_block=True,
    )
    bundle = build_urtext(urtext_config, latent_config, int(lab_config.text_length), seed=spec.seed)
    original_tokens = [int(token) for token in bundle.token_ids[: int(lab_config.text_length)]]
    utility = build_reference_rgram_utility(
        original_tokens,
        span=SPAN,
        min_count=1,
        max_patterns=spec.top_k,
        label=f"urtext_top{spec.top_k}",
        unseen_category="undesirable",
    )
    manifest = {
        "run_name": spec.run_name,
        "version": VERSION,
        "created_at": timestamp(),
        "seed": int(spec.seed),
        "policy_name": f"soft_prob_beta_{spec.beta:g}".replace(".", "p"),
        "policy_beta": float(spec.beta),
        "utility_basis": "fixed_topk_urtext_reference",
        "utility_top_k": int(spec.top_k),
        "lookahead_span": SPAN,
        "urtext_config": base._json_ready(urtext_config.__dict__),
        "latent_config": base._json_ready(latent_config.__dict__),
        "lab_config": base._json_ready(lab_config.__dict__),
        "notes": [
            "Exact exhaustive depth-5 lookahead.",
            "Desirable set is the fixed top-K most frequent distinct 5-grams from the original urtext U0.",
            "Soft continuation weighting is proportional to P_model(extension | context) * exp(beta * 1_good(extension)).",
            "KL compares the lookahead first-token law to the one-step empirical n-gram law with maximal context.",
        ],
        "id_to_label": base._json_ready(bundle.id_to_label),
    }
    atomic_save_json(paths.manifest_path, manifest)

    current_tokens = list(original_tokens)
    metrics_rows: list[dict[str, Any]] = []
    all_decision_rows: list[dict[str, Any]] = []
    all_context_rows: list[dict[str, Any]] = []
    snapshots: dict[int, list[int]] = {0: list(current_tokens)}
    policy = _policy(spec.beta)

    for generation in range(0, spec.generations + 1):
        planner = follow._planner_for_policy(
            policy=policy,
            model=build_empirical_model(current_tokens, int(lab_config.max_order)),
            utility=utility,
        )
        consistency_metrics, context_rows = follow._evaluate_self_consistency(
            tokens=current_tokens,
            max_order=int(lab_config.max_order),
            planner=planner,
            generation=generation,
        )
        all_context_rows.extend(context_rows)

        if generation == 0:
            agent_stats = follow._decision_summary([])
            row = metrics_for_generation(
                current_tokens,
                lab_config,
                generation=0,
                baseline=None,
                utility=utility,
                agent_stats=agent_stats,
            )
        else:
            baseline = {
                key: float(metrics_rows[0][key])
                for key in [f"distinct_{order}grams" for order in range(1, int(lab_config.max_order) + 1)]
            }
            agent_stats = follow._decision_summary(generation_decisions)
            row = metrics_for_generation(
                current_tokens,
                lab_config,
                generation=generation,
                baseline=baseline,
                utility=utility,
                agent_stats=agent_stats,
            )
        row.update(consistency_metrics)
        row["policy_name"] = policy.name
        row["policy_beta"] = float(spec.beta)
        row["reference_source_generation"] = 0
        row["reference_unique_grams"] = int(len(utility.desirable_set))
        row["reference_top_k"] = int(spec.top_k)
        metrics_rows.append(row)

        if generation == spec.generations:
            break

        generation_rng = random.Random(spec.seed + 1009 * (generation + 1))
        generated_tokens, generation_decisions = follow._generate_replacement(
            source_tokens=current_tokens,
            config=lab_config,
            planner=planner,
            rng=generation_rng,
            generation=generation + 1,
            policy_name=policy.name,
        )
        all_decision_rows.extend(generation_decisions)
        current_tokens = base._combine_generation(
            previous_tokens=current_tokens,
            generated_tokens=generated_tokens,
            config=lab_config,
            rng=generation_rng,
        )
        base._save_generation_snapshot(paths.state_dir, generation + 1, current_tokens)
        if generation + 1 in {spec.generations // 2, spec.generations}:
            snapshots[generation + 1] = list(current_tokens)
        atomic_save_json(
            paths.checkpoint_path,
            {
                "run_name": spec.run_name,
                "version": VERSION,
                "last_completed_generation": generation,
                "updated_at": timestamp(),
                "policy_name": policy.name,
                "utility_top_k": int(spec.top_k),
            },
        )

    base._write_rows_csv(paths.metrics_final_path, metrics_rows)
    base._write_rows_csv(paths.metrics_partial_path, metrics_rows)
    base._write_rows_csv(paths.tables_dir / "decision_log.csv", all_decision_rows)
    base._write_rows_csv(paths.tables_dir / "context_metrics.csv", all_context_rows)
    base._write_sample_texts(paths, snapshots, {int(k): str(v) for k, v in bundle.id_to_label.items()})
    figure_paths = follow._write_run_figures(
        paths,
        metrics_rows,
        title_suffix=f"{VERSION} | {spec.run_name} | {timestamp()}",
    )
    summary = {
        "run_name": spec.run_name,
        "version": VERSION,
        "policy_name": policy.name,
        "policy_beta": float(spec.beta),
        "utility_top_k": int(spec.top_k),
        "finished_at": timestamp(),
        "metrics_final_path": str(paths.metrics_final_path),
        "decision_log_path": str(paths.tables_dir / "decision_log.csv"),
        "context_log_path": str(paths.tables_dir / "context_metrics.csv"),
        "figure_paths": figure_paths,
    }
    atomic_save_json(paths.summary_path, summary)
    atomic_save_json(paths.checkpoint_path, {"status": "finished", **summary})

    df = json.loads(pd_to_json(metrics_rows))
    initial = df[0]
    final = df[-1]
    min_row = min(df, key=lambda row: float(row["lookahead_kl_bits"]))
    return {
        "run_name": spec.run_name,
        "seed": int(spec.seed),
        "top_k": int(spec.top_k),
        "beta": float(spec.beta),
        "initial_kl_bits": float(initial["lookahead_kl_bits"]),
        "final_kl_bits": float(final["lookahead_kl_bits"]),
        "min_kl_bits": float(min_row["lookahead_kl_bits"]),
        "generation_of_min_kl": int(min_row["generation"]),
        "final_top1_match_rate": float(final["lookahead_top1_match_rate"]),
        "final_desirable_window_share": float(final["desirable_window_share"]),
        "run_dir": str(paths.run_dir),
        "figure_path": str(paths.figures_dir / "soft_policy_self_consistency.png"),
    }


def pd_to_json(rows: list[dict[str, Any]]) -> str:
    import pandas as pd

    return pd.DataFrame(rows).to_json(orient="records")


def main() -> int:
    specs = [
        TopKSpec(top_k=250, beta=3.0),
        TopKSpec(top_k=500, beta=3.0),
        TopKSpec(top_k=1000, beta=3.0),
        TopKSpec(top_k=250, beta=6.0),
        TopKSpec(top_k=500, beta=6.0),
        TopKSpec(top_k=1000, beta=6.0),
    ]
    rows = [_run_single(spec) for spec in specs]
    summary_path = OUTPUT_ROOT / "topk_probe_summary.csv"
    base._write_rows_csv(summary_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
