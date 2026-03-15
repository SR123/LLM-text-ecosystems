#!/usr/bin/env python3
from __future__ import annotations

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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_theorem2_exact_5gram_reference_sweep as base  # noqa: E402

GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.checkpoints import atomic_save_json, atomic_save_text  # noqa: E402
from drift_selection.ngram_regeneration_lab import build_empirical_model  # noqa: E402
from drift_selection.utils import ensure_dir, stable_slug, timestamp  # noqa: E402


SPAN = 5


@dataclass(frozen=True)
class PolicyPreset:
    name: str
    description: str
    beta: float | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    index: int
    preset: base.UrtextPreset
    scale: base.ScalePreset
    policy: PolicyPreset
    alpha: float
    generations: int
    version: str

    @property
    def run_name(self) -> str:
        alpha_tag = str(self.alpha).replace(".", "p")
        return (
            f"theorem2_softfollow_{self.policy.name}_prevref_"
            f"{self.preset.name}_{self.scale.label}_"
            f"o{self.preset.max_order}_g{self.generations}_a{alpha_tag}"
        )

    @property
    def panel_label(self) -> str:
        return f"{self.policy.name}|{self.scale.label}"

    @property
    def seed(self) -> int:
        return base.ExperimentSpec(
            index=self.index,
            preset=self.preset,
            scale=self.scale,
            utility_basis=base.UtilityBasisPreset(
                name="previous_generation_reference",
                description="Desirable 5-grams come from the previous generation.",
            ),
            alpha=self.alpha,
            generations=self.generations,
            version=self.version,
        ).seed


@dataclass
class PolicyContextSummary:
    policy_name: str
    active_pool: str
    first_token_distribution: dict[int, float]
    desirable_mass: float
    fallback_indicator: float
    all_extension_count: int | None = None
    desirable_extension_count: int | None = None


@dataclass
class DecisionMetadata:
    policy_name: str
    active_pool: str
    first_token_distribution: dict[int, float]
    desirable_mass: float
    fallback_indicator: float
    chosen_category: str
    all_extension_count: int | None = None
    desirable_extension_count: int | None = None


class HardCountPlanner:
    def __init__(self, *, model: Any, utility: Any):
        self.model = model
        self.utility = utility
        self.inner = base.ExactLookaheadPlanner(
            model=model,
            desirable_grams=list(utility.desirable_set),
            span=SPAN,
        )

    def first_token_summary(self, context: tuple[int, ...] | list[int]) -> PolicyContextSummary:
        policy = self.inner.first_token_policy(context)
        total_all = int(sum(policy.all_counts.values()))
        desirable_mass = (
            float(policy.desirable_extensions) / float(total_all or 1)
        )
        return PolicyContextSummary(
            policy_name="hard_uniform_good",
            active_pool=policy.active_pool,
            first_token_distribution=policy.active_distribution,
            desirable_mass=float(desirable_mass),
            fallback_indicator=1.0 if policy.active_pool == "fallback_all" else 0.0,
            all_extension_count=total_all,
            desirable_extension_count=int(policy.desirable_extensions),
        )

    def sample_extension(
        self,
        context: tuple[int, ...] | list[int],
        rng: random.Random,
    ) -> tuple[tuple[int, ...], DecisionMetadata]:
        extension, policy = self.inner.sample_extension(context, rng)
        total_all = int(sum(policy.all_counts.values()))
        desirable_mass = float(policy.desirable_extensions) / float(total_all or 1)
        chosen_category = "desirable" if tuple(extension) in self.utility.desirable_set else "undesirable"
        return extension, DecisionMetadata(
            policy_name="hard_uniform_good",
            active_pool=policy.active_pool,
            first_token_distribution=policy.active_distribution,
            desirable_mass=float(desirable_mass),
            fallback_indicator=1.0 if policy.active_pool == "fallback_all" else 0.0,
            chosen_category=chosen_category,
            all_extension_count=total_all,
            desirable_extension_count=int(policy.desirable_extensions),
        )


class SoftProbTiltPlanner(base.ExactLookaheadPlanner):
    def __init__(self, *, model: Any, utility: Any, beta: float):
        super().__init__(model=model, desirable_grams=list(utility.desirable_set), span=SPAN)
        self.utility = utility
        self.beta = float(beta)
        self.exp_beta = math.exp(self.beta)
        self.policy_name = f"soft_prob_beta_{self.beta:g}".replace(".", "p")
        self._good_prob_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}

    def good_prob_mass(self, context: tuple[int, ...] | list[int], prefix: tuple[int, ...]) -> float:
        ctx = self._normalize_context(context)
        norm_prefix = tuple(int(x) for x in prefix)
        key = (ctx, norm_prefix)
        if key in self._good_prob_cache:
            return self._good_prob_cache[key]
        if len(norm_prefix) >= self.span:
            value = 1.0 if norm_prefix in self.utility.desirable_set else 0.0
            self._good_prob_cache[key] = value
            return value
        children = self.prefix_children.get(norm_prefix, ())
        if not children:
            self._good_prob_cache[key] = 0.0
            return 0.0
        _, support_set, dist, _ = self.support(ctx)
        total = 0.0
        for token in children:
            if token not in support_set:
                continue
            prob = float(dist.get(token, 0.0))
            if prob <= 0.0:
                continue
            next_ctx = self._next_context(ctx, int(token))
            total += prob * self.good_prob_mass(next_ctx, (*norm_prefix, int(token)))
        self._good_prob_cache[key] = float(total)
        return float(total)

    def _token_weights(
        self,
        context: tuple[int, ...] | list[int],
        prefix: tuple[int, ...],
    ) -> tuple[dict[int, float], dict[int, float], float]:
        ctx = self._normalize_context(context)
        support, _, dist, _ = self.support(ctx)
        weights: dict[int, float] = {}
        total_good_mass = 0.0
        for token in support:
            next_ctx = self._next_context(ctx, int(token))
            good_mass = self.good_prob_mass(next_ctx, (*prefix, int(token)))
            total_good_mass += float(dist[int(token)]) * float(good_mass)
            weights[int(token)] = float(dist[int(token)]) * (1.0 + (self.exp_beta - 1.0) * float(good_mass))
        return {int(token): float(dist[int(token)]) for token in support}, weights, float(total_good_mass)

    @staticmethod
    def _normalize(weights: dict[int, float]) -> dict[int, float]:
        total = float(sum(weights.values()) or 1.0)
        return {int(token): float(weight) / total for token, weight in weights.items() if weight > 0.0}

    def first_token_summary(self, context: tuple[int, ...] | list[int]) -> PolicyContextSummary:
        base_dist, weights, good_mass = self._token_weights(context, tuple())
        distribution = self._normalize(weights)
        _ = base_dist
        return PolicyContextSummary(
            policy_name=self.policy_name,
            active_pool=self.policy_name,
            first_token_distribution=distribution,
            desirable_mass=float(good_mass),
            fallback_indicator=1.0 if good_mass <= 0.0 else 0.0,
            all_extension_count=None,
            desirable_extension_count=None,
        )

    @staticmethod
    def _sample_from_weights(weights: dict[int, float], rng: random.Random) -> int:
        total = float(sum(weights.values()))
        if total <= 0.0:
            raise ValueError("Cannot sample from empty weights")
        draw = rng.random() * total
        acc = 0.0
        for token in sorted(weights):
            acc += float(weights[token])
            if draw <= acc:
                return int(token)
        return int(sorted(weights)[-1])

    def sample_extension(
        self,
        context: tuple[int, ...] | list[int],
        rng: random.Random,
    ) -> tuple[tuple[int, ...], DecisionMetadata]:
        first_summary = self.first_token_summary(context)
        prefix: tuple[int, ...] = tuple()
        ctx = self._normalize_context(context)
        chosen: list[int] = []
        while len(chosen) < self.span:
            _, weights, _ = self._token_weights(ctx, prefix)
            token = self._sample_from_weights(weights, rng)
            chosen.append(int(token))
            prefix = tuple(chosen)
            ctx = self._next_context(ctx, int(token))
        extension = tuple(chosen)
        chosen_category = "desirable" if extension in self.utility.desirable_set else "undesirable"
        return extension, DecisionMetadata(
            policy_name=self.policy_name,
            active_pool=self.policy_name,
            first_token_distribution=first_summary.first_token_distribution,
            desirable_mass=float(first_summary.desirable_mass),
            fallback_indicator=float(first_summary.fallback_indicator),
            chosen_category=chosen_category,
            all_extension_count=None,
            desirable_extension_count=None,
        )


def _build_policies() -> list[PolicyPreset]:
    return [
        PolicyPreset(
            name="hard_uniform_good",
            description="Current hard policy: uniformly choose among desirable reachable 5-step continuations, else uniformly among all reachable continuations.",
        ),
        PolicyPreset(
            name="soft_prob_beta_1",
            description="Probability-tilted lookahead with exp(beta * 1_good) weighting and beta=1.",
            beta=1.0,
        ),
        PolicyPreset(
            name="soft_prob_beta_3",
            description="Probability-tilted lookahead with exp(beta * 1_good) weighting and beta=3.",
            beta=3.0,
        ),
    ]


def _build_experiments(*, version: str, alpha: float, generations: int) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    index = 1
    for policy in _build_policies():
        for preset in base._build_presets():
            for scale in base._build_scales():
                experiments.append(
                    ExperimentSpec(
                        index=index,
                        preset=preset,
                        scale=scale,
                        policy=policy,
                        alpha=alpha,
                        generations=generations,
                        version=version,
                    )
                )
                index += 1
    return experiments


def _planner_for_policy(*, policy: PolicyPreset, model: Any, utility: Any) -> HardCountPlanner | SoftProbTiltPlanner:
    if policy.name == "hard_uniform_good":
        return HardCountPlanner(model=model, utility=utility)
    return SoftProbTiltPlanner(model=model, utility=utility, beta=float(policy.beta or 0.0))


def _context_counts(tokens: list[int], max_order: int) -> dict[tuple[int, ...], int]:
    return dict(base._context_counts(tokens, max_order))


def _evaluate_self_consistency(
    *,
    tokens: list[int],
    max_order: int,
    planner: HardCountPlanner | SoftProbTiltPlanner,
    generation: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = build_empirical_model(tokens, max_order)
    context_counter = _context_counts(tokens, max_order)
    total_mass = float(sum(context_counter.values()) or 1.0)
    rows: list[dict[str, Any]] = []
    weights: list[float] = []
    kl_values: list[float] = []
    js_values: list[float] = []
    tv_values: list[float] = []
    top1_values: list[float] = []
    fallback_values: list[float] = []
    desirable_mass_values: list[float] = []

    for context, count in sorted(context_counter.items()):
        base_dist, order_used = model.distribution(context)
        summary = planner.first_token_summary(context)
        agent_dist = summary.first_token_distribution
        kl_bits = base._kl_bits(agent_dist, base_dist)
        js_bits = base._js_bits(agent_dist, base_dist)
        tv = base._tv_distance(agent_dist, base_dist)
        base_top = base._argmax_token(base_dist)
        agent_top = base._argmax_token(agent_dist)
        row = {
            "generation": int(generation),
            "context": base._serialize_tokens(context),
            "context_count": int(count),
            "context_weight": float(count) / total_mass,
            "policy_name": summary.policy_name,
            "active_pool": summary.active_pool,
            "base_order_used": int(order_used),
            "base_support_size": int(len(base_dist)),
            "lookahead_support_size": int(len(agent_dist)),
            "lookahead_kl_bits": float(kl_bits),
            "lookahead_js_bits": float(js_bits),
            "lookahead_tv_distance": float(tv),
            "base_top_token": int(base_top),
            "agent_top_token": int(agent_top),
            "top1_match": 1 if int(base_top) == int(agent_top) else 0,
            "desirable_mass": float(summary.desirable_mass),
            "fallback_indicator": float(summary.fallback_indicator),
            "all_extension_count": summary.all_extension_count,
            "desirable_extension_count": summary.desirable_extension_count,
        }
        rows.append(row)
        weight = float(count)
        weights.append(weight)
        kl_values.append(float(kl_bits))
        js_values.append(float(js_bits))
        tv_values.append(float(tv))
        top1_values.append(1.0 if int(base_top) == int(agent_top) else 0.0)
        fallback_values.append(float(summary.fallback_indicator))
        desirable_mass_values.append(float(summary.desirable_mass))

    summary = {
        "lookahead_context_count": int(len(rows)),
        "lookahead_kl_bits": base._weighted_average(kl_values, weights),
        "lookahead_js_bits": base._weighted_average(js_values, weights),
        "lookahead_tv_distance": base._weighted_average(tv_values, weights),
        "lookahead_top1_match_rate": base._weighted_average(top1_values, weights),
        "lookahead_fallback_context_share": base._weighted_average(fallback_values, weights),
        "lookahead_avg_desirable_mass": base._weighted_average(desirable_mass_values, weights),
    }
    return summary, rows


def _decision_summary(decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not decision_rows:
        return {
            "agent_decisions": 0,
            "agent_desirable_choices": 0,
            "agent_neutral_choices": 0,
            "agent_undesirable_choices": 0,
            "agent_desirable_rate": 0.0,
            "agent_neutral_rate": 0.0,
            "agent_undesirable_rate": 0.0,
            "agent_avg_utility_score": 0.0,
            "agent_avg_total_score": 0.0,
            "agent_avg_model_logprob": 0.0,
            "agent_avg_search_cost": 0.0,
            "agent_tokens_published": 0,
            "agent_fallback_rate": 0.0,
            "agent_avg_desirable_mass": 0.0,
        }
    total = float(len(decision_rows))
    desirable = sum(1 for row in decision_rows if row["chosen_category"] == "desirable")
    undesirable = sum(1 for row in decision_rows if row["chosen_category"] == "undesirable")
    fallback = sum(float(row["fallback_indicator"]) for row in decision_rows) / total
    avg_mass = sum(float(row["desirable_mass"]) for row in decision_rows) / total
    tokens_published = sum(int(row["append_count"]) for row in decision_rows)
    return {
        "agent_decisions": int(len(decision_rows)),
        "agent_desirable_choices": int(desirable),
        "agent_neutral_choices": 0,
        "agent_undesirable_choices": int(undesirable),
        "agent_desirable_rate": float(desirable / total),
        "agent_neutral_rate": 0.0,
        "agent_undesirable_rate": float(undesirable / total),
        "agent_avg_utility_score": float(desirable / total),
        "agent_avg_total_score": float(desirable / total),
        "agent_avg_model_logprob": 0.0,
        "agent_avg_search_cost": 0.0,
        "agent_tokens_published": int(tokens_published),
        "agent_fallback_rate": float(fallback),
        "agent_avg_desirable_mass": float(avg_mass),
    }


def _generate_replacement(
    *,
    source_tokens: list[int],
    config: Any,
    planner: HardCountPlanner | SoftProbTiltPlanner,
    rng: random.Random,
    generation: int,
    policy_name: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    replacement_length = int(round(float(config.alpha) * int(config.text_length)))
    if replacement_length <= 0:
        return [], []
    seed_length = min(int(config.max_order), replacement_length)
    out = base.random_seed_ngram(source_tokens, seed_length, rng)
    decisions: list[dict[str, Any]] = []
    decision_index = 0
    model = build_empirical_model(source_tokens, int(config.max_order))
    while len(out) < replacement_length:
        decision_index += 1
        context = tuple(int(x) for x in out[-(int(config.max_order) - 1):]) if int(config.max_order) > 1 else tuple()
        base_dist, base_order_used = model.distribution(context)
        extension, meta = planner.sample_extension(context, rng)
        remaining = replacement_length - len(out)
        append_count = min(remaining, SPAN, len(extension))
        out.extend(int(token) for token in extension[:append_count])
        first_token = int(extension[0])
        decisions.append(
            {
                "generation": int(generation),
                "decision_index": int(decision_index),
                "policy_name": policy_name,
                "prefix_length_before": int(len(out) - append_count),
                "context": base._serialize_tokens(context),
                "base_order_used": int(base_order_used),
                "base_support_size": int(len(base_dist)),
                "base_top_token": int(base._argmax_token(base_dist)),
                "base_top_prob": float(max(base_dist.values())),
                "chosen_extension": base._serialize_tokens(extension),
                "chosen_category": meta.chosen_category,
                "chosen_first_token": int(first_token),
                "chosen_first_token_policy_prob": float(meta.first_token_distribution.get(first_token, 0.0)),
                "chosen_first_token_base_prob": float(base_dist.get(first_token, 0.0)),
                "active_pool": meta.active_pool,
                "desirable_mass": float(meta.desirable_mass),
                "fallback_indicator": float(meta.fallback_indicator),
                "all_extension_count": meta.all_extension_count,
                "desirable_extension_count": meta.desirable_extension_count,
                "append_count": int(append_count),
            }
        )
    return out[:replacement_length], decisions


def _write_run_figures(paths: base.RunPaths, rows: list[dict[str, Any]], title_suffix: str) -> list[str]:
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values("generation")
    generations = df["generation"].tolist()
    saved: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    axes[0].plot(generations, df["lookahead_kl_bits"], marker="o", linewidth=2.0)
    axes[0].plot(generations, df["lookahead_js_bits"], marker="o", linewidth=2.0)
    axes[0].set_title("KL and JS")
    axes[0].legend(["KL(bits)", "JS(bits)"])
    axes[0].grid(alpha=0.25)
    axes[1].plot(generations, df["lookahead_top1_match_rate"], marker="o", linewidth=2.0)
    axes[1].plot(generations, df["lookahead_avg_desirable_mass"], marker="o", linewidth=2.0)
    axes[1].legend(["top-1 match", "avg desirable mass"])
    axes[1].set_title("Agreement and desirable mass")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.set_xlabel("Generation")
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    path = paths.figures_dir / "soft_policy_self_consistency.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    axes[0].plot(generations, df["desirable_window_share"], marker="o", linewidth=2.0)
    axes[0].plot(generations, df["agent_desirable_rate"], marker="o", linewidth=2.0)
    axes[0].set_title("Desirable windows and chosen extensions")
    axes[0].legend(["desirable window share", "desirable choice rate"])
    axes[0].grid(alpha=0.25)
    max_order = int(df["max_order"].iloc[0])
    ratio_col = f"distinct_{max_order}grams_ratio_vs_gen0"
    axes[1].plot(generations, df[ratio_col], marker="o", linewidth=2.0)
    axes[1].plot(generations, df["vocab_ratio_vs_gen0"], marker="o", linewidth=2.0)
    axes[1].legend([f"{max_order}-gram ratio", "vocabulary ratio"])
    axes[1].set_title("Support retention")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.set_xlabel("Generation")
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    path = paths.figures_dir / "soft_policy_support_and_publication.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def _summarise_run(spec: ExperimentSpec, rows: list[dict[str, Any]], paths: base.RunPaths) -> dict[str, Any]:
    first = rows[0]
    final = rows[-1]
    min_row = min(rows, key=lambda row: float(row["lookahead_kl_bits"]))
    return {
        "experiment_index": int(spec.index),
        "version": spec.version,
        "run_name": spec.run_name,
        "policy_name": spec.policy.name,
        "policy_description": spec.policy.description,
        "policy_beta": spec.policy.beta,
        "preset_name": spec.preset.name,
        "scale_label": spec.scale.label,
        "panel_label": spec.panel_label,
        "vocab_size": int(spec.scale.vocab_size),
        "text_length": int(spec.scale.text_length),
        "alpha": float(spec.alpha),
        "generations": int(spec.generations),
        "seed": int(spec.seed),
        "run_dir": str(paths.run_dir),
        "metrics_final_path": str(paths.metrics_final_path),
        "summary_path": str(paths.summary_path),
        "decision_log_path": str(paths.tables_dir / "decision_log.csv"),
        "context_log_path": str(paths.tables_dir / "context_metrics.csv"),
        "initial_lookahead_kl_bits": float(first["lookahead_kl_bits"]),
        "final_lookahead_kl_bits": float(final["lookahead_kl_bits"]),
        "delta_lookahead_kl_bits": float(final["lookahead_kl_bits"]) - float(first["lookahead_kl_bits"]),
        "min_lookahead_kl_bits": float(min_row["lookahead_kl_bits"]),
        "generation_of_min_kl": int(min_row["generation"]),
        "final_minus_min_kl_bits": float(final["lookahead_kl_bits"]) - float(min_row["lookahead_kl_bits"]),
        "final_top1_match_rate": float(final["lookahead_top1_match_rate"]),
        "final_fallback_context_share": float(final["lookahead_fallback_context_share"]),
        "final_avg_desirable_mass": float(final["lookahead_avg_desirable_mass"]),
        "final_desirable_window_share": float(final["desirable_window_share"]),
        "final_agent_desirable_rate": float(final["agent_desirable_rate"]),
        "final_agent_fallback_rate": float(final["agent_fallback_rate"]),
    }


def _run_experiment(
    spec: ExperimentSpec,
    *,
    output_root: Path,
    force_rebuild: bool,
    progress_bar: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = base._ensure_run_paths(output_root=output_root, run_name=spec.run_name, version=spec.version)
    if force_rebuild:
        base._remove_tree_contents(paths.run_dir)
        paths = base._ensure_run_paths(output_root=output_root, run_name=spec.run_name, version=spec.version)

    urtext_config = base.UrtextConfig(mode=spec.preset.mode, vocab_size=spec.scale.vocab_size)
    latent_config = base.LatentGrammarConfig(
        max_order=spec.preset.max_order,
        order_keep_probs=spec.preset.order_keep_probs,
        exact_support=False,
        support_cache_limit=8192,
    )
    lab_config = base.RegenerationLabConfig(
        text_length=spec.scale.text_length,
        generations=spec.generations,
        alpha=spec.alpha,
        max_order=spec.preset.max_order,
        restart_probability=0.0,
        sample_retained_block=True,
    )

    bundle = base.build_urtext(urtext_config, latent_config, int(lab_config.text_length), seed=spec.seed)
    manifest = {
        "run_name": spec.run_name,
        "version": spec.version,
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "seed": int(spec.seed),
        "policy_name": spec.policy.name,
        "policy_description": spec.policy.description,
        "policy_beta": spec.policy.beta,
        "utility_basis": "previous_generation_reference",
        "lookahead_span": SPAN,
        "alpha": float(spec.alpha),
        "generations": int(spec.generations),
        "urtext_config": base._json_ready(urtext_config.__dict__),
        "latent_config": base._json_ready(latent_config.__dict__),
        "lab_config": base._json_ready(lab_config.__dict__),
        "id_to_label": base._json_ready(bundle.id_to_label),
        "notes": [
            "Follow-up exact theorem-2 batch with smaller alpha and longer runs.",
            "Utility basis is previous_generation_reference only.",
            "At generation t -> t+1, the desirable set is the distinct set of 5-grams observed in Ut.",
            "Soft probability-tilted policies weight each depth-5 continuation by P_model(extension | context) * exp(beta * 1_good(extension)).",
            "When beta=0 this reduces exactly to the one-step model marginal over the first token.",
        ],
    }
    atomic_save_json(paths.manifest_path, manifest)

    original_tokens = [int(token) for token in bundle.token_ids[:int(lab_config.text_length)]]
    current_tokens = list(original_tokens)
    metrics_rows: list[dict[str, Any]] = []
    all_decision_rows: list[dict[str, Any]] = []
    all_context_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    snapshots: dict[int, list[int]] = {0: list(current_tokens)}

    for generation in tqdm(range(0, spec.generations + 1), desc=spec.run_name, unit="gen", disable=not progress_bar):
        utility = base._build_reference_utility(
            reference_tokens=current_tokens,
            span=SPAN,
            label=f"prev_generation_reference_gen{generation}",
        )
        reference_summary = base._reference_summary(
            basis_name="previous_generation_reference",
            source_generation=generation,
            reference_tokens=current_tokens,
            span=SPAN,
        )
        planner = _planner_for_policy(
            policy=spec.policy,
            model=build_empirical_model(current_tokens, int(lab_config.max_order)),
            utility=utility,
        )
        consistency_metrics, context_rows = _evaluate_self_consistency(
            tokens=current_tokens,
            max_order=int(lab_config.max_order),
            planner=planner,
            generation=generation,
        )
        all_context_rows.extend(context_rows)
        reference_rows.append({"generation": int(generation), **reference_summary})

        if generation == 0:
            agent_stats = _decision_summary([])
            row = base.metrics_for_generation(
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
            agent_stats = _decision_summary(generation_decisions)
            row = base.metrics_for_generation(
                current_tokens,
                lab_config,
                generation=generation,
                baseline=baseline,
                utility=utility,
                agent_stats=agent_stats,
            )
        row.update(consistency_metrics)
        row["policy_name"] = spec.policy.name
        row["policy_beta"] = spec.policy.beta
        row["reference_source_generation"] = int(generation)
        row["reference_unique_grams"] = int(reference_summary["reference_unique_grams"])
        metrics_rows.append(row)

        if generation == spec.generations:
            break

        generation_rng = random.Random(spec.seed + 1009 * (generation + 1))
        generated_tokens, generation_decisions = _generate_replacement(
            source_tokens=current_tokens,
            config=lab_config,
            planner=planner,
            rng=generation_rng,
            generation=generation + 1,
            policy_name=spec.policy.name,
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
                "version": spec.version,
                "last_completed_generation": generation,
                "updated_at": timestamp(),
                "policy_name": spec.policy.name,
            },
        )

    base._write_rows_csv(paths.metrics_final_path, metrics_rows)
    base._write_rows_csv(paths.metrics_partial_path, metrics_rows)
    base._write_rows_csv(paths.tables_dir / "decision_log.csv", all_decision_rows)
    base._write_rows_csv(paths.tables_dir / "context_metrics.csv", all_context_rows)
    base._write_rows_csv(paths.tables_dir / "reference_summaries.csv", reference_rows)
    base._write_sample_texts(paths, snapshots, {int(k): str(v) for k, v in bundle.id_to_label.items()})
    figure_paths = _write_run_figures(
        paths,
        metrics_rows,
        title_suffix=f"{spec.version} | {spec.run_name} | {timestamp()}",
    )
    summary = {
        "run_name": spec.run_name,
        "version": spec.version,
        "policy_name": spec.policy.name,
        "policy_beta": spec.policy.beta,
        "finished_at": timestamp(),
        "metrics_final_path": str(paths.metrics_final_path),
        "decision_log_path": str(paths.tables_dir / "decision_log.csv"),
        "context_log_path": str(paths.tables_dir / "context_metrics.csv"),
        "reference_summary_path": str(paths.tables_dir / "reference_summaries.csv"),
        "figure_paths": figure_paths,
    }
    atomic_save_json(paths.summary_path, summary)
    atomic_save_json(paths.checkpoint_path, {"status": "finished", **summary})
    registry_row = _summarise_run(spec, metrics_rows, paths)
    return registry_row, metrics_rows


def _plot_metric_grid(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    panels: list[str],
    presets: list[base.UrtextPreset],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    axes_flat = axes.flatten()
    colors = plt.get_cmap("tab10")
    for axis, panel in zip(axes_flat, panels):
        panel_df = df[df["panel_label"] == panel]
        for idx, preset in enumerate(presets):
            preset_df = panel_df[panel_df["preset_name"] == preset.name].sort_values("generation")
            if preset_df.empty:
                continue
            axis.plot(
                preset_df["generation"],
                preset_df[metric],
                marker="o",
                linewidth=2.0,
                markersize=3.0,
                label=preset.name,
                color=colors(idx),
            )
        axis.set_title(panel.replace("|", " | "))
        axis.set_xlabel("Generation")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, registry_df: pd.DataFrame, sweep_name: str, alpha: float, generations: int) -> None:
    mean_by_policy_scale = (
        registry_df.groupby(["policy_name", "scale_label"])[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "min_lookahead_kl_bits",
                "generation_of_min_kl",
                "final_top1_match_rate",
                "final_fallback_context_share",
            ]
        ]
        .mean()
        .reset_index()
    )
    mean_by_policy = (
        registry_df.groupby("policy_name")[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "min_lookahead_kl_bits",
                "final_top1_match_rate",
                "final_fallback_context_share",
            ]
        ]
        .mean()
        .reset_index()
    )
    best_min = registry_df.nsmallest(8, "min_lookahead_kl_bits")[
        [
            "run_name",
            "policy_name",
            "preset_name",
            "scale_label",
            "initial_lookahead_kl_bits",
            "min_lookahead_kl_bits",
            "generation_of_min_kl",
            "final_lookahead_kl_bits",
        ]
    ]
    lines = [
        f"# {sweep_name}",
        "",
        f"- created_at: {timestamp()}",
        f"- experiments: {len(registry_df)}",
        f"- alpha: {alpha}",
        f"- generations: {generations}",
        "- utility basis: previous_generation_reference",
        "- question: do softer exact publication policies reduce the KL between the 5-lookahead first-token law and the one-step n-gram law?",
        "",
        "## Best minimum KL(bits) achieved during the run",
        base._df_to_markdown(best_min),
        "",
        "## Mean metrics by policy and scale",
        base._df_to_markdown(mean_by_policy_scale),
        "",
        "## Mean metrics by policy",
        base._df_to_markdown(mean_by_policy),
        "",
    ]
    atomic_save_text(path, "\n".join(lines) + "\n")


def _generate_outputs(
    *,
    output_root: Path,
    version: str,
    alpha: float,
    generations: int,
    registry_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sweep_name = f"theorem2_soft_policy_followup_{len(registry_rows)}runs_a{str(alpha).replace('.', 'p')}_g{generations}"
    sweep_dir = ensure_dir(output_root / "sweeps" / stable_slug(f"{version}_{sweep_name}"))
    figures_dir = ensure_dir(sweep_dir / "figures")
    tables_dir = ensure_dir(sweep_dir / "tables")

    registry_df = pd.DataFrame(registry_rows).sort_values("experiment_index").reset_index(drop=True)
    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["experiment_index", "generation"]).reset_index(drop=True)
    registry_path = sweep_dir / "experiment_registry.csv"
    trajectory_path = sweep_dir / "trajectory_metrics.csv"
    registry_df.to_csv(registry_path, index=False)
    trajectory_df.to_csv(trajectory_path, index=False)

    summary_by_policy = (
        registry_df.groupby("policy_name")[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "min_lookahead_kl_bits",
                "generation_of_min_kl",
                "final_top1_match_rate",
                "final_fallback_context_share",
            ]
        ]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    summary_by_policy.columns = ["_".join(col).strip("_") for col in summary_by_policy.columns]
    summary_by_policy.to_csv(tables_dir / "summary_by_policy.csv", index=False)

    summary_by_policy_scale = (
        registry_df.groupby(["policy_name", "scale_label"])[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "min_lookahead_kl_bits",
                "generation_of_min_kl",
                "final_top1_match_rate",
            ]
        ]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    summary_by_policy_scale.columns = ["_".join(col).strip("_") for col in summary_by_policy_scale.columns]
    summary_by_policy_scale.to_csv(tables_dir / "summary_by_policy_scale.csv", index=False)

    panels = sorted(registry_df["panel_label"].unique())
    presets = base._build_presets()
    _plot_metric_grid(
        trajectory_df,
        metric="lookahead_kl_bits",
        ylabel="KL(bits)",
        title="Soft-policy theorem-2 follow-up: exact KL trajectories",
        path=figures_dir / "trajectory_kl_bits.png",
        panels=panels,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="lookahead_top1_match_rate",
        ylabel="Top-1 match",
        title="Soft-policy theorem-2 follow-up: top-1 agreement",
        path=figures_dir / "trajectory_top1_match_rate.png",
        panels=panels,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="desirable_window_share",
        ylabel="Desirable-window share",
        title="Soft-policy theorem-2 follow-up: desirable-window trajectories",
        path=figures_dir / "trajectory_desirable_window_share.png",
        panels=panels,
        presets=presets,
    )

    final_pivot = registry_df.pivot(index="preset_name", columns="panel_label", values="final_lookahead_kl_bits")
    min_pivot = registry_df.pivot(index="preset_name", columns="panel_label", values="min_lookahead_kl_bits")
    delta_pivot = registry_df.pivot(index="preset_name", columns="panel_label", values="delta_lookahead_kl_bits")
    final_pivot.to_csv(tables_dir / "final_kl_pivot.csv")
    min_pivot.to_csv(tables_dir / "min_kl_pivot.csv")
    delta_pivot.to_csv(tables_dir / "delta_kl_pivot.csv")
    base._plot_heatmap(
        final_pivot,
        title="Final exact KL(bits)",
        cmap="YlOrRd",
        value_format=".4f",
        path=figures_dir / "final_kl_heatmap.png",
    )
    base._plot_heatmap(
        min_pivot,
        title="Minimum exact KL(bits) reached during run",
        cmap="YlGnBu",
        value_format=".4f",
        path=figures_dir / "min_kl_heatmap.png",
    )
    base._plot_heatmap(
        delta_pivot,
        title="Delta exact KL(bits): final - initial",
        cmap="coolwarm",
        value_format=".4f",
        path=figures_dir / "delta_kl_heatmap.png",
    )

    report_path = sweep_dir / "report.md"
    _write_report(report_path, registry_df, sweep_name, alpha, generations)
    manifest = {
        "sweep_name": sweep_name,
        "version": version,
        "created_at": timestamp(),
        "registry_path": str(registry_path),
        "trajectory_path": str(trajectory_path),
        "report_path": str(report_path),
        "figure_paths": [
            str(figures_dir / "trajectory_kl_bits.png"),
            str(figures_dir / "trajectory_top1_match_rate.png"),
            str(figures_dir / "trajectory_desirable_window_share.png"),
            str(figures_dir / "final_kl_heatmap.png"),
            str(figures_dir / "min_kl_heatmap.png"),
            str(figures_dir / "delta_kl_heatmap.png"),
        ],
        "table_paths": [
            str(tables_dir / "summary_by_policy.csv"),
            str(tables_dir / "summary_by_policy_scale.csv"),
            str(tables_dir / "final_kl_pivot.csv"),
            str(tables_dir / "min_kl_pivot.csv"),
            str(tables_dir / "delta_kl_pivot.csv"),
        ],
    }
    atomic_save_json(sweep_dir / "sweep_manifest.json", manifest)
    return manifest


def main() -> int:
    root = Path(".").resolve()
    version = "V0_01"
    alpha = 0.10
    generations = 40
    output_root = ensure_dir(root / "GitHub" / "data" / "outputs" / "theorem2_soft_policy_followup")

    experiments = _build_experiments(version=version, alpha=alpha, generations=generations)
    registry_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    for spec in tqdm(experiments, desc="Soft-policy follow-up", unit="run"):
        registry_row, rows = _run_experiment(
            spec,
            output_root=output_root,
            force_rebuild=True,
            progress_bar=False,
        )
        registry_rows.append(registry_row)
        for row in rows:
            enriched = dict(row)
            enriched.update(
                {
                    "experiment_index": int(spec.index),
                    "run_name": spec.run_name,
                    "policy_name": spec.policy.name,
                    "preset_name": spec.preset.name,
                    "scale_label": spec.scale.label,
                    "panel_label": spec.panel_label,
                }
            )
            trajectory_rows.append(enriched)

    manifest = _generate_outputs(
        output_root=output_root,
        version=version,
        alpha=alpha,
        generations=generations,
        registry_rows=registry_rows,
        trajectory_rows=trajectory_rows,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
