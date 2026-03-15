#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
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

from drift_selection.checkpoints import atomic_save_json, atomic_save_text, load_pickle_checkpoint, save_pickle_checkpoint  # noqa: E402
from drift_selection.ngram_regeneration_lab import (  # noqa: E402
    LatentGrammarConfig,
    ReferenceRGramUtility,
    RegenerationLabConfig,
    UrtextConfig,
    build_empirical_model,
    build_reference_rgram_utility,
    build_urtext,
    metrics_for_generation,
    random_seed_ngram,
    sample_retained_tokens,
)
from drift_selection.utils import ensure_dir, stable_slug, timestamp  # noqa: E402


SPAN = 5


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
class UtilityBasisPreset:
    name: str
    description: str

    @property
    def short_label(self) -> str:
        return "urtext" if self.name == "urtext_reference" else "previous"


@dataclass(frozen=True)
class ExperimentSpec:
    index: int
    preset: UrtextPreset
    scale: ScalePreset
    utility_basis: UtilityBasisPreset
    alpha: float
    generations: int
    version: str

    @property
    def run_name(self) -> str:
        alpha_tag = str(self.alpha).replace(".", "p")
        return (
            f"theorem2_exact5_{self.utility_basis.short_label}_"
            f"{self.preset.name}_{self.scale.label}_"
            f"o{self.preset.max_order}_g{self.generations}_a{alpha_tag}"
        )

    @property
    def scale_basis_label(self) -> str:
        return f"{self.utility_basis.short_label}|{self.scale.label}"

    @property
    def seed(self) -> int:
        raw = (
            f"{self.version}:{self.index}:{self.utility_basis.name}:{self.preset.name}:"
            f"{self.scale.label}:{self.alpha}:{self.generations}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


@dataclass(frozen=True)
class RunPaths:
    output_root: Path
    run_dir: Path
    state_dir: Path
    logs_dir: Path
    figures_dir: Path
    tables_dir: Path
    manifest_path: Path
    checkpoint_path: Path
    metrics_partial_path: Path
    metrics_final_path: Path
    summary_path: Path
    sample_text_path: Path
    decision_log_path: Path
    context_log_path: Path
    reference_summary_path: Path


@dataclass(frozen=True)
class FirstTokenPolicy:
    active_pool: str
    total_extensions: int
    desirable_extensions: int
    active_counts: dict[int, int]
    all_counts: dict[int, int]
    desirable_counts: dict[int, int]

    @property
    def active_distribution(self) -> dict[int, float]:
        total = float(self.total_extensions or 1)
        return {int(token): float(count) / total for token, count in self.active_counts.items() if count > 0}


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
        ScalePreset(vocab_size=200, text_length=1000),
    ]


def _build_utility_bases() -> list[UtilityBasisPreset]:
    return [
        UtilityBasisPreset(
            name="urtext_reference",
            description="Desirable 5-grams are all distinct 5-grams observed in the original urtext U0.",
        ),
        UtilityBasisPreset(
            name="previous_generation_reference",
            description="At generation t -> t+1, desirable 5-grams are all distinct 5-grams observed in the current corpus Ut.",
        ),
    ]


def _build_experiments(*, version: str, alpha: float, generations: int) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    index = 1
    for basis in _build_utility_bases():
        for preset in _build_presets():
            for scale in _build_scales():
                experiments.append(
                    ExperimentSpec(
                        index=index,
                        preset=preset,
                        scale=scale,
                        utility_basis=basis,
                        alpha=alpha,
                        generations=generations,
                        version=version,
                    )
                )
                index += 1
    return experiments


def _remove_tree_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, list):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    return obj


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)


def _load_rows_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _snapshot_path(state_dir: Path, generation: int) -> Path:
    return state_dir / f"tokens_gen_{generation:03d}.pkl"


def _save_generation_snapshot(state_dir: Path, generation: int, tokens: list[int]) -> None:
    save_pickle_checkpoint(_snapshot_path(state_dir, generation), [int(token) for token in tokens])


def _load_generation_snapshot(state_dir: Path, generation: int) -> list[int]:
    return [int(token) for token in load_pickle_checkpoint(_snapshot_path(state_dir, generation))]


def _serialize_tokens(tokens: tuple[int, ...] | list[int]) -> str:
    if not tokens:
        return "<BOS>"
    return " ".join(str(int(token)) for token in tokens)


def _gram_signature(grams: list[tuple[int, ...]]) -> str:
    payload = "\n".join(_serialize_tokens(gram) for gram in sorted(set(grams)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ngrams(tokens: list[int], span: int) -> list[tuple[int, ...]]:
    if len(tokens) < span:
        return []
    return [tuple(int(x) for x in tokens[i:i + span]) for i in range(len(tokens) - span + 1)]


def _reference_summary(
    *,
    basis_name: str,
    source_generation: int,
    reference_tokens: list[int],
    span: int,
) -> dict[str, Any]:
    grams = _ngrams(reference_tokens, span)
    unique_grams = sorted(set(grams))
    return {
        "basis_name": basis_name,
        "source_generation": int(source_generation),
        "span": int(span),
        "reference_window_total": int(len(grams)),
        "reference_unique_grams": int(len(unique_grams)),
        "reference_signature": _gram_signature(unique_grams) if unique_grams else "",
    }


def _build_reference_utility(
    *,
    reference_tokens: list[int],
    span: int,
    label: str,
) -> ReferenceRGramUtility:
    return build_reference_rgram_utility(
        reference_tokens,
        span=span,
        min_count=1,
        label=label,
        unseen_category="undesirable",
    )


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
    midpoint = {token: 0.5 * float(q.get(token, 0.0) + p.get(token, 0.0)) for token in keys}
    return 0.5 * _kl_bits(q, midpoint) + 0.5 * _kl_bits(p, midpoint)


def _tv_distance(q: dict[int, float], p: dict[int, float]) -> float:
    keys = set(q) | set(p)
    return 0.5 * sum(abs(float(q.get(token, 0.0)) - float(p.get(token, 0.0))) for token in keys)


def _argmax_token(dist: dict[int, float]) -> int:
    return min(
        (int(token) for token, prob in dist.items() if prob == max(dist.values())),
        default=0,
    )


def _weighted_average(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


class ExactLookaheadPlanner:
    def __init__(self, *, model: Any, desirable_grams: list[tuple[int, ...]], span: int):
        self.model = model
        self.span = int(span)
        self.context_len = max(0, int(model.order) - 1)
        self._support_cache: dict[tuple[int, ...], tuple[tuple[int, ...], frozenset[int], dict[int, float], int]] = {}
        self._all_cache: dict[tuple[tuple[int, ...], int], int] = {}
        self._good_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        prefix_children: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for gram in sorted(set(tuple(int(x) for x in gram) for gram in desirable_grams)):
            if len(gram) != self.span:
                raise ValueError("All desirable grams must have length equal to span")
            for prefix_len in range(self.span):
                prefix_children[gram[:prefix_len]].add(int(gram[prefix_len]))
        self.prefix_children = {prefix: tuple(sorted(children)) for prefix, children in prefix_children.items()}

    def _normalize_context(self, context: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        if self.context_len <= 0:
            return tuple()
        return tuple(int(x) for x in context[-self.context_len:])

    def _next_context(self, context: tuple[int, ...], token: int) -> tuple[int, ...]:
        if self.context_len <= 0:
            return tuple()
        return self._normalize_context((*context, int(token)))

    def support(self, context: tuple[int, ...] | list[int]) -> tuple[tuple[int, ...], frozenset[int], dict[int, float], int]:
        ctx = self._normalize_context(context)
        cached = self._support_cache.get(ctx)
        if cached is not None:
            return cached
        base_dist, order_used = self.model.distribution(ctx)
        support = tuple(sorted(int(token) for token, prob in base_dist.items() if float(prob) > 0.0))
        normalized_dist = {int(token): float(prob) for token, prob in base_dist.items() if float(prob) > 0.0}
        cached = (support, frozenset(support), normalized_dist, int(order_used))
        self._support_cache[ctx] = cached
        return cached

    def all_count(self, context: tuple[int, ...] | list[int], remaining: int) -> int:
        ctx = self._normalize_context(context)
        key = (ctx, int(remaining))
        if key in self._all_cache:
            return self._all_cache[key]
        if remaining <= 0:
            return 1
        support, _, _, _ = self.support(ctx)
        total = 0
        for token in support:
            total += self.all_count(self._next_context(ctx, token), remaining - 1)
        self._all_cache[key] = int(total)
        return int(total)

    def good_count(self, context: tuple[int, ...] | list[int], prefix: tuple[int, ...]) -> int:
        ctx = self._normalize_context(context)
        norm_prefix = tuple(int(x) for x in prefix)
        key = (ctx, norm_prefix)
        if key in self._good_cache:
            return self._good_cache[key]
        if len(norm_prefix) >= self.span:
            return 1
        children = self.prefix_children.get(norm_prefix, ())
        if not children:
            self._good_cache[key] = 0
            return 0
        _, support_set, _, _ = self.support(ctx)
        total = 0
        for token in children:
            if token not in support_set:
                continue
            total += self.good_count(self._next_context(ctx, token), (*norm_prefix, int(token)))
        self._good_cache[key] = int(total)
        return int(total)

    def first_token_policy(self, context: tuple[int, ...] | list[int]) -> FirstTokenPolicy:
        ctx = self._normalize_context(context)
        support, support_set, _, _ = self.support(ctx)
        all_counts: dict[int, int] = {}
        for token in support:
            all_counts[int(token)] = int(self.all_count(self._next_context(ctx, int(token)), self.span - 1))
        desirable_counts: dict[int, int] = {}
        for token in self.prefix_children.get(tuple(), ()):
            if token not in support_set:
                continue
            count = int(self.good_count(self._next_context(ctx, int(token)), (int(token),)))
            if count > 0:
                desirable_counts[int(token)] = count
        desirable_total = int(sum(desirable_counts.values()))
        if desirable_total > 0:
            active_counts = dict(desirable_counts)
            active_pool = "desirable"
            total_extensions = desirable_total
        else:
            active_counts = dict(all_counts)
            active_pool = "fallback_all"
            total_extensions = int(sum(all_counts.values()))
        return FirstTokenPolicy(
            active_pool=active_pool,
            total_extensions=int(total_extensions),
            desirable_extensions=int(desirable_total),
            active_counts=active_counts,
            all_counts=all_counts,
            desirable_counts=desirable_counts,
        )

    def _sample_from_counts(self, counts: dict[int, int], rng: random.Random) -> int:
        total = int(sum(counts.values()))
        if total <= 0:
            raise ValueError("Cannot sample from an empty continuation pool")
        draw = rng.randrange(total)
        acc = 0
        for token in sorted(counts):
            acc += int(counts[token])
            if draw < acc:
                return int(token)
        return int(sorted(counts)[-1])

    def _sample_all_suffix(
        self,
        context: tuple[int, ...],
        prefix: tuple[int, ...],
        rng: random.Random,
    ) -> tuple[int, ...]:
        if len(prefix) >= self.span:
            return prefix
        support, _, _, _ = self.support(context)
        counts: dict[int, int] = {}
        for token in support:
            counts[int(token)] = int(self.all_count(self._next_context(context, int(token)), self.span - len(prefix) - 1))
        chosen = self._sample_from_counts(counts, rng)
        next_context = self._next_context(context, chosen)
        return self._sample_all_suffix(next_context, (*prefix, chosen), rng)

    def _sample_good_suffix(
        self,
        context: tuple[int, ...],
        prefix: tuple[int, ...],
        rng: random.Random,
    ) -> tuple[int, ...]:
        if len(prefix) >= self.span:
            return prefix
        _, support_set, _, _ = self.support(context)
        counts: dict[int, int] = {}
        for token in self.prefix_children.get(prefix, ()):
            if token not in support_set:
                continue
            counts[int(token)] = int(self.good_count(self._next_context(context, int(token)), (*prefix, int(token))))
        chosen = self._sample_from_counts(counts, rng)
        next_context = self._next_context(context, chosen)
        return self._sample_good_suffix(next_context, (*prefix, chosen), rng)

    def sample_extension(self, context: tuple[int, ...] | list[int], rng: random.Random) -> tuple[tuple[int, ...], FirstTokenPolicy]:
        ctx = self._normalize_context(context)
        policy = self.first_token_policy(ctx)
        if policy.active_pool == "desirable":
            extension = self._sample_good_suffix(ctx, tuple(), rng)
        else:
            extension = self._sample_all_suffix(ctx, tuple(), rng)
        return extension, policy


def _context_counts(tokens: list[int], max_order: int) -> Counter[tuple[int, ...]]:
    context_len = max(0, int(max_order) - 1)
    if context_len <= 0:
        return Counter({tuple(): 1})
    if len(tokens) <= context_len:
        return Counter({tuple(tokens[-context_len:]): 1})
    counts: Counter[tuple[int, ...]] = Counter()
    for idx in range(context_len, len(tokens)):
        counts[tuple(int(x) for x in tokens[idx - context_len:idx])] += 1
    return counts


def _evaluate_self_consistency(
    *,
    tokens: list[int],
    max_order: int,
    planner: ExactLookaheadPlanner,
    generation: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = build_empirical_model(tokens, max_order)
    context_counter = _context_counts(tokens, max_order)
    total_context_mass = float(sum(context_counter.values()) or 1.0)
    rows: list[dict[str, Any]] = []
    kl_values: list[float] = []
    js_values: list[float] = []
    tv_values: list[float] = []
    top1_values: list[float] = []
    fallback_values: list[float] = []
    desirable_values: list[float] = []
    desirable_fraction_values: list[float] = []
    weights: list[float] = []

    for context, count in sorted(context_counter.items()):
        base_dist, order_used = model.distribution(context)
        policy = planner.first_token_policy(context)
        agent_dist = policy.active_distribution
        kl_bits = _kl_bits(agent_dist, base_dist)
        js_bits = _js_bits(agent_dist, base_dist)
        tv = _tv_distance(agent_dist, base_dist)
        base_top = _argmax_token(base_dist)
        agent_top = _argmax_token(agent_dist)
        desirable_fraction = (
            float(policy.desirable_extensions) / float(sum(policy.all_counts.values()) or 1)
        )
        row = {
            "generation": int(generation),
            "context": _serialize_tokens(context),
            "context_count": int(count),
            "context_weight": float(count) / total_context_mass,
            "base_order_used": int(order_used),
            "base_support_size": int(len(base_dist)),
            "lookahead_support_size": int(len(agent_dist)),
            "all_extension_count": int(sum(policy.all_counts.values())),
            "desirable_extension_count": int(policy.desirable_extensions),
            "desirable_extension_fraction": float(desirable_fraction),
            "active_pool": policy.active_pool,
            "lookahead_kl_bits": float(kl_bits),
            "lookahead_js_bits": float(js_bits),
            "lookahead_tv_distance": float(tv),
            "base_top_token": int(base_top),
            "agent_top_token": int(agent_top),
            "top1_match": 1 if int(base_top) == int(agent_top) else 0,
        }
        rows.append(row)
        weight = float(count)
        weights.append(weight)
        kl_values.append(float(kl_bits))
        js_values.append(float(js_bits))
        tv_values.append(float(tv))
        top1_values.append(1.0 if int(base_top) == int(agent_top) else 0.0)
        fallback_values.append(1.0 if policy.active_pool == "fallback_all" else 0.0)
        desirable_values.append(1.0 if policy.desirable_extensions > 0 else 0.0)
        desirable_fraction_values.append(float(desirable_fraction))

    summary = {
        "lookahead_context_count": int(len(rows)),
        "lookahead_kl_bits": _weighted_average(kl_values, weights),
        "lookahead_js_bits": _weighted_average(js_values, weights),
        "lookahead_tv_distance": _weighted_average(tv_values, weights),
        "lookahead_top1_match_rate": _weighted_average(top1_values, weights),
        "lookahead_fallback_context_share": _weighted_average(fallback_values, weights),
        "lookahead_desirable_context_share": _weighted_average(desirable_values, weights),
        "lookahead_avg_desirable_extension_fraction": _weighted_average(desirable_fraction_values, weights),
    }
    return summary, rows


def _decision_summary(decision_rows: list[dict[str, Any]], span: int) -> dict[str, Any]:
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
            "agent_avg_desirable_extension_fraction": 0.0,
        }
    total = float(len(decision_rows))
    desirable_choices = sum(1 for row in decision_rows if row["active_pool"] == "desirable")
    fallback_choices = sum(1 for row in decision_rows if row["active_pool"] == "fallback_all")
    fractions = [float(row["desirable_extension_fraction"]) for row in decision_rows]
    published_tokens = sum(int(row.get("append_count", span)) for row in decision_rows)
    return {
        "agent_decisions": int(len(decision_rows)),
        "agent_desirable_choices": int(desirable_choices),
        "agent_neutral_choices": 0,
        "agent_undesirable_choices": int(fallback_choices),
        "agent_desirable_rate": float(desirable_choices / total),
        "agent_neutral_rate": 0.0,
        "agent_undesirable_rate": float(fallback_choices / total),
        "agent_avg_utility_score": float(desirable_choices / total),
        "agent_avg_total_score": float(desirable_choices / total),
        "agent_avg_model_logprob": 0.0,
        "agent_avg_search_cost": 0.0,
        "agent_tokens_published": int(published_tokens),
        "agent_fallback_rate": float(fallback_choices / total),
        "agent_avg_desirable_extension_fraction": float(sum(fractions) / total),
    }


def _generate_exact_replacement(
    *,
    source_tokens: list[int],
    config: RegenerationLabConfig,
    planner: ExactLookaheadPlanner,
    rng: random.Random,
    generation: int,
    reference_summary: dict[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    replacement_length = int(round(float(config.alpha) * int(config.text_length)))
    if replacement_length <= 0:
        return [], []
    seed_length = min(int(config.max_order), replacement_length)
    out = random_seed_ngram(source_tokens, seed_length, rng)
    decisions: list[dict[str, Any]] = []
    decision_index = 0
    model = build_empirical_model(source_tokens, int(config.max_order))
    while len(out) < replacement_length:
        decision_index += 1
        context = tuple(int(x) for x in out[-(int(config.max_order) - 1):]) if int(config.max_order) > 1 else tuple()
        base_dist, base_order_used = model.distribution(context)
        extension, policy = planner.sample_extension(context, rng)
        remaining = replacement_length - len(out)
        append_count = min(remaining, SPAN, len(extension))
        out.extend(int(token) for token in extension[:append_count])
        first_token = int(extension[0])
        active_total = float(sum(policy.active_counts.values()) or 1.0)
        all_total = float(sum(policy.all_counts.values()) or 1.0)
        decision_rows = {
            "generation": int(generation),
            "decision_index": int(decision_index),
            "reference_basis": reference_summary["basis_name"],
            "reference_source_generation": int(reference_summary["source_generation"]),
            "reference_unique_grams": int(reference_summary["reference_unique_grams"]),
            "reference_signature": reference_summary["reference_signature"],
            "prefix_length_before": int(len(out) - append_count),
            "context": _serialize_tokens(context),
            "base_order_used": int(base_order_used),
            "base_support_size": int(len(base_dist)),
            "base_top_token": int(_argmax_token(base_dist)),
            "base_top_prob": float(max(base_dist.values())),
            "all_extension_count": int(sum(policy.all_counts.values())),
            "desirable_extension_count": int(policy.desirable_extensions),
            "desirable_extension_fraction": float(policy.desirable_extensions / all_total),
            "active_pool": policy.active_pool,
            "chosen_extension": _serialize_tokens(extension),
            "chosen_first_token": int(first_token),
            "chosen_first_token_active_prob": float(policy.active_counts.get(first_token, 0) / active_total),
            "chosen_first_token_all_prob": float(policy.all_counts.get(first_token, 0) / all_total),
            "chosen_first_token_base_prob": float(base_dist.get(first_token, 0.0)),
            "append_count": int(append_count),
        }
        decisions.append(decision_rows)
    return out[:replacement_length], decisions


def _combine_generation(
    *,
    previous_tokens: list[int],
    generated_tokens: list[int],
    config: RegenerationLabConfig,
    rng: random.Random,
) -> list[int]:
    retain_length = int(round((1.0 - float(config.alpha)) * int(config.text_length)))
    retained = sample_retained_tokens(previous_tokens, retain_length, rng, contiguous=bool(config.sample_retained_block))
    combined = retained + generated_tokens
    return [int(token) for token in combined[:int(config.text_length)]]


def _ensure_run_paths(*, output_root: Path, run_name: str, version: str) -> RunPaths:
    run_dir = ensure_dir(output_root / "runs" / stable_slug(f"{version}_{run_name}"))
    return RunPaths(
        output_root=output_root,
        run_dir=run_dir,
        state_dir=ensure_dir(run_dir / "state"),
        logs_dir=ensure_dir(run_dir / "logs"),
        figures_dir=ensure_dir(run_dir / "figures"),
        tables_dir=ensure_dir(run_dir / "tables"),
        manifest_path=run_dir / "run_manifest.json",
        checkpoint_path=run_dir / "checkpoint_state.json",
        metrics_partial_path=run_dir / "metrics_partial.csv",
        metrics_final_path=run_dir / "metrics_final.csv",
        summary_path=run_dir / "run_summary.json",
        sample_text_path=run_dir / "sample_texts.txt",
        decision_log_path=run_dir / "tables" / "decision_log.csv",
        context_log_path=run_dir / "tables" / "context_metrics.csv",
        reference_summary_path=run_dir / "tables" / "reference_summaries.csv",
    )


def _load_or_create_manifest(
    *,
    paths: RunPaths,
    spec: ExperimentSpec,
    urtext_config: UrtextConfig,
    latent_config: LatentGrammarConfig,
    lab_config: RegenerationLabConfig,
    id_to_label: dict[int, str],
    reference_policy_description: str,
    resume: bool,
) -> dict[str, Any]:
    expected = {
        "run_name": spec.run_name,
        "version": spec.version,
        "seed": int(spec.seed),
        "utility_basis": spec.utility_basis.name,
        "urtext_config": _json_ready(urtext_config.__dict__),
        "latent_config": _json_ready(latent_config.__dict__),
        "lab_config": _json_ready(lab_config.__dict__),
    }
    if resume and paths.manifest_path.exists():
        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                raise ValueError(
                    f"Existing manifest mismatch for {key} in {paths.manifest_path}. "
                    "Use --force-rebuild or change the version/run name."
                )
        return manifest
    manifest = {
        **expected,
        "experiment_index": int(spec.index),
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "status": "running",
        "id_to_label": _json_ready(id_to_label),
        "policy_description": reference_policy_description,
        "lookahead_span": SPAN,
        "replacement_length": int(round(float(lab_config.alpha) * int(lab_config.text_length))),
        "retained_length": int(round((1.0 - float(lab_config.alpha)) * int(lab_config.text_length))),
        "rng_scheme": {
            "run_seed": int(spec.seed),
            "generation_seed_rule": "seed + 1009 * generation",
            "replacement_sampling": "same generation RNG stream is used for retained-block sampling and exact continuation sampling",
        },
        "notes": [
            "Exact exhaustive 5-lookahead is computed over all reachable depth-5 continuations under the empirical n-gram model.",
            "If at least one reachable continuation belongs to the desirable 5-gram set, the publication agent samples uniformly over those desirable continuations.",
            "If none are desirable, the publication agent falls back to a uniform sample over all reachable depth-5 continuations.",
            "The zero-shot baseline for KL/JS/TV is the ordinary one-step empirical n-gram next-token distribution with maximal available context.",
            "For the 'previous_generation_reference' basis, the generation t -> t+1 policy uses the distinct 5-grams from Ut as its desirable set.",
        ],
    }
    atomic_save_json(paths.manifest_path, manifest)
    return manifest


def _decode_tokens(tokens: list[int], id_to_label: dict[int, str], limit: int = 80) -> str:
    return " ".join(id_to_label.get(int(token), str(int(token))) for token in tokens[:limit])


def _write_sample_texts(paths: RunPaths, snapshots: dict[int, list[int]], id_to_label: dict[int, str]) -> None:
    generations = sorted(snapshots)
    picks: list[int] = []
    if generations:
        picks.append(generations[0])
        if len(generations) > 2:
            picks.append(generations[len(generations) // 2])
        if len(generations) > 1:
            picks.append(generations[-1])
    ordered = []
    for generation in picks:
        if generation not in ordered:
            ordered.append(generation)
    lines: list[str] = []
    for generation in ordered:
        lines.append(f"[generation {generation}]")
        lines.append(_decode_tokens(snapshots[generation], id_to_label))
        lines.append("")
    atomic_save_text(paths.sample_text_path, "\n".join(lines))


def _write_run_figures(paths: RunPaths, rows: list[dict[str, Any]], title_suffix: str) -> list[str]:
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values("generation")
    generations = df["generation"].tolist()
    saved: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    axes[0].plot(generations, df["lookahead_kl_bits"], marker="o", linewidth=2.0)
    axes[0].plot(generations, df["lookahead_js_bits"], marker="o", linewidth=2.0)
    axes[0].set_title("Exact self-consistency")
    axes[0].legend(["KL(bits)", "JS(bits)"])
    axes[0].set_xlabel("Generation")
    axes[0].grid(alpha=0.25)
    axes[1].plot(generations, df["lookahead_fallback_context_share"], marker="o", linewidth=2.0)
    axes[1].plot(generations, df["lookahead_top1_match_rate"], marker="o", linewidth=2.0)
    axes[1].set_title("Fallback and top-1 match")
    axes[1].legend(["fallback share", "top-1 match"])
    axes[1].set_xlabel("Generation")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    path = paths.figures_dir / "exact_self_consistency.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    axes[0].plot(generations, df["desirable_window_share"], marker="o", linewidth=2.0)
    axes[0].plot(generations, df["agent_desirable_rate"], marker="o", linewidth=2.0)
    axes[0].set_title("Desired windows and decisions")
    axes[0].legend(["desirable window share", "desirable decision rate"])
    axes[0].set_xlabel("Generation")
    axes[0].grid(alpha=0.25)
    max_order = int(df["max_order"].iloc[0])
    axes[1].plot(generations, df[f"distinct_{max_order}grams_ratio_vs_gen0"], marker="o", linewidth=2.0)
    axes[1].plot(generations, df["vocab_ratio_vs_gen0"], marker="o", linewidth=2.0)
    axes[1].set_title("Support retention")
    axes[1].legend([f"{max_order}-gram ratio", "vocabulary ratio"])
    axes[1].set_xlabel("Generation")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    path = paths.figures_dir / "support_and_publication.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def _summarize_run(
    spec: ExperimentSpec,
    rows: list[dict[str, Any]],
    paths: RunPaths,
) -> dict[str, Any]:
    first = rows[0]
    final = rows[-1]
    max_order = int(spec.preset.max_order)
    return {
        "experiment_index": int(spec.index),
        "version": spec.version,
        "run_name": spec.run_name,
        "preset_name": spec.preset.name,
        "preset_description": spec.preset.description,
        "utility_basis": spec.utility_basis.name,
        "utility_basis_description": spec.utility_basis.description,
        "scale_label": spec.scale.label,
        "scale_basis_label": spec.scale_basis_label,
        "vocab_size": int(spec.scale.vocab_size),
        "text_length": int(spec.scale.text_length),
        "alpha": float(spec.alpha),
        "generations": int(spec.generations),
        "max_order": int(spec.preset.max_order),
        "seed": int(spec.seed),
        "run_dir": str(paths.run_dir),
        "metrics_final_path": str(paths.metrics_final_path),
        "summary_path": str(paths.summary_path),
        "decision_log_path": str(paths.decision_log_path),
        "context_log_path": str(paths.context_log_path),
        "final_generation": int(final["generation"]),
        "initial_lookahead_kl_bits": float(first["lookahead_kl_bits"]),
        "final_lookahead_kl_bits": float(final["lookahead_kl_bits"]),
        "delta_lookahead_kl_bits": float(final["lookahead_kl_bits"]) - float(first["lookahead_kl_bits"]),
        "initial_lookahead_js_bits": float(first["lookahead_js_bits"]),
        "final_lookahead_js_bits": float(final["lookahead_js_bits"]),
        "initial_top1_match_rate": float(first["lookahead_top1_match_rate"]),
        "final_top1_match_rate": float(final["lookahead_top1_match_rate"]),
        "final_fallback_context_share": float(final["lookahead_fallback_context_share"]),
        "final_desirable_context_share": float(final["lookahead_desirable_context_share"]),
        "final_desirable_window_share": float(final["desirable_window_share"]),
        "final_agent_desirable_rate": float(final["agent_desirable_rate"]),
        "final_agent_fallback_rate": float(final["agent_fallback_rate"]),
        "final_max_order_ratio": float(final[f"distinct_{max_order}grams_ratio_vs_gen0"]),
        "final_vocab_ratio": float(final["vocab_ratio_vs_gen0"]),
    }


def _df_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
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
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _plot_metric_grid(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    panels: list[str],
    presets: list[UrtextPreset],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()
    colors = plt.get_cmap("tab10")
    for axis, panel in zip(axes_flat, panels):
        panel_df = df[df["scale_basis_label"] == panel]
        for idx, preset in enumerate(presets):
            preset_df = panel_df[panel_df["preset_name"] == preset.name].sort_values("generation")
            if preset_df.empty:
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
        axis.set_title(panel.replace("|", " | "))
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
    fig, ax = plt.subplots(figsize=(11, 5.2))
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


def _run_exact_experiment(
    spec: ExperimentSpec,
    *,
    output_root: Path,
    force_rebuild: bool,
    progress_bar: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _ensure_run_paths(output_root=output_root, run_name=spec.run_name, version=spec.version)
    if force_rebuild:
        _remove_tree_contents(paths.run_dir)
        paths = _ensure_run_paths(output_root=output_root, run_name=spec.run_name, version=spec.version)

    urtext_config = UrtextConfig(mode=spec.preset.mode, vocab_size=spec.scale.vocab_size)
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
    bundle = build_urtext(urtext_config, latent_config, int(lab_config.text_length), seed=spec.seed)
    manifest = _load_or_create_manifest(
        paths=paths,
        spec=spec,
        urtext_config=urtext_config,
        latent_config=latent_config,
        lab_config=lab_config,
        id_to_label=bundle.id_to_label,
        reference_policy_description=spec.utility_basis.description,
        resume=not force_rebuild,
    )
    id_to_label = {int(k): str(v) for k, v in manifest["id_to_label"].items()}

    metrics_rows: list[dict[str, Any]] = []
    current_tokens: list[int] = []
    start_generation = 0
    snapshots: dict[int, list[int]] = {}

    if not force_rebuild and paths.checkpoint_path.exists() and paths.metrics_partial_path.exists():
        checkpoint = json.loads(paths.checkpoint_path.read_text(encoding="utf-8"))
        last_completed_generation = int(checkpoint.get("last_completed_generation", -1))
        if last_completed_generation >= 0:
            current_tokens = _load_generation_snapshot(paths.state_dir, last_completed_generation)
            metrics_rows = _load_rows_csv(paths.metrics_partial_path)
            start_generation = last_completed_generation + 1
            for generation in {0, last_completed_generation, spec.generations}:
                snap_path = _snapshot_path(paths.state_dir, generation)
                if snap_path.exists():
                    snapshots[generation] = _load_generation_snapshot(paths.state_dir, generation)

    original_tokens = [int(token) for token in bundle.token_ids[:int(lab_config.text_length)]]
    original_reference_utility = _build_reference_utility(
        reference_tokens=original_tokens,
        span=SPAN,
        label="urtext_reference",
    )

    if not current_tokens:
        current_tokens = list(original_tokens)
        baseline_reference_summary = _reference_summary(
            basis_name=spec.utility_basis.name,
            source_generation=0,
            reference_tokens=original_tokens,
            span=SPAN,
        )
        baseline_planner = ExactLookaheadPlanner(
            model=build_empirical_model(current_tokens, int(lab_config.max_order)),
            desirable_grams=list(original_reference_utility.desirable_set),
            span=SPAN,
        )
        baseline_metrics, context_rows = _evaluate_self_consistency(
            tokens=current_tokens,
            max_order=int(lab_config.max_order),
            planner=baseline_planner,
            generation=0,
        )
        base_row = metrics_for_generation(
            current_tokens,
            lab_config,
            generation=0,
            baseline=None,
            utility=original_reference_utility,
            agent_stats=_decision_summary([], SPAN),
        )
        base_row.update(baseline_metrics)
        base_row["generation_reference_basis"] = spec.utility_basis.name
        base_row["generation_reference_source_generation"] = 0
        base_row["generation_reference_unique_grams"] = int(baseline_reference_summary["reference_unique_grams"])
        base_row["operator_reference_basis"] = spec.utility_basis.name
        base_row["operator_reference_source_generation"] = 0
        base_row["operator_reference_unique_grams"] = int(baseline_reference_summary["reference_unique_grams"])
        metrics_rows = [base_row]
        _save_generation_snapshot(paths.state_dir, 0, current_tokens)
        _write_rows_csv(paths.logs_dir / "context_metrics_gen_000.csv", context_rows)
        _write_rows_csv(paths.logs_dir / "decision_log_gen_000.csv", [])
        _write_rows_csv(paths.reference_summary_path, [
            {
                "generation": 0,
                "role": "generation_reference",
                **baseline_reference_summary,
            },
            {
                "generation": 0,
                "role": "operator_reference",
                **baseline_reference_summary,
            },
        ])
        snapshots[0] = list(current_tokens)
        manifest["updated_at"] = timestamp()
        manifest["status"] = "running"
        atomic_save_json(paths.manifest_path, manifest)
        atomic_save_json(
            paths.checkpoint_path,
            {
                "run_name": spec.run_name,
                "version": spec.version,
                "last_completed_generation": 0,
                "status": "running",
                "updated_at": manifest["updated_at"],
            },
        )
        _write_rows_csv(paths.metrics_partial_path, metrics_rows)
        start_generation = 1

    if start_generation > spec.generations:
        final_registry = _summarize_run(spec, metrics_rows, paths)
        return final_registry, metrics_rows

    iterator = range(start_generation, spec.generations + 1)
    iterator = tqdm(iterator, desc=f"exact5:{spec.run_name}", unit="gen", disable=not progress_bar)

    for generation in iterator:
        generation_rng = random.Random(spec.seed + 1009 * generation)
        if spec.utility_basis.name == "urtext_reference":
            generation_reference_tokens = list(original_tokens)
            generation_reference_source = 0
        else:
            generation_reference_tokens = list(current_tokens)
            generation_reference_source = generation - 1
        generation_reference_utility = _build_reference_utility(
            reference_tokens=generation_reference_tokens,
            span=SPAN,
            label=f"{spec.utility_basis.name}_gen{generation_reference_source}",
        )
        generation_reference_summary = _reference_summary(
            basis_name=spec.utility_basis.name,
            source_generation=generation_reference_source,
            reference_tokens=generation_reference_tokens,
            span=SPAN,
        )
        generation_model = build_empirical_model(current_tokens, int(lab_config.max_order))
        generation_planner = ExactLookaheadPlanner(
            model=generation_model,
            desirable_grams=list(generation_reference_utility.desirable_set),
            span=SPAN,
        )
        generated_tokens, decision_rows = _generate_exact_replacement(
            source_tokens=current_tokens,
            config=lab_config,
            planner=generation_planner,
            rng=generation_rng,
            generation=generation,
            reference_summary=generation_reference_summary,
        )
        next_tokens = _combine_generation(
            previous_tokens=current_tokens,
            generated_tokens=generated_tokens,
            config=lab_config,
            rng=generation_rng,
        )
        if spec.utility_basis.name == "urtext_reference":
            operator_reference_tokens = list(original_tokens)
            operator_reference_source = 0
        else:
            operator_reference_tokens = list(next_tokens)
            operator_reference_source = generation
        operator_reference_utility = _build_reference_utility(
            reference_tokens=operator_reference_tokens,
            span=SPAN,
            label=f"{spec.utility_basis.name}_operator_gen{operator_reference_source}",
        )
        operator_reference_summary = _reference_summary(
            basis_name=spec.utility_basis.name,
            source_generation=operator_reference_source,
            reference_tokens=operator_reference_tokens,
            span=SPAN,
        )
        operator_planner = ExactLookaheadPlanner(
            model=build_empirical_model(next_tokens, int(lab_config.max_order)),
            desirable_grams=list(operator_reference_utility.desirable_set),
            span=SPAN,
        )
        consistency_metrics, context_rows = _evaluate_self_consistency(
            tokens=next_tokens,
            max_order=int(lab_config.max_order),
            planner=operator_planner,
            generation=generation,
        )
        baseline = {
            key: float(metrics_rows[0][key])
            for key in [f"distinct_{order}grams" for order in range(1, int(lab_config.max_order) + 1)]
        }
        row = metrics_for_generation(
            next_tokens,
            lab_config,
            generation=generation,
            baseline=baseline,
            utility=generation_reference_utility,
            agent_stats=_decision_summary(decision_rows, SPAN),
        )
        row.update(consistency_metrics)
        row["generation_reference_basis"] = spec.utility_basis.name
        row["generation_reference_source_generation"] = int(generation_reference_source)
        row["generation_reference_unique_grams"] = int(generation_reference_summary["reference_unique_grams"])
        row["operator_reference_basis"] = spec.utility_basis.name
        row["operator_reference_source_generation"] = int(operator_reference_source)
        row["operator_reference_unique_grams"] = int(operator_reference_summary["reference_unique_grams"])
        metrics_rows = [existing for existing in metrics_rows if int(existing["generation"]) != generation]
        metrics_rows.append(row)
        metrics_rows.sort(key=lambda item: int(item["generation"]))

        _save_generation_snapshot(paths.state_dir, generation, next_tokens)
        _write_rows_csv(paths.logs_dir / f"decision_log_gen_{generation:03d}.csv", decision_rows)
        _write_rows_csv(paths.logs_dir / f"context_metrics_gen_{generation:03d}.csv", context_rows)

        reference_rows = _load_rows_csv(paths.reference_summary_path)
        reference_rows = [
            row for row in reference_rows
            if not (
                int(row["generation"]) == generation
                and row["role"] in {"generation_reference", "operator_reference"}
            )
        ]
        reference_rows.extend(
            [
                {
                    "generation": int(generation),
                    "role": "generation_reference",
                    **generation_reference_summary,
                },
                {
                    "generation": int(generation),
                    "role": "operator_reference",
                    **operator_reference_summary,
                },
            ]
        )
        reference_rows.sort(key=lambda item: (int(item["generation"]), str(item["role"])))
        _write_rows_csv(paths.reference_summary_path, reference_rows)

        current_tokens = list(next_tokens)
        if generation in {0, spec.generations // 2, spec.generations}:
            snapshots[generation] = list(current_tokens)
        _write_rows_csv(paths.metrics_partial_path, metrics_rows)
        manifest["updated_at"] = timestamp()
        manifest["status"] = "running"
        atomic_save_json(paths.manifest_path, manifest)
        atomic_save_json(
            paths.checkpoint_path,
            {
                "run_name": spec.run_name,
                "version": spec.version,
                "last_completed_generation": int(generation),
                "status": "running",
                "updated_at": manifest["updated_at"],
            },
        )
        postfix = {
            "kl": f"{float(row['lookahead_kl_bits']):.4f}",
            "fallback": f"{float(row['lookahead_fallback_context_share']):.3f}",
            "desirable": f"{float(row['agent_desirable_rate']):.3f}",
        }
        iterator.set_postfix(postfix)

    manifest["updated_at"] = timestamp()
    manifest["status"] = "finished"
    atomic_save_json(paths.manifest_path, manifest)
    atomic_save_json(
        paths.checkpoint_path,
        {
            "run_name": spec.run_name,
            "version": spec.version,
            "last_completed_generation": int(spec.generations),
            "status": "finished",
            "updated_at": manifest["updated_at"],
        },
    )
    _write_rows_csv(paths.metrics_final_path, metrics_rows)

    all_decision_rows: list[dict[str, Any]] = []
    all_context_rows: list[dict[str, Any]] = []
    for generation in range(0, spec.generations + 1):
        decision_path = paths.logs_dir / f"decision_log_gen_{generation:03d}.csv"
        context_path = paths.logs_dir / f"context_metrics_gen_{generation:03d}.csv"
        if decision_path.exists():
            all_decision_rows.extend(_load_rows_csv(decision_path))
        if context_path.exists():
            all_context_rows.extend(_load_rows_csv(context_path))
        snap_path = _snapshot_path(paths.state_dir, generation)
        if snap_path.exists() and generation in {0, spec.generations // 2, spec.generations}:
            snapshots[generation] = _load_generation_snapshot(paths.state_dir, generation)
    _write_rows_csv(paths.decision_log_path, all_decision_rows)
    _write_rows_csv(paths.context_log_path, all_context_rows)
    _write_sample_texts(paths, snapshots, id_to_label)
    figure_paths = _write_run_figures(
        paths,
        metrics_rows,
        title_suffix=f"{spec.version} | {spec.run_name} | {manifest['updated_at']}",
    )
    summary = {
        "run_name": spec.run_name,
        "version": spec.version,
        "finished_at": manifest["updated_at"],
        "utility_basis": spec.utility_basis.name,
        "seed": int(spec.seed),
        "metrics_final_path": str(paths.metrics_final_path),
        "decision_log_path": str(paths.decision_log_path),
        "context_log_path": str(paths.context_log_path),
        "reference_summary_path": str(paths.reference_summary_path),
        "sample_text_path": str(paths.sample_text_path),
        "figure_paths": figure_paths,
    }
    atomic_save_json(paths.summary_path, summary)
    final_registry = _summarize_run(spec, metrics_rows, paths)
    return final_registry, metrics_rows


def _write_sweep_report(
    *,
    path: Path,
    registry_df: pd.DataFrame,
    sweep_name: str,
    alpha: float,
    generations: int,
) -> None:
    best_final = registry_df.nsmallest(6, "final_lookahead_kl_bits")[
        [
            "run_name",
            "utility_basis",
            "preset_name",
            "scale_label",
            "final_lookahead_kl_bits",
            "final_top1_match_rate",
        ]
    ]
    best_reduction = registry_df.nsmallest(6, "delta_lookahead_kl_bits")[
        [
            "run_name",
            "utility_basis",
            "preset_name",
            "scale_label",
            "initial_lookahead_kl_bits",
            "final_lookahead_kl_bits",
            "delta_lookahead_kl_bits",
        ]
    ]
    worst_final = registry_df.nlargest(6, "final_lookahead_kl_bits")[
        [
            "run_name",
            "utility_basis",
            "preset_name",
            "scale_label",
            "final_lookahead_kl_bits",
            "final_fallback_context_share",
        ]
    ]
    mean_by_basis_scale = (
        registry_df.groupby(["utility_basis", "scale_label"])[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "final_top1_match_rate",
                "final_fallback_context_share",
                "final_desirable_window_share",
            ]
        ]
        .mean()
        .reset_index()
    )
    mean_by_basis_preset = (
        registry_df.groupby(["utility_basis", "preset_name"])[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "final_top1_match_rate",
                "final_fallback_context_share",
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
        f"- exact lookahead span: {SPAN}",
        "- scales: focused exact sweep over M=1000 with V in {100, 200}",
        "- policy: enumerate all reachable depth-5 continuations; sample uniformly from desirable continuations if any exist, otherwise uniformly from all reachable continuations",
        "- baseline distribution: ordinary one-step empirical n-gram next-token law with maximal available context",
        "",
        "## Lowest final exact KL(bits)",
        _df_to_markdown(best_final),
        "",
        "## Largest exact KL(bits) reductions",
        _df_to_markdown(best_reduction),
        "",
        "## Highest final exact KL(bits)",
        _df_to_markdown(worst_final),
        "",
        "## Mean final metrics by utility basis and scale",
        _df_to_markdown(mean_by_basis_scale),
        "",
        "## Mean final metrics by utility basis and urtext preset",
        _df_to_markdown(mean_by_basis_preset),
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
    sweep_name = f"theorem2_exact5_sweep_{len(registry_rows)}runs_a{str(alpha).replace('.', 'p')}_g{generations}"
    sweep_dir = ensure_dir(output_root / "sweeps" / stable_slug(f"{version}_{sweep_name}"))
    figures_dir = ensure_dir(sweep_dir / "figures")
    tables_dir = ensure_dir(sweep_dir / "tables")

    registry_df = pd.DataFrame(registry_rows).sort_values("experiment_index").reset_index(drop=True)
    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(
        ["experiment_index", "generation"]
    ).reset_index(drop=True)

    registry_path = sweep_dir / "experiment_registry.csv"
    trajectory_path = sweep_dir / "trajectory_metrics.csv"
    registry_df.to_csv(registry_path, index=False)
    trajectory_df.to_csv(trajectory_path, index=False)

    summary_by_basis_scale = (
        registry_df.groupby(["utility_basis", "scale_label"])[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "final_top1_match_rate",
                "final_fallback_context_share",
                "final_desirable_window_share",
                "final_agent_desirable_rate",
            ]
        ]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    summary_by_basis_scale.columns = ["_".join(col).strip("_") for col in summary_by_basis_scale.columns]
    summary_by_basis_scale.to_csv(tables_dir / "summary_by_basis_scale.csv", index=False)

    summary_by_basis_preset = (
        registry_df.groupby(["utility_basis", "preset_name"])[
            [
                "initial_lookahead_kl_bits",
                "final_lookahead_kl_bits",
                "delta_lookahead_kl_bits",
                "final_top1_match_rate",
                "final_fallback_context_share",
            ]
        ]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    summary_by_basis_preset.columns = ["_".join(col).strip("_") for col in summary_by_basis_preset.columns]
    summary_by_basis_preset.to_csv(tables_dir / "summary_by_basis_preset.csv", index=False)

    panels = sorted(registry_df["scale_basis_label"].unique())
    presets = _build_presets()
    _plot_metric_grid(
        trajectory_df,
        metric="lookahead_kl_bits",
        ylabel="Exact KL(bits)",
        title="Exact theorem-2 self-consistency trajectories",
        path=figures_dir / "trajectory_exact_kl_bits.png",
        panels=panels,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="lookahead_fallback_context_share",
        ylabel="Fallback-context share",
        title="Exact theorem-2 fallback-context trajectories",
        path=figures_dir / "trajectory_fallback_context_share.png",
        panels=panels,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="agent_desirable_rate",
        ylabel="Desirable-publication rate",
        title="Exact theorem-2 desirable-publication trajectories",
        path=figures_dir / "trajectory_agent_desirable_rate.png",
        panels=panels,
        presets=presets,
    )
    _plot_metric_grid(
        trajectory_df,
        metric="lookahead_top1_match_rate",
        ylabel="Top-1 match rate",
        title="Exact theorem-2 top-1 agreement trajectories",
        path=figures_dir / "trajectory_top1_match_rate.png",
        panels=panels,
        presets=presets,
    )

    kl_pivot = registry_df.pivot(index="preset_name", columns="scale_basis_label", values="final_lookahead_kl_bits")
    delta_pivot = registry_df.pivot(index="preset_name", columns="scale_basis_label", values="delta_lookahead_kl_bits")
    fallback_pivot = registry_df.pivot(index="preset_name", columns="scale_basis_label", values="final_fallback_context_share")
    top1_pivot = registry_df.pivot(index="preset_name", columns="scale_basis_label", values="final_top1_match_rate")
    desirable_pivot = registry_df.pivot(index="preset_name", columns="scale_basis_label", values="final_desirable_window_share")
    kl_pivot.to_csv(tables_dir / "final_exact_kl_bits_pivot.csv")
    delta_pivot.to_csv(tables_dir / "delta_exact_kl_bits_pivot.csv")
    fallback_pivot.to_csv(tables_dir / "final_fallback_context_share_pivot.csv")
    top1_pivot.to_csv(tables_dir / "final_top1_match_rate_pivot.csv")
    desirable_pivot.to_csv(tables_dir / "final_desirable_window_share_pivot.csv")

    _plot_heatmap(
        kl_pivot,
        title="Final exact KL(bits) between lookahead and one-step laws",
        cmap="YlOrRd",
        value_format=".4f",
        path=figures_dir / "final_exact_kl_bits_heatmap.png",
    )
    _plot_heatmap(
        delta_pivot,
        title="Delta exact KL(bits): final - initial",
        cmap="coolwarm",
        value_format=".4f",
        path=figures_dir / "delta_exact_kl_bits_heatmap.png",
    )
    _plot_heatmap(
        fallback_pivot,
        title="Final fallback-context share",
        cmap="Blues",
        value_format=".3f",
        path=figures_dir / "final_fallback_context_share_heatmap.png",
    )
    _plot_heatmap(
        top1_pivot,
        title="Final top-1 agreement rate",
        cmap="YlGnBu",
        value_format=".3f",
        path=figures_dir / "final_top1_match_rate_heatmap.png",
    )
    _plot_heatmap(
        desirable_pivot,
        title="Final desirable-window share under generation utility",
        cmap="YlGn",
        value_format=".3f",
        path=figures_dir / "final_desirable_window_share_heatmap.png",
    )

    report_path = sweep_dir / "report.md"
    _write_sweep_report(
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
        "registry_path": str(registry_path),
        "trajectory_path": str(trajectory_path),
        "report_path": str(report_path),
        "figure_paths": [
            str(figures_dir / "trajectory_exact_kl_bits.png"),
            str(figures_dir / "trajectory_fallback_context_share.png"),
            str(figures_dir / "trajectory_agent_desirable_rate.png"),
            str(figures_dir / "trajectory_top1_match_rate.png"),
            str(figures_dir / "final_exact_kl_bits_heatmap.png"),
            str(figures_dir / "delta_exact_kl_bits_heatmap.png"),
            str(figures_dir / "final_fallback_context_share_heatmap.png"),
            str(figures_dir / "final_top1_match_rate_heatmap.png"),
            str(figures_dir / "final_desirable_window_share_heatmap.png"),
        ],
        "table_paths": [
            str(tables_dir / "summary_by_basis_scale.csv"),
            str(tables_dir / "summary_by_basis_preset.csv"),
            str(tables_dir / "final_exact_kl_bits_pivot.csv"),
            str(tables_dir / "delta_exact_kl_bits_pivot.csv"),
            str(tables_dir / "final_fallback_context_share_pivot.csv"),
            str(tables_dir / "final_top1_match_rate_pivot.csv"),
            str(tables_dir / "final_desirable_window_share_pivot.csv"),
        ],
    }
    atomic_save_json(sweep_dir / "sweep_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an exact exhaustive theorem-2 5-lookahead sweep.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", default="V0_01")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--run-progress", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = ensure_dir(root / "GitHub" / "data" / "outputs" / "theorem2_exact_5lookahead")
    experiments = _build_experiments(version=args.version, alpha=args.alpha, generations=args.generations)
    if args.limit is not None:
        experiments = experiments[: args.limit]

    registry_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for spec in tqdm(experiments, desc="Exact theorem-2 sweep", unit="run"):
        registry_row, rows = _run_exact_experiment(
            spec,
            output_root=output_root,
            force_rebuild=args.force_rebuild,
            progress_bar=args.run_progress,
        )
        registry_rows.append(registry_row)
        for row in rows:
            enriched = dict(row)
            enriched.update(
                {
                    "experiment_index": int(spec.index),
                    "run_name": spec.run_name,
                    "preset_name": spec.preset.name,
                    "scale_label": spec.scale.label,
                    "scale_basis_label": spec.scale_basis_label,
                    "utility_basis": spec.utility_basis.name,
                    "seed": int(spec.seed),
                }
            )
            trajectory_rows.append(enriched)

    manifest = _generate_outputs(
        output_root=output_root,
        version=args.version,
        alpha=args.alpha,
        generations=args.generations,
        registry_rows=registry_rows,
        trajectory_rows=trajectory_rows,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
