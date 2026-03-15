#!/usr/bin/env python3
"""Build exact theorem-2 information-theoretic tutorial figures.

This script generates a matched descriptive/normative pair of exact block-law
recursions and exports the figures used in the tutorial appendix.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "GitHub" / "data" / "outputs" / "theorem2_information_tutorial_case"
APPENDIX_FIG_DIR = ROOT / "GitHub" / "appendix" / "figures" / "generated"

FIG_COMPARE_STEM = "figure_appx_theorem2_info_diagnostics_compare"
FIG_GAP_STEM = "figure_appx_theorem2_info_normative_gap_structure"


# Matched exact configuration used in the tutorial.
V = 5
N = 3
R = 5
ALPHA_REPLACE = 1.0
BETA_MIX = 0.8
ITERATIONS = 40
SEED = 1
SEED_INIT_MODE = "random_support_uniform"
SEED_SUPPORT_SIZE = 25
INITIAL_PUBLIC_R_MODE = "induced_from_seed_q"
NORMATIVE_TARGET_MODE = "random_support_dirichlet"
ROLLOUT_MODE = "exact"
EXACT_MAX_STATES = 50_000


def format_gram(gram: tuple[int, ...]) -> str:
    return "".join(str(x) for x in gram)


def format_compact_gram_label(label: str) -> str:
    return "-".join(label)


def normalize_distribution(dist: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    total = float(sum(dist.values()))
    if total <= 0:
        raise ValueError("Distribution has non-positive total mass.")
    return {k: float(v / total) for k, v in dist.items() if v > 0}


def l1_distance(q: dict[tuple[int, ...], float], p: dict[tuple[int, ...], float]) -> float:
    keys = set(q) | set(p)
    return float(sum(abs(q.get(k, 0.0) - p.get(k, 0.0)) for k in keys))


def kl_divergence(q: dict[tuple[int, ...], float], p: dict[tuple[int, ...], float]) -> float:
    kl = 0.0
    for key, qv in q.items():
        if qv <= 0:
            continue
        pv = p.get(key, 0.0)
        if pv <= 0:
            return float("inf")
        kl += qv * math.log2(qv / pv)
    if kl < 0 and abs(kl) < 1e-14:
        kl = 0.0
    return float(kl)


def entropy(dist: dict[tuple[int, ...], float]) -> float:
    return float(-sum(v * math.log2(v) for v in dist.values() if v > 0))


def sample_unique_ngrams(vocab_size: int, gram_len: int, support_size: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    total = vocab_size**gram_len
    if support_size > total:
        raise ValueError(f"support_size={support_size} exceeds V^L={total}.")
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < support_size:
        chosen.add(tuple(rng.integers(0, vocab_size, size=gram_len).tolist()))
    return list(chosen)


def initialize_distribution(
    vocab_size: int,
    gram_len: int,
    init_mode: str,
    support_size: int = 50,
    text_length: int = 5_000,
    seed: int = 0,
) -> dict[tuple[int, ...], float]:
    rng = np.random.default_rng(seed)

    if init_mode == "random_support_uniform":
        grams = sample_unique_ngrams(vocab_size, gram_len, support_size, rng)
        return {gram: 1.0 / len(grams) for gram in grams}

    if init_mode == "random_support_dirichlet":
        grams = sample_unique_ngrams(vocab_size, gram_len, support_size, rng)
        weights = rng.dirichlet(np.ones(len(grams)))
        return {gram: float(weight) for gram, weight in zip(grams, weights)}

    if init_mode == "random_text":
        if text_length < gram_len:
            raise ValueError("text_length must be at least gram_len.")
        text = rng.integers(0, vocab_size, size=text_length).tolist()
        counts: Counter[tuple[int, ...]] = Counter()
        for idx in range(text_length - gram_len + 1):
            counts[tuple(text[idx : idx + gram_len])] += 1
        return normalize_distribution(dict(counts))

    raise ValueError(f"Unknown init_mode: {init_mode}")


def induced_window_marginals(dist: dict[tuple[int, ...], float], gram_len: int) -> list[dict[tuple[int, ...], float]]:
    marginals: list[dict[tuple[int, ...], float]] = [{(): 1.0}]
    for k in range(1, gram_len + 1):
        counts: defaultdict[tuple[int, ...], float] = defaultdict(float)
        windows_per_block = gram_len - k + 1
        for gram, prob in dist.items():
            weight = prob / windows_per_block
            for start in range(windows_per_block):
                counts[gram[start : start + k]] += weight
        marginals.append(normalize_distribution(dict(counts)))
    return marginals


def next_token_row(
    prefix: tuple[int, ...],
    marginals: list[dict[tuple[int, ...], float]],
    vocab_size: int,
    order: int,
    row_cache: dict[tuple[int, ...], tuple[dict[int, float], int, tuple[int, ...]]] | None = None,
) -> tuple[dict[int, float], int, tuple[int, ...]]:
    if row_cache is not None and prefix in row_cache:
        return row_cache[prefix]

    max_len = min(len(prefix), order - 1)
    for ctx_len in range(max_len, -1, -1):
        ctx = prefix[-ctx_len:] if ctx_len > 0 else ()
        ctx_mass = marginals[ctx_len].get(ctx, 0.0) if ctx_len > 0 else 1.0
        if ctx_mass <= 0:
            continue

        row: dict[int, float] = {}
        total = 0.0
        for token in range(vocab_size):
            extended = ctx + (token,)
            ext_mass = marginals[ctx_len + 1].get(extended, 0.0)
            if ext_mass > 0:
                prob = ext_mass / ctx_mass
                row[token] = prob
                total += prob

        if total > 0:
            inv_total = 1.0 / total
            for token in list(row.keys()):
                row[token] *= inv_total
            result = (row, ctx_len, ctx)
            if row_cache is not None:
                row_cache[prefix] = result
            return result

    fallback = ({token: 1.0 / vocab_size for token in range(vocab_size)}, 0, ())
    if row_cache is not None:
        row_cache[prefix] = fallback
    return fallback


def exact_rollout_from_law(
    dist: dict[tuple[int, ...], float],
    vocab_size: int,
    order: int,
    out_length: int,
    max_states: int = EXACT_MAX_STATES,
) -> dict[tuple[int, ...], float]:
    marginals = induced_window_marginals(dist, order)
    frontier: dict[tuple[int, ...], float] = {(): 1.0}
    row_cache: dict[tuple[int, ...], tuple[dict[int, float], int, tuple[int, ...]]] = {}

    for _ in range(out_length):
        new_frontier: defaultdict[tuple[int, ...], float] = defaultdict(float)
        for prefix, prefix_prob in frontier.items():
            row, _, _ = next_token_row(prefix, marginals, vocab_size, order, row_cache=row_cache)
            for token, token_prob in row.items():
                new_frontier[prefix + (token,)] += prefix_prob * token_prob
        frontier = dict(new_frontier)
        if len(frontier) > max_states:
            raise RuntimeError(f"Exact rollout exceeded max_states={max_states}.")

    return normalize_distribution(frontier)


def project_to_n_from_r(r_dist: dict[tuple[int, ...], float], order_n: int) -> dict[tuple[int, ...], float]:
    if not r_dist:
        raise ValueError("Cannot project an empty distribution.")
    r_local = len(next(iter(r_dist)))
    if order_n > r_local:
        raise ValueError(f"Cannot project to n={order_n} from r={r_local}.")
    return induced_window_marginals(r_dist, r_local)[order_n]


def mix_distributions(*weighted_terms: tuple[float, dict[tuple[int, ...], float]]) -> dict[tuple[int, ...], float]:
    counts: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for weight, dist in weighted_terms:
        if weight == 0:
            continue
        for gram, prob in dist.items():
            counts[gram] += weight * prob
    return normalize_distribution(dict(counts))


def prepare_target_r(
    vocab_size: int,
    block_len: int,
    target_mode: str,
    seed_q: dict[tuple[int, ...], float],
    seed_rho: dict[tuple[int, ...], float],
    support_size: int,
    text_length: int,
    seed: int,
) -> tuple[dict[tuple[int, ...], float], str]:
    if target_mode == "seed_induced_from_q":
        target_r = exact_rollout_from_law(seed_q, vocab_size, len(next(iter(seed_q))), block_len)
        return target_r, "seed_induced_from_q_exact"
    if target_mode == "initial_public_r":
        return dict(seed_rho), "initial_public_r"
    if target_mode in {"random_support_uniform", "random_support_dirichlet", "random_text"}:
        return (
            initialize_distribution(
                vocab_size,
                block_len,
                init_mode=target_mode,
                support_size=support_size,
                text_length=text_length,
                seed=seed,
            ),
            target_mode,
        )
    raise ValueError(f"Unknown target_mode: {target_mode}")


def apply_r_agent(
    lookahead_mode: str,
    rho_t: dict[tuple[int, ...], float],
    q_t: dict[tuple[int, ...], float],
    target_r: dict[tuple[int, ...], float] | None = None,
) -> tuple[dict[tuple[int, ...], float], str]:
    if lookahead_mode == "descriptive_prev_r_direct":
        return dict(rho_t), "direct_prev_r"
    if lookahead_mode == "normative_fixed_direct":
        if target_r is None:
            raise ValueError("target_r is required for normative_fixed_direct.")
        return dict(target_r), "normative_direct"
    raise ValueError(f"Unsupported lookahead_mode: {lookahead_mode}")


def iterate_theorem2_map(
    *,
    seed_q: dict[tuple[int, ...], float],
    lookahead_mode: str,
    beta_mix: float,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    if INITIAL_PUBLIC_R_MODE != "induced_from_seed_q":
        raise ValueError("This script expects induced_from_seed_q initial public law.")

    rho0 = exact_rollout_from_law(seed_q, V, N, R)
    target_r = None
    target_info = None
    if lookahead_mode == "normative_fixed_direct":
        target_r, target_info = prepare_target_r(
            vocab_size=V,
            block_len=R,
            target_mode=NORMATIVE_TARGET_MODE,
            seed_q=seed_q,
            seed_rho=rho0,
            support_size=30,
            text_length=4_000,
            seed=seed + 123,
        )

    rho_t = normalize_distribution(rho0)
    q_t = project_to_n_from_r(rho_t, N)

    history_rows: list[dict[str, object]] = []
    states: list[dict[str, object]] = [{"generation": 0, "rho_r": rho_t, "q_n": q_t}]

    for generation in range(iterations):
        generated_from_n = exact_rollout_from_law(q_t, V, N, R)
        strong_gap_l1 = l1_distance(rho_t, generated_from_n)
        strong_gap_kl = kl_divergence(rho_t, generated_from_n)

        mismatch_keys = set(rho_t) | set(generated_from_n)
        worst_key = max(mismatch_keys, key=lambda gram: abs(rho_t.get(gram, 0.0) - generated_from_n.get(gram, 0.0)))
        worst_abs = abs(rho_t.get(worst_key, 0.0) - generated_from_n.get(worst_key, 0.0))

        r_agent, r_agent_mode = apply_r_agent(
            lookahead_mode=lookahead_mode,
            rho_t=rho_t,
            q_t=q_t,
            target_r=target_r,
        )

        mixed_new = mix_distributions((1.0 - beta_mix, generated_from_n), (beta_mix, r_agent))
        rho_next = mix_distributions((1.0 - ALPHA_REPLACE, rho_t), (ALPHA_REPLACE, mixed_new))
        q_next = project_to_n_from_r(rho_next, N)

        r_step_l1 = l1_distance(rho_next, rho_t)
        n_step_l1 = l1_distance(q_next, q_t)

        if lookahead_mode == "descriptive_prev_r_direct" and ALPHA_REPLACE * (1.0 - beta_mix) > 0:
            implied_strong = r_step_l1 / (ALPHA_REPLACE * (1.0 - beta_mix))
            identity_error = abs(implied_strong - strong_gap_l1)
        else:
            implied_strong = float("nan")
            identity_error = float("nan")

        history_rows.append(
            {
                "generation": generation,
                "support_r": len(rho_t),
                "entropy_r": entropy(rho_t),
                "support_n": len(q_t),
                "entropy_n": entropy(q_t),
                "strong_gap_l1": strong_gap_l1,
                "strong_gap_kl": strong_gap_kl,
                "max_rgram_abs_diff": worst_abs,
                "top_mismatch_rgram": format_gram(worst_key),
                "r_step_l1": r_step_l1,
                "n_step_l1": n_step_l1,
                "implied_strong_from_step": implied_strong,
                "identity_relation_error": identity_error,
                "r_agent_mode": r_agent_mode,
            }
        )

        rho_t = rho_next
        q_t = q_next
        states.append(
            {
                "generation": generation + 1,
                "rho_r": rho_t,
                "q_n": q_t,
                "generated_from_n": generated_from_n,
            }
        )

    history = pd.DataFrame(history_rows)
    return {
        "history": history,
        "states": states,
        "target_r": target_r,
        "target_info": target_info,
    }


def top_mismatch_table(
    public_r: dict[tuple[int, ...], float],
    generated_r: dict[tuple[int, ...], float],
    top_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for gram in set(public_r) | set(generated_r):
        public_prob = public_r.get(gram, 0.0)
        generated_prob = generated_r.get(gram, 0.0)
        rows.append(
            {
                "rgram": format_gram(gram),
                "public_prob": public_prob,
                "generated_prob": generated_prob,
                "signed_diff": public_prob - generated_prob,
                "abs_diff": abs(public_prob - generated_prob),
            }
        )
    full_df = pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)
    return full_df.head(top_k), full_df


def prefix_conditional_diagnostics(
    public_r: dict[tuple[int, ...], float],
    generated_r: dict[tuple[int, ...], float],
    prefix_len: int,
    top_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    public_mass: defaultdict[tuple[int, ...], float] = defaultdict(float)
    generated_mass: defaultdict[tuple[int, ...], float] = defaultdict(float)
    public_suffix: defaultdict[tuple[int, ...], defaultdict[tuple[int, ...], float]] = defaultdict(lambda: defaultdict(float))
    generated_suffix: defaultdict[tuple[int, ...], defaultdict[tuple[int, ...], float]] = defaultdict(lambda: defaultdict(float))

    for gram, prob in public_r.items():
        prefix = gram[:prefix_len]
        suffix = gram[prefix_len:]
        public_mass[prefix] += prob
        public_suffix[prefix][suffix] += prob

    for gram, prob in generated_r.items():
        prefix = gram[:prefix_len]
        suffix = gram[prefix_len:]
        generated_mass[prefix] += prob
        generated_suffix[prefix][suffix] += prob

    rows: list[dict[str, object]] = []
    for prefix in set(public_mass) | set(generated_mass):
        public_prefix = public_mass.get(prefix, 0.0)
        generated_prefix = generated_mass.get(prefix, 0.0)

        prefix_l1_contribution = 0.0
        for suffix in set(public_suffix[prefix]) | set(generated_suffix[prefix]):
            prefix_l1_contribution += abs(
                public_suffix[prefix].get(suffix, 0.0) - generated_suffix[prefix].get(suffix, 0.0)
            )

        if public_prefix > 0 and generated_prefix > 0:
            cond_public = normalize_distribution(dict(public_suffix[prefix]))
            cond_generated = normalize_distribution(dict(generated_suffix[prefix]))
            suffix_l1 = l1_distance(cond_public, cond_generated)
            suffix_kl = kl_divergence(cond_public, cond_generated)
        elif public_prefix == 0 and generated_prefix == 0:
            suffix_l1 = 0.0
            suffix_kl = 0.0
        else:
            suffix_l1 = 2.0
            suffix_kl = float("inf") if public_prefix > 0 else 0.0

        rows.append(
            {
                "prefix": format_gram(prefix),
                "public_prefix_mass": public_prefix,
                "generated_prefix_mass": generated_prefix,
                "prefix_mass_abs_diff": abs(public_prefix - generated_prefix),
                "prefix_l1_contribution": prefix_l1_contribution,
                "suffix_l1": suffix_l1,
                "suffix_kl": suffix_kl,
                "weighted_suffix_l1": public_prefix * suffix_l1,
            }
        )

    full_df = pd.DataFrame(rows).sort_values("prefix_l1_contribution", ascending=False).reset_index(drop=True)
    return full_df.head(top_k), full_df


def summarise_case(case_name: str, history: pd.DataFrame, rho_final: dict[tuple[int, ...], float], q_final: dict[tuple[int, ...], float], generated_final: dict[tuple[int, ...], float]) -> dict[str, object]:
    final_kl = kl_divergence(rho_final, generated_final)
    return {
        "case_name": case_name,
        "initial_strong_gap_l1": float(history["strong_gap_l1"].iloc[0]),
        "final_strong_gap_l1": l1_distance(rho_final, generated_final),
        "initial_strong_gap_kl": float(history["strong_gap_kl"].iloc[0]),
        "final_strong_gap_kl": None if math.isinf(final_kl) else final_kl,
        "final_public_entropy_r": entropy(rho_final),
        "final_induced_entropy_n": entropy(q_final),
        "final_public_support_r": len(rho_final),
        "final_induced_support_n": len(q_final),
        "final_step_l1": float(history["r_step_l1"].iloc[-1]),
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (OUTPUT_DIR, APPENDIX_FIG_DIR):
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{stem}.png", bbox_inches="tight", dpi=220)


def build_compare_figure(case_results: dict[str, dict[str, object]]) -> None:
    descriptive_history = case_results["descriptive"]["history"]
    normative_history = case_results["normative"]["history"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    colors = {"descriptive": "#1f4e79", "normative": "#b55d24"}

    for label, history in [("descriptive", descriptive_history), ("normative", normative_history)]:
        axes[0, 0].plot(
            history["generation"],
            history["strong_gap_kl"],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            color=colors[label],
            label=label.capitalize(),
        )
        axes[0, 1].plot(
            history["generation"],
            history["strong_gap_l1"],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            color=colors[label],
            label=label.capitalize(),
        )
        axes[1, 0].plot(
            history["generation"],
            history["entropy_r"],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            color=colors[label],
            label=label.capitalize(),
        )
        axes[1, 1].plot(
            history["generation"],
            history["entropy_n"],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            color=colors[label],
            label=label.capitalize(),
        )

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Strong KL gap")
    axes[0, 0].set_ylabel(r"$\mathrm{KL}(\rho_t \| G_r(R_n(\rho_t)))$")

    axes[0, 1].set_title("Strong $L^1$ gap")
    axes[0, 1].set_ylabel(r"$\|\rho_t - G_r(R_n(\rho_t))\|_1$")

    axes[1, 0].set_title(f"Public {R}-gram entropy")
    axes[1, 0].set_ylabel("Entropy (bits)")

    axes[1, 1].set_title(f"Induced {N}-gram entropy")
    axes[1, 1].set_ylabel("Entropy (bits)")

    for ax in axes.flat:
        ax.set_xlabel("Generation")
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=False)

    descriptive_final = case_results["descriptive"]["summary"]["final_strong_gap_kl"]
    normative_final = case_results["normative"]["summary"]["final_strong_gap_kl"]
    fig.suptitle(
        "Matched exact theorem-2 diagnostics: descriptive collapse vs normative plateau\n"
        + f"V={V}, n={N}, r={R}, alpha={ALPHA_REPLACE:.1f}, beta={BETA_MIX:.1f}; "
        + f"final KL = {descriptive_final:.2e} (descriptive) vs {normative_final:.3f} (normative)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, FIG_COMPARE_STEM)
    plt.close(fig)


def build_normative_gap_figure(case_results: dict[str, dict[str, object]]) -> None:
    normative_top = case_results["normative"]["top_rgram_df"].iloc[:12].copy()
    normative_prefix = case_results["normative"]["top_prefix_df"].iloc[:12].copy().sort_values("prefix_l1_contribution", ascending=True)
    normative_top["rgram_label"] = normative_top["rgram"].map(format_compact_gram_label)
    normative_prefix["prefix_label"] = normative_prefix["prefix"].map(format_compact_gram_label)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    x = np.arange(len(normative_top))
    width = 0.38
    axes[0].bar(x - width / 2, normative_top["public_prob"], width=width, label="Corpus 5-gram distribution", color="#b55d24")
    axes[0].bar(x + width / 2, normative_top["generated_prob"], width=width, label="Rollout from induced 3-gram law", color="#1f4e79")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(normative_top["rgram_label"], rotation=60, ha="right")
    axes[0].set_ylabel("Probability")
    axes[0].set_title(f"Largest final {R}-gram mismatches")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.28)

    axes[1].barh(normative_prefix["prefix_label"], normative_prefix["prefix_l1_contribution"], color="#5f7f4b")
    axes[1].set_xlabel("Contribution to total $L^1$ gap")
    axes[1].set_ylabel(f"Prefix of length {N}")
    axes[1].set_title("Prefix contribution to total $L^1$ gap")
    axes[1].grid(True, axis="x", alpha=0.28)

    worst = normative_top.iloc[0]
    fig.suptitle(
        "Structured project-lift gap at the final normative equilibrium\n"
        + f"Worst {R}-gram {format_compact_gram_label(worst['rgram'])} differs by {worst['abs_diff']:.4f}",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    save_figure(fig, FIG_GAP_STEM)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)

    seed_q = initialize_distribution(
        V,
        N,
        init_mode=SEED_INIT_MODE,
        support_size=SEED_SUPPORT_SIZE,
        seed=SEED,
    )

    case_specs = {
        "descriptive": {"lookahead_mode": "descriptive_prev_r_direct"},
        "normative": {"lookahead_mode": "normative_fixed_direct"},
    }

    case_results: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []

    for case_name, spec in case_specs.items():
        result = iterate_theorem2_map(
            seed_q=seed_q,
            lookahead_mode=spec["lookahead_mode"],
            beta_mix=BETA_MIX,
            iterations=ITERATIONS,
            seed=SEED,
        )
        history = result["history"]
        rho_final = result["states"][-1]["rho_r"]
        q_final = result["states"][-1]["q_n"]
        generated_final = exact_rollout_from_law(q_final, V, N, R)
        top_rgram_df, full_rgram_df = top_mismatch_table(rho_final, generated_final, top_k=12)
        top_prefix_df, full_prefix_df = prefix_conditional_diagnostics(rho_final, generated_final, prefix_len=N, top_k=12)
        summary = summarise_case(case_name, history, rho_final, q_final, generated_final)
        summary_rows.append(summary)

        history.to_csv(OUTPUT_DIR / f"{case_name}_history.csv", index=False)
        full_rgram_df.to_csv(OUTPUT_DIR / f"{case_name}_final_rgram_mismatch_full.csv", index=False)
        full_prefix_df.to_csv(OUTPUT_DIR / f"{case_name}_final_prefix_diagnostics_full.csv", index=False)
        with open(OUTPUT_DIR / f"{case_name}_summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)

        case_results[case_name] = {
            "history": history,
            "rho_final": rho_final,
            "q_final": q_final,
            "generated_final": generated_final,
            "top_rgram_df": top_rgram_df,
            "top_prefix_df": top_prefix_df,
            "summary": summary,
            "target_info": result["target_info"],
        }

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "case_summary.csv", index=False)

    manifest = {
        "configuration": {
            "V": V,
            "n": N,
            "r": R,
            "alpha_replace": ALPHA_REPLACE,
            "beta_mix": BETA_MIX,
            "iterations": ITERATIONS,
            "seed": SEED,
            "seed_init_mode": SEED_INIT_MODE,
            "seed_support_size": SEED_SUPPORT_SIZE,
            "initial_public_r_mode": INITIAL_PUBLIC_R_MODE,
            "normative_target_mode": NORMATIVE_TARGET_MODE,
            "rollout_mode": ROLLOUT_MODE,
        },
        "cases": summary_rows,
        "figures": [f"{FIG_COMPARE_STEM}.pdf", f"{FIG_COMPARE_STEM}.png", f"{FIG_GAP_STEM}.pdf", f"{FIG_GAP_STEM}.png"],
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    build_compare_figure(case_results)
    build_normative_gap_figure(case_results)

    summary_lines = [
        "# theorem2 information tutorial case",
        "",
        f"- exact matched configuration: V={V}, n={N}, r={R}, alpha={ALPHA_REPLACE}, beta={BETA_MIX}, iterations={ITERATIONS}, seed={SEED}",
        f"- descriptive final strong KL: {case_results['descriptive']['summary']['final_strong_gap_kl']:.6e}",
        f"- normative final strong KL: {case_results['normative']['summary']['final_strong_gap_kl']:.6f}",
        f"- descriptive final strong L1: {case_results['descriptive']['summary']['final_strong_gap_l1']:.6e}",
        f"- normative final strong L1: {case_results['normative']['summary']['final_strong_gap_l1']:.6f}",
        f"- worst normative {R}-gram mismatch: {case_results['normative']['top_rgram_df'].iloc[0]['rgram']} "
        + f"with abs diff {case_results['normative']['top_rgram_df'].iloc[0]['abs_diff']:.6f}",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
