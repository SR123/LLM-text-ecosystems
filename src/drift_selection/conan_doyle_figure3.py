from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")
CANONICAL_GUTENBERG_IDS = (1661, 834, 108, 244, 2097, 2852)


@dataclass(frozen=True)
class Figure3Config:
    order: int = 3
    lookahead_horizon: int = 5
    generations: int = 10
    transition_budget: int = 120_000
    one_shot_budget: int = 20_000
    replicates: int = 5
    evaluation_trajectories: int = 1_000
    evaluation_horizon: int = 100
    heldout_floor: float = 1e-12
    corpus_split: tuple[float, float, float] = (0.80, 0.10, 0.10)
    seed: int = 20260309

    def __post_init__(self) -> None:
        if self.order != 3:
            raise ValueError("Figure 3 reproduction currently assumes a trigram model (`order=3`).")
        if self.lookahead_horizon < 1:
            raise ValueError("lookahead_horizon must be >= 1")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.transition_budget < 1 or self.one_shot_budget < 1:
            raise ValueError("transition budgets must be positive")
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if self.evaluation_trajectories < 1 or self.evaluation_horizon < 1:
            raise ValueError("evaluation settings must be positive")
        if self.heldout_floor <= 0.0:
            raise ValueError("heldout_floor must be > 0")
        if not math.isclose(sum(self.corpus_split), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("corpus_split must sum to 1")


@dataclass(frozen=True)
class CorpusBundle:
    files: tuple[str, ...]
    train_tokens: tuple[str, ...]
    val_tokens: tuple[str, ...]
    test_tokens: tuple[str, ...]


@dataclass
class TrigramTable:
    trigram_counts: Counter[tuple[str, str, str]]

    def __post_init__(self) -> None:
        prefix_to_next: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        bigram_to_next: dict[str, Counter[str]] = defaultdict(Counter)
        unigram_counts: Counter[str] = Counter()
        prefix_totals: Counter[tuple[str, str]] = Counter()

        for (u, v, w), count in self.trigram_counts.items():
            if count <= 0:
                continue
            prefix_to_next[(u, v)][w] += int(count)
            bigram_to_next[v][w] += int(count)
            unigram_counts[w] += int(count)
            prefix_totals[(u, v)] += int(count)

        self.prefix_to_next = {ctx: ctr for ctx, ctr in prefix_to_next.items() if ctr}
        self.bigram_to_next = {ctx: ctr for ctx, ctr in bigram_to_next.items() if ctr}
        self.unigram_counts = Counter({tok: c for tok, c in unigram_counts.items() if c > 0})
        self.prefix_totals = Counter({ctx: c for ctx, c in prefix_totals.items() if c > 0})
        self.total_trigrams = int(sum(self.trigram_counts.values()))
        self.alive_contexts = set(self.prefix_to_next.keys())

    def trigram_distribution(self, context: tuple[str, str]) -> dict[str, float]:
        counter = self.prefix_to_next.get(context)
        if not counter:
            return {}
        total = float(self.prefix_totals[context])
        return {tok: count / total for tok, count in counter.items()}

    def plain_distribution(self, context: tuple[str, str]) -> tuple[dict[str, float], int]:
        trigram_counter = self.prefix_to_next.get(context)
        if trigram_counter:
            total = float(self.prefix_totals[context])
            return ({tok: count / total for tok, count in trigram_counter.items()}, 3)

        bigram_counter = self.bigram_to_next.get(context[1])
        if bigram_counter:
            total = float(sum(bigram_counter.values()))
            return ({tok: count / total for tok, count in bigram_counter.items()}, 2)

        total = float(sum(self.unigram_counts.values()) or 1.0)
        return ({tok: count / total for tok, count in self.unigram_counts.items()}, 1)

    def joint_probabilities(self) -> tuple[list[tuple[str, str, str]], np.ndarray]:
        trigrams = list(self.trigram_counts.keys())
        total = float(self.total_trigrams or 1.0)
        probs = np.asarray([self.trigram_counts[tri] / total for tri in trigrams], dtype=float)
        return trigrams, probs


def _load_canonical_conan_doyle_corpus(root: Path, cfg: Figure3Config) -> CorpusBundle:
    corpus_dir = root / "GitHub" / "corpora" / "cleaned" / "conan_doyle"
    paths: list[Path] = []
    for gid in CANONICAL_GUTENBERG_IDS:
        matches = sorted(corpus_dir.glob(f"{gid}_*.txt"))
        if not matches:
            raise FileNotFoundError(f"Missing canonical Conan Doyle source for Gutenberg id {gid} in {corpus_dir}")
        paths.append(matches[0])

    text = "\n\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths)
    tokens = tuple(tok.lower() for tok in TOKEN_RE.findall(text))
    n_total = len(tokens)
    if n_total < 10:
        raise ValueError("Canonical Conan Doyle corpus is unexpectedly small.")

    train_end = int(cfg.corpus_split[0] * n_total)
    val_end = train_end + int(cfg.corpus_split[1] * n_total)
    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]
    return CorpusBundle(
        files=tuple(path.name for path in paths),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
    )


def _trigram_counts_from_tokens(tokens: Iterable[str]) -> Counter[tuple[str, str, str]]:
    seq = list(tokens)
    counts: Counter[tuple[str, str, str]] = Counter()
    for i in range(len(seq) - 2):
        counts[(seq[i], seq[i + 1], seq[i + 2])] += 1
    return counts


def _sample_from_counter(counter: Counter[str], rng: random.Random) -> str:
    total = sum(counter.values())
    draw = rng.random() * float(total)
    acc = 0.0
    for token, count in counter.items():
        acc += float(count)
        if draw <= acc:
            return token
    return next(reversed(counter))


def _sample_from_distribution(dist: dict[str, float], rng: random.Random) -> str:
    draw = rng.random()
    acc = 0.0
    last = None
    for token, prob in dist.items():
        last = token
        acc += float(prob)
        if draw <= acc:
            return token
    if last is None:
        raise ValueError("Cannot sample from an empty distribution.")
    return last


def _seed_bigram_bank(tokens: tuple[str, ...], count: int, seed: int) -> list[tuple[str, str]]:
    if len(tokens) < 2:
        raise ValueError("Need at least two training tokens to build a bigram seed bank.")
    bigram_counts: Counter[tuple[str, str]] = Counter()
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i + 1])] += 1
    bigrams = list(bigram_counts.keys())
    weights = [bigram_counts[bigram] for bigram in bigrams]
    rng = random.Random(seed)
    return [rng.choices(bigrams, weights=weights, k=1)[0] for _ in range(count)]


def _distinct_n(sequences: list[list[str]], n: int) -> float:
    grams: list[tuple[str, ...]] = []
    for seq in sequences:
        if len(seq) < n:
            continue
        grams.extend(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))
    if not grams:
        return 0.0
    return len(set(grams)) / float(len(grams))


def _survival_probabilities(table: TrigramTable, horizon: int) -> dict[tuple[str, str], float]:
    if horizon <= 0:
        return {ctx: 1.0 for ctx in table.alive_contexts}
    prev = {ctx: 1.0 for ctx in table.alive_contexts}
    for _ in range(horizon):
        current: dict[tuple[str, str], float] = {}
        for context, counter in table.prefix_to_next.items():
            total = float(table.prefix_totals[context])
            value = 0.0
            _, v = context
            for token, count in counter.items():
                next_context = (v, token)
                if next_context in table.alive_contexts:
                    value += (count / total) * prev.get(next_context, 0.0)
            current[context] = value
        prev = current
    return prev


def _lookahead_distributions(table: TrigramTable, lookahead_horizon: int) -> dict[tuple[str, str], dict[str, float]]:
    if lookahead_horizon < 1:
        return {ctx: table.trigram_distribution(ctx) for ctx in table.alive_contexts}

    future_survival = _survival_probabilities(table, lookahead_horizon - 1)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for context, counter in table.prefix_to_next.items():
        _, v = context
        weights: dict[str, float] = {}
        total = 0.0
        for token, count in counter.items():
            next_context = (v, token)
            weight = float(count) * future_survival.get(next_context, 0.0)
            if weight > 0.0:
                weights[token] = weight
                total += weight
        if total <= 0.0:
            out[context] = table.trigram_distribution(context)
        else:
            out[context] = {token: weight / total for token, weight in weights.items()}
    return out


def _sample_resampled_table(
    table: TrigramTable,
    budget: int,
    rng: np.random.Generator,
    *,
    joint_override: dict[tuple[str, str, str], float] | None = None,
) -> TrigramTable:
    if joint_override is None:
        trigrams, probs = table.joint_probabilities()
    else:
        trigrams = list(joint_override.keys())
        probs = np.asarray([joint_override[tri] for tri in trigrams], dtype=float)
        probs = probs / probs.sum()

    sampled = rng.multinomial(budget, probs)
    counts = Counter({tri: int(count) for tri, count in zip(trigrams, sampled) if int(count) > 0})
    return TrigramTable(counts)


def _filtered_joint_distribution(
    table: TrigramTable,
    lookahead_dist: dict[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str, str], float]:
    total = float(table.total_trigrams or 1.0)
    out: dict[tuple[str, str, str], float] = {}
    for context, prefix_total in table.prefix_totals.items():
        context_mass = prefix_total / total
        dist = lookahead_dist[context]
        u, v = context
        for token, prob in dist.items():
            out[(u, v, token)] = context_mass * float(prob)
    return out


def _next_distribution(
    table: TrigramTable,
    context: tuple[str, str],
    *,
    lookahead_dist: dict[tuple[str, str], dict[str, float]] | None,
    explicit_lookahead: bool,
) -> tuple[dict[str, float], int]:
    if explicit_lookahead and lookahead_dist is not None and context in lookahead_dist:
        return lookahead_dist[context], 3
    return table.plain_distribution(context)


def _evaluate_trajectories(
    table: TrigramTable,
    seed_bank: list[tuple[str, str]],
    cfg: Figure3Config,
    *,
    random_seed: int,
    explicit_lookahead: bool,
    lookahead_dist: dict[tuple[str, str], dict[str, float]] | None,
) -> dict[str, float]:
    rng = random.Random(random_seed)
    continuations: list[list[str]] = []
    first_backoff_steps: list[int] = []
    full_order_tokens = 0
    survived = 0

    for seed_context in seed_bank:
        context = (seed_context[0], seed_context[1])
        continuation: list[str] = []
        first_backoff = cfg.evaluation_horizon
        all_full_order = True
        for step in range(1, cfg.evaluation_horizon + 1):
            dist, order_used = _next_distribution(
                table,
                context,
                lookahead_dist=lookahead_dist,
                explicit_lookahead=explicit_lookahead,
            )
            token = _sample_from_distribution(dist, rng)
            continuation.append(token)
            if order_used == 3:
                full_order_tokens += 1
            elif first_backoff == cfg.evaluation_horizon:
                first_backoff = step - 1
                all_full_order = False
            context = (context[1], token)
        if all_full_order:
            survived += 1
        first_backoff_steps.append(first_backoff)
        continuations.append(continuation)

    return {
        "mean_time_to_first_backoff": float(sum(first_backoff_steps) / len(first_backoff_steps)),
        "full_order_token_fraction": float(full_order_tokens / (len(seed_bank) * cfg.evaluation_horizon)),
        "survival_without_backoff_100": float(survived / len(seed_bank)),
        "early_backoff_before_100": float(1.0 - (survived / len(seed_bank))),
        "distinct_2": float(_distinct_n(continuations, 2)),
        "repetition_4": float(1.0 - _distinct_n(continuations, 4)),
    }


def _heldout_perplexity(
    table: TrigramTable,
    test_tokens: tuple[str, ...],
    cfg: Figure3Config,
    *,
    explicit_lookahead: bool,
    lookahead_dist: dict[tuple[str, str], dict[str, float]] | None,
) -> float:
    if len(test_tokens) < 3:
        return float("nan")
    nll = 0.0
    count = 0
    for i in range(2, len(test_tokens)):
        context = (test_tokens[i - 2], test_tokens[i - 1])
        dist, _ = _next_distribution(
            table,
            context,
            lookahead_dist=lookahead_dist,
            explicit_lookahead=explicit_lookahead,
        )
        prob = max(float(dist.get(test_tokens[i], 0.0)), cfg.heldout_floor)
        nll -= math.log(prob)
        count += 1
    return float(math.exp(nll / max(count, 1)))


def _collect_replicate_rows(corpus: CorpusBundle, cfg: Figure3Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    initial_table = TrigramTable(_trigram_counts_from_tokens(corpus.train_tokens))
    seed_bank = _seed_bigram_bank(corpus.train_tokens, cfg.evaluation_trajectories, cfg.seed + 17)

    rows: list[dict[str, float | int | str]] = []
    bar_rows: list[dict[str, float | int | str]] = []
    final_filtered_tables: list[TrigramTable] = []

    for replicate in range(cfg.replicates):
        neutral_table = initial_table
        filtered_table = initial_table
        neutral_rng = np.random.default_rng(cfg.seed + 1000 + replicate)
        filtered_rng = np.random.default_rng(cfg.seed + 2000 + replicate)

        for generation in range(cfg.generations + 1):
            neutral_eval = _evaluate_trajectories(
                neutral_table,
                seed_bank,
                cfg,
                random_seed=cfg.seed + 10_000 * (replicate + 1) + generation,
                explicit_lookahead=False,
                lookahead_dist=None,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "generation": generation,
                    "condition": "neutral / plain",
                    **neutral_eval,
                    "heldout_perplexity": _heldout_perplexity(
                        neutral_table,
                        corpus.test_tokens,
                        cfg,
                        explicit_lookahead=False,
                        lookahead_dist=None,
                    ),
                }
            )

            filtered_lookahead = _lookahead_distributions(filtered_table, cfg.lookahead_horizon)
            filtered_plain_eval = _evaluate_trajectories(
                filtered_table,
                seed_bank,
                cfg,
                random_seed=cfg.seed + 20_000 * (replicate + 1) + generation,
                explicit_lookahead=False,
                lookahead_dist=filtered_lookahead,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "generation": generation,
                    "condition": "lookahead-trained / plain",
                    **filtered_plain_eval,
                    "heldout_perplexity": _heldout_perplexity(
                        filtered_table,
                        corpus.test_tokens,
                        cfg,
                        explicit_lookahead=False,
                        lookahead_dist=filtered_lookahead,
                    ),
                }
            )

            filtered_explicit_eval = _evaluate_trajectories(
                filtered_table,
                seed_bank,
                cfg,
                random_seed=cfg.seed + 30_000 * (replicate + 1) + generation,
                explicit_lookahead=True,
                lookahead_dist=filtered_lookahead,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "generation": generation,
                    "condition": "lookahead-trained / explicit lookahead",
                    **filtered_explicit_eval,
                    "heldout_perplexity": _heldout_perplexity(
                        filtered_table,
                        corpus.test_tokens,
                        cfg,
                        explicit_lookahead=True,
                        lookahead_dist=filtered_lookahead,
                    ),
                }
            )

            if generation == cfg.generations:
                final_filtered_tables.append(filtered_table)
                break

            neutral_table = _sample_resampled_table(neutral_table, cfg.transition_budget, neutral_rng)
            filtered_joint = _filtered_joint_distribution(filtered_table, filtered_lookahead)
            filtered_table = _sample_resampled_table(
                filtered_table,
                cfg.transition_budget,
                filtered_rng,
                joint_override=filtered_joint,
            )

        original_one_shot = _sample_resampled_table(
            initial_table,
            cfg.one_shot_budget,
            np.random.default_rng(cfg.seed + 40_000 + replicate),
        )
        original_eval = _evaluate_trajectories(
            original_one_shot,
            seed_bank,
            cfg,
            random_seed=cfg.seed + 50_000 + replicate,
            explicit_lookahead=False,
            lookahead_dist=None,
        )
        bar_rows.append(
            {
                "replicate": replicate,
                "condition": "one-shot from original",
                "mean_time_to_first_backoff": original_eval["mean_time_to_first_backoff"],
                "full_order_token_fraction": original_eval["full_order_token_fraction"],
            }
        )

    for replicate, final_table in enumerate(final_filtered_tables):
        filtered_one_shot = _sample_resampled_table(
            final_table,
            cfg.one_shot_budget,
            np.random.default_rng(cfg.seed + 60_000 + replicate),
        )
        filtered_eval = _evaluate_trajectories(
            filtered_one_shot,
            seed_bank,
            cfg,
            random_seed=cfg.seed + 70_000 + replicate,
            explicit_lookahead=False,
            lookahead_dist=None,
        )
        bar_rows.append(
            {
                "replicate": replicate,
                "condition": "one-shot from iterated filtered text",
                "mean_time_to_first_backoff": filtered_eval["mean_time_to_first_backoff"],
                "full_order_token_fraction": filtered_eval["full_order_token_fraction"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    bars_df = pd.DataFrame(bar_rows)
    summary_df = (
        metrics_df.groupby(["generation", "condition"], as_index=False)
        .agg(
            mean_time_to_first_backoff=("mean_time_to_first_backoff", "mean"),
            full_order_token_fraction=("full_order_token_fraction", "mean"),
            survival_without_backoff_100=("survival_without_backoff_100", "mean"),
            early_backoff_before_100=("early_backoff_before_100", "mean"),
            distinct_2=("distinct_2", "mean"),
            repetition_4=("repetition_4", "mean"),
            heldout_perplexity=("heldout_perplexity", "mean"),
        )
    )
    return metrics_df, summary_df, bars_df


def _plot_main_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    pivot = summary_df.pivot(index="generation", columns="condition")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("mean_time_to_first_backoff", "A. Mean time to first back-off", "tokens before back-off"),
        ("full_order_token_fraction", "B. Fraction of tokens at full trigram order", "full-order token fraction"),
        ("early_backoff_before_100", "C. Fraction backing off before 100 tokens", "early-back-off rate"),
        ("distinct_2", "D. Distinct-2", "distinct-2"),
    ]
    colors = {
        "neutral / plain": "#1f77b4",
        "lookahead-trained / plain": "#ff7f0e",
    }
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        for condition, color in colors.items():
            ax.plot(
                pivot.index,
                pivot[(metric, condition)],
                linewidth=1.8,
                label=condition,
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("generation")
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_explicit_lookahead_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    pivot = summary_df.pivot(index="generation", columns="condition")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("mean_time_to_first_backoff", "A. Mean time to first back-off", "tokens before back-off"),
        ("full_order_token_fraction", "B. Fraction of tokens at full trigram order", "full-order token fraction"),
        ("survival_without_backoff_100", "C. Fraction surviving 100 tokens without back-off", "100-token survival probability"),
        ("distinct_2", "D. Distinct-2", "distinct-2"),
    ]
    colors = {
        "neutral / plain": "#1f77b4",
        "lookahead-trained / plain": "#ff7f0e",
        "lookahead-trained / explicit lookahead": "#2ca02c",
    }
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        for condition, color in colors.items():
            ax.plot(
                pivot.index,
                pivot[(metric, condition)],
                linewidth=1.8,
                label=condition,
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("generation")
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_perplexity(summary_df: pd.DataFrame, output_path: Path) -> None:
    subset = summary_df[summary_df["condition"].isin(
        ["neutral / plain", "lookahead-trained / plain", "lookahead-trained / explicit lookahead"]
    )]
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {
        "neutral / plain": "#1f77b4",
        "lookahead-trained / plain": "#ff7f0e",
        "lookahead-trained / explicit lookahead": "#2ca02c",
    }
    for condition, color in colors.items():
        frame = subset[subset["condition"] == condition].sort_values("generation")
        ax.plot(frame["generation"], frame["heldout_perplexity"], linewidth=1.8, label=condition, color=color)
    ax.set_title("Held-out perplexity on Doyle test set (supplementary)")
    ax.set_xlabel("generation")
    ax.set_ylabel("perplexity")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_one_shot_bars(bars_df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        bars_df.groupby("condition", as_index=False)
        .agg(
            mean_time_to_first_backoff=("mean_time_to_first_backoff", "mean"),
            full_order_token_fraction=("full_order_token_fraction", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(summary))
    labels = summary["condition"].tolist()

    axes[0].bar(x, summary["mean_time_to_first_backoff"], color="#1f77b4")
    axes[0].set_title("Mean time to first back-off")
    axes[0].set_ylabel("tokens before back-off")
    axes[0].set_xticks(x, labels)
    for i, value in enumerate(summary["mean_time_to_first_backoff"]):
        axes[0].text(i, value + 0.15, f"{value:.2f}", ha="center", va="bottom")

    axes[1].bar(x, summary["full_order_token_fraction"], color="#1f77b4")
    axes[1].set_title("Fraction of tokens at full trigram order")
    axes[1].set_ylabel("full-order token fraction")
    axes[1].set_xticks(x, labels)
    for i, value in enumerate(summary["full_order_token_fraction"]):
        axes[1].text(i, value + 0.03, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def run_figure3_reproduction(
    root: Path,
    output_dir: Path,
    cfg: Figure3Config | None = None,
) -> dict[str, str]:
    cfg = cfg or Figure3Config()
    corpus = _load_canonical_conan_doyle_corpus(root, cfg)
    metrics_df, summary_df, bars_df = _collect_replicate_rows(corpus, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    metrics_path = output_dir / "figure3_replicate_metrics.csv"
    summary_path = output_dir / "figure3_summary.csv"
    bars_path = output_dir / "figure3_one_shot_bars.csv"
    manifest_path = output_dir / "run_manifest.json"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    bars_df.to_csv(bars_path, index=False)

    main_figure_pdf = figures_dir / "figure_metrics_plain_vs_plain.pdf"
    explicit_pdf = figures_dir / "figure_metrics_with_explicit_lookahead.pdf"
    perplexity_pdf = figures_dir / "figure_perplexity_supplement.pdf"
    bars_pdf = figures_dir / "figure_theorem2_one_shot_bars.pdf"

    _plot_main_figure(summary_df, main_figure_pdf)
    _plot_explicit_lookahead_figure(summary_df, explicit_pdf)
    _plot_perplexity(summary_df, perplexity_pdf)
    _plot_one_shot_bars(bars_df, bars_pdf)

    manifest = {
        "config": asdict(cfg),
        "canonical_gutenberg_ids": list(CANONICAL_GUTENBERG_IDS),
        "corpus_files": list(corpus.files),
        "train_tokens": len(corpus.train_tokens),
        "val_tokens": len(corpus.val_tokens),
        "test_tokens": len(corpus.test_tokens),
        "outputs": {
            "figure3_replicate_metrics": str(metrics_path),
            "figure3_summary": str(summary_path),
            "figure3_one_shot_bars": str(bars_path),
            "figure_metrics_plain_vs_plain": str(main_figure_pdf),
            "figure_metrics_with_explicit_lookahead": str(explicit_pdf),
            "figure_perplexity_supplement": str(perplexity_pdf),
            "figure_theorem2_one_shot_bars": str(bars_pdf),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "metrics_csv": str(metrics_path),
        "summary_csv": str(summary_path),
        "bars_csv": str(bars_path),
        "manifest": str(manifest_path),
        "main_figure": str(main_figure_pdf),
        "explicit_figure": str(explicit_pdf),
        "perplexity_figure": str(perplexity_pdf),
        "bars_figure": str(bars_pdf),
    }
