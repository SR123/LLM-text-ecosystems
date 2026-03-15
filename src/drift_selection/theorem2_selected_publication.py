from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterable, Sequence

from tqdm.auto import tqdm

from .checkpoints import atomic_save_json, load_pickle_checkpoint, save_pickle_checkpoint
from .utils import ensure_dir, stable_slug, timestamp


MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def stable_hash64(seed: int, *values: int) -> int:
    x = seed & MASK64
    for value in values:
        x = _splitmix64(x ^ (int(value) & MASK64))
    return x


def stable_u01(seed: int, *values: int) -> float:
    return stable_hash64(seed, *values) / float(1 << 64)


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _weighted_choice(tokens: Sequence[int], weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        return int(rng.choice(list(tokens)))
    draw = rng.random() * total
    acc = 0.0
    for token, weight in zip(tokens, weights):
        acc += weight
        if draw <= acc:
            return int(token)
    return int(tokens[-1])


@dataclass(frozen=True)
class SyntheticGrammarConfig:
    vocab_size: int = 256
    keep_prob_2gram: float = 0.50
    keep_prob_3gram: float = 0.25
    exact_support: bool = False
    support_cache_limit: int = 4096
    base_weight_seed: int = 11
    support_seed: int = 29
    edge_weight_seed: int = 37

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        _validate_probability("keep_prob_2gram", self.keep_prob_2gram)
        _validate_probability("keep_prob_3gram", self.keep_prob_3gram)
        if self.support_cache_limit < 1:
            raise ValueError("support_cache_limit must be >= 1")


@dataclass(frozen=True)
class PublicationPolicy:
    name: str
    publish_neutral_prob: float = 0.35
    publish_desirable_prob: float = 1.0
    publish_undesirable_prob: float = 0.0
    publish_mixed_prob: float = 0.0

    def __post_init__(self) -> None:
        _validate_probability("publish_neutral_prob", self.publish_neutral_prob)
        _validate_probability("publish_desirable_prob", self.publish_desirable_prob)
        _validate_probability("publish_undesirable_prob", self.publish_undesirable_prob)
        _validate_probability("publish_mixed_prob", self.publish_mixed_prob)

    def publication_probability(self, category: str) -> float:
        if category == "desirable":
            return self.publish_desirable_prob
        if category == "undesirable":
            return self.publish_undesirable_prob
        if category == "mixed":
            return self.publish_mixed_prob
        return self.publish_neutral_prob


@dataclass(frozen=True)
class ExperimentConfig:
    sequence_length: int = 40
    rounds: int = 6
    candidate_count: int = 400
    evaluation_count: int = 200
    search_trials: int = 24
    prefix_length: int = 2
    learner_order: int = 3
    branch_factor: int = 6
    max_expansions: int = 2500

    def __post_init__(self) -> None:
        if self.sequence_length < 3:
            raise ValueError("sequence_length must be >= 3")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.evaluation_count < 1:
            raise ValueError("evaluation_count must be >= 1")
        if self.search_trials < 1:
            raise ValueError("search_trials must be >= 1")
        if not 0 <= self.prefix_length < self.sequence_length:
            raise ValueError("prefix_length must satisfy 0 <= prefix_length < sequence_length")
        if self.learner_order < 1:
            raise ValueError("learner_order must be >= 1")
        if self.branch_factor < 1:
            raise ValueError("branch_factor must be >= 1")
        if self.max_expansions < 1:
            raise ValueError("max_expansions must be >= 1")


@dataclass
class PatternLibrary:
    span: int
    desirable: tuple[tuple[int, ...], ...]
    undesirable: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.span < 2:
            raise ValueError("span must be >= 2")
        desirable = tuple(tuple(int(x) for x in pat) for pat in self.desirable)
        undesirable = tuple(tuple(int(x) for x in pat) for pat in self.undesirable)
        if any(len(pat) != self.span for pat in desirable + undesirable):
            raise ValueError("all patterns must have length equal to span")
        desirable_set = set(desirable)
        undesirable_set = set(undesirable)
        if desirable_set & undesirable_set:
            raise ValueError("desirable and undesirable patterns must be disjoint")
        self.desirable = desirable
        self.undesirable = undesirable
        self.desirable_set = desirable_set
        self.undesirable_set = undesirable_set

    def count_hits(self, tokens: Sequence[int]) -> tuple[int, int]:
        if len(tokens) < self.span:
            return 0, 0
        desirable_hits = 0
        undesirable_hits = 0
        for i in range(len(tokens) - self.span + 1):
            window = tuple(int(x) for x in tokens[i:i + self.span])
            if window in self.desirable_set:
                desirable_hits += 1
            if window in self.undesirable_set:
                undesirable_hits += 1
        return desirable_hits, undesirable_hits

    def classify(self, tokens: Sequence[int]) -> tuple[int, int, str]:
        desirable_hits, undesirable_hits = self.count_hits(tokens)
        if desirable_hits and undesirable_hits:
            return desirable_hits, undesirable_hits, "mixed"
        if desirable_hits:
            return desirable_hits, undesirable_hits, "desirable"
        if undesirable_hits:
            return desirable_hits, undesirable_hits, "undesirable"
        return desirable_hits, undesirable_hits, "neutral"

    def suffix_flags(self, tokens: Sequence[int]) -> tuple[bool, bool]:
        if len(tokens) < self.span:
            return False, False
        window = tuple(int(x) for x in tokens[-self.span:])
        return window in self.desirable_set, window in self.undesirable_set


@dataclass(frozen=True)
class SequenceRecord:
    tokens: tuple[int, ...]
    order_counts: tuple[int, int, int]
    desirable_hits: int
    undesirable_hits: int
    category: str

    @property
    def has_desirable(self) -> bool:
        return self.desirable_hits > 0

    @property
    def has_undesirable(self) -> bool:
        return self.undesirable_hits > 0


@dataclass(frozen=True)
class GeneratedSequence:
    tokens: tuple[int, ...]
    order_counts: tuple[int, int, int]


@dataclass
class CandidateBatch:
    round_index: int
    records: tuple[SequenceRecord, ...]


@dataclass
class PolicyTrajectory:
    policy: PublicationPolicy
    rows: list[dict[str, float | int | str]]
    published_sequences: list[tuple[int, ...]]
    learner: "CountBackoffModel"


@dataclass
class ComparisonResult:
    patterns: PatternLibrary
    candidate_batches: list[CandidateBatch]
    trajectories: dict[str, PolicyTrajectory]

    def table_rows(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for trajectory in self.trajectories.values():
            rows.extend(trajectory.rows)
        return rows


@dataclass(frozen=True)
class RunPaths:
    root_dir: Path
    run_dir: Path
    state_dir: Path
    figures_dir: Path
    tables_dir: Path
    manifest_path: Path
    checkpoint_path: Path
    metrics_partial_path: Path
    metrics_final_path: Path
    summary_path: Path
    patterns_path: Path
    samples_path: Path


@dataclass
class ExperimentRun:
    version: str
    run_name: str
    paths: RunPaths
    manifest: dict[str, Any]
    result: ComparisonResult

    @property
    def table_rows(self) -> list[dict[str, float | int | str]]:
        return self.result.table_rows()


class SyntheticBackoffGrammar:
    """Synthetic 3-gram-with-backoff generator.

    For scalability, the default support mode uses a fixed random fanout
    `round(p * M)` per context instead of scanning all M continuations.
    Setting `exact_support=True` restores Bernoulli edge inclusion.
    """

    def __init__(self, config: SyntheticGrammarConfig):
        self.config = config
        self.tokens = tuple(range(1, config.vocab_size + 1))
        self._bigram_cache: dict[int, tuple[int, ...]] = {}
        self._trigram_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self._bigram_fanout = max(0, min(config.vocab_size, int(round(config.keep_prob_2gram * config.vocab_size))))
        self._trigram_fanout = max(0, min(config.vocab_size, int(round(config.keep_prob_3gram * config.vocab_size))))
        self._base_weights = {
            token: 0.25 + stable_u01(config.base_weight_seed, token) + stable_u01(config.base_weight_seed + 1, token)
            for token in self.tokens
        }

    def _clear_cache_if_needed(self, cache: dict[object, object]) -> None:
        if len(cache) >= self.config.support_cache_limit:
            cache.clear()

    def _exact_successors(self, order_marker: int, context: Sequence[int], keep_prob: float) -> tuple[int, ...]:
        if keep_prob <= 0.0:
            return tuple()
        return tuple(
            token
            for token in self.tokens
            if stable_u01(self.config.support_seed, order_marker, *context, token) < keep_prob
        )

    def _sampled_successors(self, order_marker: int, context: Sequence[int], fanout: int) -> tuple[int, ...]:
        if fanout <= 0:
            return tuple()
        if fanout >= self.config.vocab_size:
            return self.tokens
        seed = stable_hash64(self.config.support_seed, order_marker, *context)
        rng = random.Random(seed)
        return tuple(int(x) for x in rng.sample(range(1, self.config.vocab_size + 1), fanout))

    def _bigram_successors(self, prev_token: int) -> tuple[int, ...]:
        if prev_token in self._bigram_cache:
            return self._bigram_cache[prev_token]
        if self.config.exact_support:
            successors = self._exact_successors(2, (prev_token,), self.config.keep_prob_2gram)
        else:
            successors = self._sampled_successors(2, (prev_token,), self._bigram_fanout)
        self._clear_cache_if_needed(self._bigram_cache)
        self._bigram_cache[prev_token] = successors
        return successors

    def _trigram_successors(self, prev2: int, prev1: int) -> tuple[int, ...]:
        key = (prev2, prev1)
        if key in self._trigram_cache:
            return self._trigram_cache[key]
        if self.config.exact_support:
            successors = self._exact_successors(3, key, self.config.keep_prob_3gram)
        else:
            successors = self._sampled_successors(3, key, self._trigram_fanout)
        self._clear_cache_if_needed(self._trigram_cache)
        self._trigram_cache[key] = successors
        return successors

    def _edge_weight(self, order_used: int, context: Sequence[int], token: int) -> float:
        return self._base_weights[int(token)] * (0.5 + stable_u01(self.config.edge_weight_seed, order_used, *context, int(token)))

    def weighted_candidates(self, context: Sequence[int]) -> tuple[tuple[int, ...], list[float], int]:
        ctx = tuple(int(x) for x in context[-2:])
        if len(ctx) == 2:
            trigram = self._trigram_successors(ctx[0], ctx[1])
            if trigram:
                return trigram, [self._edge_weight(3, ctx, token) for token in trigram], 3
        if len(ctx) >= 1:
            bigram = self._bigram_successors(ctx[-1])
            if bigram:
                return bigram, [self._edge_weight(2, (ctx[-1],), token) for token in bigram], 2
        return self.tokens, [self._base_weights[token] for token in self.tokens], 1

    def distribution(self, context: Sequence[int]) -> tuple[dict[int, float], int]:
        tokens, weights, order_used = self.weighted_candidates(context)
        total = sum(weights) or 1.0
        return {int(token): weight / total for token, weight in zip(tokens, weights)}, order_used

    def top_tokens(self, context: Sequence[int], limit: int = 10) -> list[tuple[int, float]]:
        dist, _ = self.distribution(context)
        return sorted(dist.items(), key=lambda item: item[1], reverse=True)[:limit]

    def sample_next(self, context: Sequence[int], rng: random.Random | None = None) -> tuple[int, int]:
        rng = rng or random.Random()
        tokens, weights, order_used = self.weighted_candidates(context)
        return _weighted_choice(tokens, weights, rng), order_used

    def generate(
        self,
        length: int,
        rng: random.Random | None = None,
        prefix: Sequence[int] | None = None,
    ) -> tuple[list[int], tuple[int, int, int]]:
        rng = rng or random.Random()
        out = [int(x) for x in (prefix or [])]
        order_counts = [0, 0, 0]
        while len(out) < length:
            token, order_used = self.sample_next(out[-2:], rng=rng)
            out.append(token)
            order_counts[order_used - 1] += 1
        return out, (order_counts[0], order_counts[1], order_counts[2])


class CountBackoffModel:
    def __init__(self, vocab_size: int, order: int = 3):
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if order < 1:
            raise ValueError("order must be >= 1")
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        self.tokens = tuple(range(1, self.vocab_size + 1))
        self.counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.totals: Counter[tuple[int, ...]] = Counter()

    def fit_sequences(self, sequences: Iterable[Sequence[int]]) -> None:
        self.counts.clear()
        self.totals.clear()
        for sequence in sequences:
            seq = [int(x) for x in sequence]
            for i, token in enumerate(seq):
                max_ctx_len = min(self.order - 1, i)
                for ctx_len in range(max_ctx_len + 1):
                    ctx = tuple(seq[i - ctx_len:i]) if ctx_len else tuple()
                    self.counts[ctx][token] += 1
                    self.totals[ctx] += 1

    def distribution(self, context: Sequence[int]) -> tuple[dict[int, float], int]:
        ctx = tuple(int(x) for x in context)
        for ctx_len in range(min(self.order - 1, len(ctx)), -1, -1):
            key = ctx[-ctx_len:] if ctx_len else tuple()
            total = self.totals.get(key, 0)
            if total > 0:
                counter = self.counts[key]
                return {int(token): count / float(total) for token, count in counter.items()}, ctx_len + 1
        uniform = 1.0 / float(self.vocab_size)
        return {token: uniform for token in self.tokens}, 1

    def top_tokens(self, context: Sequence[int], limit: int = 10) -> list[tuple[int, float]]:
        dist, _ = self.distribution(context)
        return sorted(dist.items(), key=lambda item: item[1], reverse=True)[:limit]

    def sample_next(self, context: Sequence[int], rng: random.Random | None = None) -> tuple[int, int]:
        rng = rng or random.Random()
        dist, order_used = self.distribution(context)
        tokens = list(dist.keys())
        weights = list(dist.values())
        return _weighted_choice(tokens, weights, rng), order_used

    def generate(
        self,
        length: int,
        rng: random.Random | None = None,
        prefix: Sequence[int] | None = None,
    ) -> tuple[list[int], tuple[int, int, int]]:
        rng = rng or random.Random()
        out = [int(x) for x in (prefix or [])]
        order_counts = [0, 0, 0]
        while len(out) < length:
            token, order_used = self.sample_next(out[-(self.order - 1):], rng=rng)
            out.append(token)
            order_counts[min(order_used, 3) - 1] += 1
        return out, (order_counts[0], order_counts[1], order_counts[2])


def make_sequence_record(
    tokens: Sequence[int],
    order_counts: tuple[int, int, int],
    patterns: PatternLibrary,
) -> SequenceRecord:
    desirable_hits, undesirable_hits, category = patterns.classify(tokens)
    return SequenceRecord(
        tokens=tuple(int(x) for x in tokens),
        order_counts=tuple(int(x) for x in order_counts),
        desirable_hits=int(desirable_hits),
        undesirable_hits=int(undesirable_hits),
        category=category,
    )


def summarize_records(records: Sequence[SequenceRecord]) -> dict[str, float]:
    total = len(records)
    if total == 0:
        return {
            "count": 0.0,
            "desirable_rate": 0.0,
            "undesirable_rate": 0.0,
            "mixed_rate": 0.0,
            "neutral_rate": 0.0,
            "avg_desirable_hits": 0.0,
            "avg_undesirable_hits": 0.0,
            "unigram_rate": 0.0,
            "bigram_rate": 0.0,
            "trigram_rate": 0.0,
        }
    category_counts = Counter(record.category for record in records)
    total_steps = sum(sum(record.order_counts) for record in records) or 1
    return {
        "count": float(total),
        "desirable_rate": category_counts["desirable"] / float(total),
        "undesirable_rate": category_counts["undesirable"] / float(total),
        "mixed_rate": category_counts["mixed"] / float(total),
        "neutral_rate": category_counts["neutral"] / float(total),
        "avg_desirable_hits": sum(record.desirable_hits for record in records) / float(total),
        "avg_undesirable_hits": sum(record.undesirable_hits for record in records) / float(total),
        "unigram_rate": sum(record.order_counts[0] for record in records) / float(total_steps),
        "bigram_rate": sum(record.order_counts[1] for record in records) / float(total_steps),
        "trigram_rate": sum(record.order_counts[2] for record in records) / float(total_steps),
    }


def sample_pattern_library(
    grammar: SyntheticBackoffGrammar,
    *,
    span: int = 5,
    desirable_count: int = 4,
    undesirable_count: int = 4,
    pool_sequences: int = 2000,
    sequence_length: int = 40,
    min_count: int = 2,
    selection_pool_size: int = 200,
    seed: int = 123,
    progress_bar: bool = False,
) -> PatternLibrary:
    if desirable_count < 1 or undesirable_count < 1:
        raise ValueError("desirable_count and undesirable_count must be >= 1")
    if pool_sequences < 1:
        raise ValueError("pool_sequences must be >= 1")
    rng = random.Random(seed)
    iterator = range(pool_sequences)
    iterator = tqdm(iterator, desc="pattern-pool", unit="seq", disable=not progress_bar)
    sequences: list[tuple[int, ...]] = []
    for _ in iterator:
        tokens, _ = grammar.generate(sequence_length, rng=rng)
        sequences.append(tuple(tokens))
    return sample_pattern_library_from_sequences(
        sequences,
        span=span,
        desirable_count=desirable_count,
        undesirable_count=undesirable_count,
        min_count=min_count,
        selection_pool_size=selection_pool_size,
        seed=seed,
    )


def sample_pattern_library_from_sequences(
    sequences: Iterable[Sequence[int]],
    *,
    span: int,
    desirable_count: int,
    undesirable_count: int,
    min_count: int = 2,
    selection_pool_size: int = 200,
    seed: int = 123,
) -> PatternLibrary:
    rng = random.Random(seed)
    window_counts: Counter[tuple[int, ...]] = Counter()
    for tokens in sequences:
        tokens = list(tokens)
        for i in range(len(tokens) - span + 1):
            window_counts[tuple(int(x) for x in tokens[i:i + span])] += 1
    candidates = [window for window, count in window_counts.most_common(selection_pool_size) if count >= min_count]
    if len(candidates) < desirable_count + undesirable_count:
        candidates = [window for window, _ in window_counts.most_common(desirable_count + undesirable_count)]
    if len(candidates) < desirable_count + undesirable_count:
        raise ValueError("not enough distinct span-patterns were observed; reduce span or increase pool_sequences")
    rng.shuffle(candidates)
    desirable = tuple(candidates[:desirable_count])
    undesirable = tuple(candidates[desirable_count:desirable_count + undesirable_count])
    return PatternLibrary(span=span, desirable=desirable, undesirable=undesirable)


def generate_candidate_batch(
    grammar: SyntheticBackoffGrammar,
    patterns: PatternLibrary,
    config: ExperimentConfig,
    *,
    round_index: int,
    seed: int = 123,
    progress_bar: bool = False,
) -> CandidateBatch:
    raw_batch = generate_raw_candidate_batch(
        grammar,
        config,
        round_index=round_index,
        seed=seed,
        progress_bar=progress_bar,
    )
    return classify_candidate_batch(raw_batch, patterns)


def generate_raw_candidate_batch(
    grammar: SyntheticBackoffGrammar,
    config: ExperimentConfig,
    *,
    round_index: int,
    seed: int = 123,
    progress_bar: bool = False,
) -> tuple[GeneratedSequence, ...]:
    rng = random.Random(seed + 7919 * int(round_index))
    records: list[GeneratedSequence] = []
    iterator = range(config.candidate_count)
    iterator = tqdm(
        iterator,
        desc=f"candidates:r{round_index}",
        unit="seq",
        leave=False,
        disable=not progress_bar,
    )
    for _ in iterator:
        tokens, order_counts = grammar.generate(config.sequence_length, rng=rng)
        records.append(GeneratedSequence(tokens=tuple(tokens), order_counts=tuple(order_counts)))
    return tuple(records)


def classify_candidate_batch(
    raw_batch: Sequence[GeneratedSequence],
    patterns: PatternLibrary,
    *,
    round_index: int | None = None,
) -> CandidateBatch:
    records = [make_sequence_record(item.tokens, item.order_counts, patterns) for item in raw_batch]
    return CandidateBatch(round_index=int(round_index or 0), records=tuple(records))


def generate_raw_candidate_batches(
    grammar: SyntheticBackoffGrammar,
    config: ExperimentConfig,
    *,
    seed: int = 123,
    progress_bar: bool = False,
) -> list[tuple[GeneratedSequence, ...]]:
    batches: list[tuple[GeneratedSequence, ...]] = []
    iterator = range(1, config.rounds + 1)
    iterator = tqdm(iterator, desc="candidate-batches", unit="round", disable=not progress_bar)
    for round_index in iterator:
        batches.append(
            generate_raw_candidate_batch(
                grammar,
                config,
                round_index=round_index,
                seed=seed,
                progress_bar=False,
            )
        )
    return batches


def generate_candidate_batches(
    grammar: SyntheticBackoffGrammar,
    patterns: PatternLibrary,
    config: ExperimentConfig,
    *,
    seed: int = 123,
    progress_bar: bool = False,
) -> list[CandidateBatch]:
    raw_batches = generate_raw_candidate_batches(grammar, config, seed=seed, progress_bar=progress_bar)
    return [
        classify_candidate_batch(raw_batch, patterns, round_index=round_index)
        for round_index, raw_batch in enumerate(raw_batches, start=1)
    ]


def apply_publication_policy(
    batch: CandidateBatch,
    policy: PublicationPolicy,
    rng: random.Random,
) -> tuple[list[tuple[int, ...]], list[SequenceRecord]]:
    published_sequences: list[tuple[int, ...]] = []
    published_records: list[SequenceRecord] = []
    for record in batch.records:
        if rng.random() <= policy.publication_probability(record.category):
            published_sequences.append(record.tokens)
            published_records.append(record)
    return published_sequences, published_records


def sample_from_model(
    model: CountBackoffModel,
    patterns: PatternLibrary,
    *,
    count: int,
    sequence_length: int,
    seed: int,
) -> list[SequenceRecord]:
    rng = random.Random(seed)
    return [
        make_sequence_record(*model.generate(sequence_length, rng=rng), patterns=patterns)
        for _ in range(count)
    ]


def greedy_completion(
    model: CountBackoffModel,
    total_length: int,
    *,
    prefix: Sequence[int] | None = None,
) -> list[int]:
    out = [int(x) for x in (prefix or [])]
    while len(out) < total_length:
        top = model.top_tokens(out[-(model.order - 1):], limit=1)
        if not top:
            break
        out.append(int(top[0][0]))
    return out


def backtracking_completion(
    model: CountBackoffModel,
    patterns: PatternLibrary,
    total_length: int,
    *,
    prefix: Sequence[int] | None = None,
    branch_factor: int = 6,
    max_expansions: int = 2500,
) -> list[int] | None:
    start = [int(x) for x in (prefix or [])]
    start_desirable, start_undesirable, _ = patterns.classify(start)
    if start_undesirable:
        return None

    expansions = 0

    def dfs(sequence: list[int], seen_desirable: bool) -> list[int] | None:
        nonlocal expansions
        if expansions >= max_expansions:
            return None
        if len(sequence) >= total_length:
            return list(sequence) if seen_desirable else None
        expansions += 1
        for token, _ in model.top_tokens(sequence[-(model.order - 1):], limit=branch_factor):
            candidate = sequence + [int(token)]
            suffix_desirable, suffix_undesirable = patterns.suffix_flags(candidate)
            if suffix_undesirable:
                continue
            result = dfs(candidate, seen_desirable or suffix_desirable)
            if result is not None:
                return result
        return None

    return dfs(start, bool(start_desirable))


def evaluate_reasoner(
    model: CountBackoffModel,
    patterns: PatternLibrary,
    config: ExperimentConfig,
    prefixes: Sequence[Sequence[int]],
) -> dict[str, float]:
    search_success = 0
    greedy_success = 0
    for prefix in prefixes:
        searched = backtracking_completion(
            model,
            patterns,
            config.sequence_length,
            prefix=prefix,
            branch_factor=config.branch_factor,
            max_expansions=config.max_expansions,
        )
        if searched is not None:
            good_hits, bad_hits, _ = patterns.classify(searched)
            if good_hits > 0 and bad_hits == 0:
                search_success += 1
        greedy = greedy_completion(model, config.sequence_length, prefix=prefix)
        good_hits, bad_hits, _ = patterns.classify(greedy)
        if good_hits > 0 and bad_hits == 0:
            greedy_success += 1
    total = float(len(prefixes) or 1)
    return {
        "search_success_rate": search_success / total,
        "greedy_success_rate": greedy_success / total,
    }


def run_publication_comparison(
    grammar: SyntheticBackoffGrammar,
    patterns: PatternLibrary,
    policies: Sequence[PublicationPolicy],
    config: ExperimentConfig,
    *,
    seed: int = 123,
) -> ComparisonResult:
    candidate_batches = generate_candidate_batches(grammar, patterns, config, seed=seed)
    trajectories: dict[str, PolicyTrajectory] = {}

    for policy_index, policy in enumerate(policies):
        rng = random.Random(seed + 1000 * (policy_index + 1))
        learner = CountBackoffModel(vocab_size=grammar.config.vocab_size, order=config.learner_order)
        published_sequences: list[tuple[int, ...]] = []
        rows: list[dict[str, float | int | str]] = []

        for batch in candidate_batches:
            new_sequences, published_records = apply_publication_policy(batch, policy, rng)
            published_sequences.extend(new_sequences)
            learner.fit_sequences(published_sequences)

            candidate_summary = summarize_records(batch.records)
            published_summary = summarize_records(published_records)
            learner_records = sample_from_model(
                learner,
                patterns,
                count=config.evaluation_count,
                sequence_length=config.sequence_length,
                seed=seed + 10000 * (policy_index + 1) + batch.round_index,
            )
            learner_summary = summarize_records(learner_records)
            prefixes = [record.tokens[:config.prefix_length] for record in batch.records[:config.search_trials]]
            search_summary = evaluate_reasoner(learner, patterns, config, prefixes)

            rows.append(
                {
                    "policy": policy.name,
                    "round": batch.round_index,
                    "candidate_count": int(candidate_summary["count"]),
                    "candidate_desirable_rate": candidate_summary["desirable_rate"],
                    "candidate_undesirable_rate": candidate_summary["undesirable_rate"],
                    "candidate_mixed_rate": candidate_summary["mixed_rate"],
                    "candidate_neutral_rate": candidate_summary["neutral_rate"],
                    "candidate_trigram_rate": candidate_summary["trigram_rate"],
                    "published_count": int(published_summary["count"]),
                    "published_rate": published_summary["count"] / float(max(1, len(batch.records))),
                    "published_desirable_rate": published_summary["desirable_rate"],
                    "published_undesirable_rate": published_summary["undesirable_rate"],
                    "published_mixed_rate": published_summary["mixed_rate"],
                    "published_neutral_rate": published_summary["neutral_rate"],
                    "published_avg_desirable_hits": published_summary["avg_desirable_hits"],
                    "published_avg_undesirable_hits": published_summary["avg_undesirable_hits"],
                    "learner_desirable_rate": learner_summary["desirable_rate"],
                    "learner_undesirable_rate": learner_summary["undesirable_rate"],
                    "learner_mixed_rate": learner_summary["mixed_rate"],
                    "learner_neutral_rate": learner_summary["neutral_rate"],
                    "learner_avg_desirable_hits": learner_summary["avg_desirable_hits"],
                    "learner_avg_undesirable_hits": learner_summary["avg_undesirable_hits"],
                    "learner_unigram_rate": learner_summary["unigram_rate"],
                    "learner_bigram_rate": learner_summary["bigram_rate"],
                    "learner_trigram_rate": learner_summary["trigram_rate"],
                    "cumulative_published": len(published_sequences),
                    "search_success_rate": search_summary["search_success_rate"],
                    "greedy_success_rate": search_summary["greedy_success_rate"],
                }
            )

        trajectories[policy.name] = PolicyTrajectory(
            policy=policy,
            rows=rows,
            published_sequences=published_sequences,
            learner=learner,
        )

    return ComparisonResult(
        patterns=patterns,
        candidate_batches=candidate_batches,
        trajectories=trajectories,
    )


def format_sequence(tokens: Sequence[int], limit: int | None = None) -> str:
    if limit is not None:
        tokens = tokens[:limit]
    return " ".join(str(int(token)) for token in tokens)


def format_pattern(pattern: Sequence[int]) -> str:
    return "(" + ", ".join(str(int(token)) for token in pattern) + ")"


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if path.name == "Drift_and_selection":
            return path
        if (path / "GitHub").exists() and (path / "Nat_Paper").exists():
            return path
    return start


def default_output_root(project_root: Path | None = None) -> Path:
    root = find_project_root(project_root)
    return ensure_dir(root / "GitHub" / "data" / "outputs" / "theorem2_selected_publication")


def ensure_run_paths(
    *,
    output_root: Path,
    run_name: str,
    version: str,
) -> RunPaths:
    slug = stable_slug(f"{version}_{run_name}")
    run_dir = ensure_dir(Path(output_root) / "runs" / slug)
    return RunPaths(
        root_dir=Path(output_root),
        run_dir=run_dir,
        state_dir=ensure_dir(run_dir / "state"),
        figures_dir=ensure_dir(run_dir / "figures"),
        tables_dir=ensure_dir(run_dir / "tables"),
        manifest_path=run_dir / "run_manifest.json",
        checkpoint_path=run_dir / "checkpoint_state.json",
        metrics_partial_path=run_dir / "metrics_partial.csv",
        metrics_final_path=run_dir / "metrics_final.csv",
        summary_path=run_dir / "run_summary.json",
        patterns_path=run_dir / "patterns.json",
        samples_path=run_dir / "sample_sequences.txt",
    )


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


def serialize_patterns(patterns: PatternLibrary) -> dict[str, Any]:
    return {
        "span": int(patterns.span),
        "desirable": [list(pattern) for pattern in patterns.desirable],
        "undesirable": [list(pattern) for pattern in patterns.undesirable],
    }


def deserialize_patterns(payload: dict[str, Any]) -> PatternLibrary:
    return PatternLibrary(
        span=int(payload["span"]),
        desirable=tuple(tuple(int(x) for x in pattern) for pattern in payload["desirable"]),
        undesirable=tuple(tuple(int(x) for x in pattern) for pattern in payload["undesirable"]),
    )


def _policy_state_path(paths: RunPaths, policy_name: str) -> Path:
    return paths.state_dir / f"policy_{stable_slug(policy_name)}.pkl"


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_patterns_table(path: Path, patterns: PatternLibrary) -> None:
    rows = [{"kind": "desirable", "pattern": format_pattern(pattern)} for pattern in patterns.desirable]
    rows.extend({"kind": "undesirable", "pattern": format_pattern(pattern)} for pattern in patterns.undesirable)
    _write_rows_csv(path, rows)


def _title_suffix(manifest: dict[str, Any]) -> str:
    return f"{manifest['version']} | {manifest['run_name']} | {manifest.get('updated_at', manifest['created_at'])}"


def _save_metric_figures(
    paths: RunPaths,
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[str]:
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(dict(row))
    for policy_rows in grouped.values():
        policy_rows.sort(key=lambda item: int(item["round"]))

    saved: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    for policy_name, policy_rows in grouped.items():
        rounds = [int(row["round"]) for row in policy_rows]
        axes[0].plot(rounds, [float(row["learner_desirable_rate"]) for row in policy_rows], marker="o", label=policy_name)
        axes[1].plot(rounds, [float(row["learner_undesirable_rate"]) for row in policy_rows], marker="o", label=policy_name)
        axes[2].plot(rounds, [float(row["search_success_rate"]) for row in policy_rows], marker="o", label=policy_name)
    axes[0].set_title("Learner desirable rate")
    axes[1].set_title("Learner undesirable rate")
    axes[2].set_title("Backtracking success rate")
    for ax in axes:
        ax.set_xlabel("Round")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    fig.suptitle(_title_suffix(manifest))
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    key_metrics_path = paths.figures_dir / f"{stable_slug(manifest['version'])}_key_metrics.png"
    fig.savefig(key_metrics_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(key_metrics_path))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    for policy_name, policy_rows in grouped.items():
        rounds = [int(row["round"]) for row in policy_rows]
        axes[0].plot(rounds, [float(row["published_rate"]) for row in policy_rows], marker="o", label=policy_name)
        axes[1].plot(rounds, [float(row["published_undesirable_rate"]) for row in policy_rows], marker="o", label=policy_name)
        axes[2].plot(rounds, [float(row["cumulative_published"]) for row in policy_rows], marker="o", label=policy_name)
    axes[0].set_title("Publication rate")
    axes[1].set_title("Published undesirable rate")
    axes[2].set_title("Cumulative published")
    for ax in axes:
        ax.set_xlabel("Round")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    axes[0].legend()
    fig.suptitle(_title_suffix(manifest))
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    publication_path = paths.figures_dir / f"{stable_slug(manifest['version'])}_publication_profile.png"
    fig.savefig(publication_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(publication_path))

    return saved


def _write_sample_sequences(
    paths: RunPaths,
    result: ComparisonResult,
    *,
    prefix_length: int,
    sequence_length: int,
    branch_factor: int,
    max_expansions: int,
    sample_seed: int,
) -> None:
    if not result.candidate_batches:
        return
    prefix = result.candidate_batches[-1].records[0].tokens[:prefix_length]
    lines = [f"prefix: {format_sequence(prefix)}", ""]
    for offset, policy_name in enumerate(sorted(result.trajectories)):
        trajectory = result.trajectories[policy_name]
        model = trajectory.learner
        sampled, _ = model.generate(
            sequence_length,
            rng=random.Random(sample_seed + 97 * (offset + 1)),
            prefix=prefix,
        )
        greedy = greedy_completion(model, sequence_length, prefix=prefix)
        searched = backtracking_completion(
            model,
            result.patterns,
            sequence_length,
            prefix=prefix,
            branch_factor=branch_factor,
            max_expansions=max_expansions,
        )
        lines.append(f"[{policy_name}]")
        lines.append(f"sampled: {format_sequence(sampled)}")
        lines.append(f"greedy: {format_sequence(greedy)} | classify={result.patterns.classify(greedy)}")
        if searched is None:
            lines.append("search: no feasible sequence found within budget")
        else:
            lines.append(f"search: {format_sequence(searched)} | classify={result.patterns.classify(searched)}")
        lines.append("")
    paths.samples_path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_artifacts(
    paths: RunPaths,
    manifest: dict[str, Any],
    result: ComparisonResult,
    *,
    final: bool,
) -> None:
    rows = result.table_rows()
    _write_rows_csv(paths.metrics_partial_path, rows)
    if final:
        _write_rows_csv(paths.metrics_final_path, rows)
    _write_patterns_table(paths.tables_dir / "patterns.csv", result.patterns)
    figure_paths = _save_metric_figures(paths, rows, manifest)
    max_candidate_signal = max(
        [0.0]
        + [max(float(row["candidate_desirable_rate"]), float(row["candidate_undesirable_rate"])) for row in rows]
    )
    max_learner_signal = max(
        [0.0]
        + [max(float(row["learner_desirable_rate"]), float(row["learner_undesirable_rate"])) for row in rows]
    )
    signal_warning = None
    if max_candidate_signal < 0.02:
        signal_warning = (
            "Exact r-gram hits are sparse for this configuration. "
            "For stronger signal, reduce r, reduce M, or replace exact r-grams with a larger motif family."
        )
    summary = {
        "run_name": manifest["run_name"],
        "version": manifest["version"],
        "created_at": manifest["created_at"],
        "updated_at": manifest["updated_at"],
        "policies": sorted(result.trajectories.keys()),
        "figure_paths": figure_paths,
        "metrics_partial_path": str(paths.metrics_partial_path),
        "metrics_final_path": str(paths.metrics_final_path if final else paths.metrics_partial_path),
        "patterns_path": str(paths.patterns_path),
        "samples_path": str(paths.samples_path),
        "max_candidate_signal": max_candidate_signal,
        "max_learner_signal": max_learner_signal,
        "signal_warning": signal_warning,
    }
    atomic_save_json(paths.summary_path, _json_ready(summary))


def _build_manifest(
    *,
    run_name: str,
    version: str,
    seed: int,
    grammar_config: SyntheticGrammarConfig,
    experiment_config: ExperimentConfig,
    policies: Sequence[PublicationPolicy],
    patterns: PatternLibrary,
) -> dict[str, Any]:
    now = timestamp()
    return {
        "run_name": run_name,
        "version": version,
        "created_at": now,
        "updated_at": now,
        "seed": int(seed),
        "grammar_config": _json_ready(asdict(grammar_config)),
        "experiment_config": _json_ready(asdict(experiment_config)),
        "policies": [_json_ready(asdict(policy)) for policy in policies],
        "patterns": serialize_patterns(patterns),
        "status": "running",
    }


def _load_or_create_manifest(
    paths: RunPaths,
    *,
    run_name: str,
    version: str,
    seed: int,
    grammar_config: SyntheticGrammarConfig,
    experiment_config: ExperimentConfig,
    policies: Sequence[PublicationPolicy],
    patterns: PatternLibrary,
    resume: bool,
) -> tuple[dict[str, Any], PatternLibrary]:
    if resume and paths.manifest_path.exists():
        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        expected = {
            "run_name": run_name,
            "version": version,
            "seed": int(seed),
            "grammar_config": _json_ready(asdict(grammar_config)),
            "experiment_config": _json_ready(asdict(experiment_config)),
            "policies": [_json_ready(asdict(policy)) for policy in policies],
        }
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                raise ValueError(
                    f"Existing run manifest at {paths.manifest_path} does not match the requested {key}. "
                    "Use FORCE_REBUILD=True or change the run name/version."
                )
        return manifest, deserialize_patterns(manifest["patterns"])
    manifest = _build_manifest(
        run_name=run_name,
        version=version,
        seed=seed,
        grammar_config=grammar_config,
        experiment_config=experiment_config,
        policies=policies,
        patterns=patterns,
    )
    atomic_save_json(paths.manifest_path, _json_ready(manifest))
    atomic_save_json(paths.patterns_path, manifest["patterns"])
    return manifest, patterns


def _remove_tree_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _load_policy_progress(
    paths: RunPaths,
    policy_name: str,
) -> tuple[list[tuple[int, ...]], list[dict[str, Any]], int]:
    state_path = _policy_state_path(paths, policy_name)
    if not state_path.exists():
        return [], [], 1
    state = load_pickle_checkpoint(state_path)
    return (
        [tuple(int(x) for x in seq) for seq in state.get("published_sequences", [])],
        [dict(row) for row in state.get("rows", [])],
        int(state.get("next_round", 1)),
    )


def _save_policy_progress(
    paths: RunPaths,
    policy_name: str,
    *,
    published_sequences: Sequence[Sequence[int]],
    rows: Sequence[dict[str, Any]],
    next_round: int,
) -> None:
    payload = {
        "policy_name": policy_name,
        "published_sequences": [list(seq) for seq in published_sequences],
        "rows": [dict(row) for row in rows],
        "next_round": int(next_round),
        "updated_at": timestamp(),
    }
    save_pickle_checkpoint(_policy_state_path(paths, policy_name), payload)


def run_publication_experiment(
    *,
    version: str,
    run_name: str,
    grammar_config: SyntheticGrammarConfig,
    experiment_config: ExperimentConfig,
    policies: Sequence[PublicationPolicy],
    seed: int = 123,
    patterns: PatternLibrary | None = None,
    output_root: Path | None = None,
    pattern_sampler_kwargs: dict[str, Any] | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
    progress_bar: bool = True,
) -> ExperimentRun:
    output_root = default_output_root(output_root) if output_root is None else ensure_dir(output_root)
    paths = ensure_run_paths(output_root=output_root, run_name=run_name, version=version)
    if force_rebuild:
        _remove_tree_contents(paths.run_dir)
        paths = ensure_run_paths(output_root=output_root, run_name=run_name, version=version)

    grammar = SyntheticBackoffGrammar(grammar_config)
    raw_candidate_batches: list[tuple[GeneratedSequence, ...]] | None = None
    if resume and not force_rebuild and paths.manifest_path.exists() and patterns is None:
        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        patterns = deserialize_patterns(manifest["patterns"])
    if patterns is None:
        raw_candidate_batches = generate_raw_candidate_batches(
            grammar,
            experiment_config,
            seed=seed,
            progress_bar=progress_bar,
        )
        sampler_kwargs = dict(pattern_sampler_kwargs or {})
        pattern_seed = int(sampler_kwargs.pop("seed", seed))
        for unused_key in ("pool_sequences", "sequence_length", "progress_bar"):
            sampler_kwargs.pop(unused_key, None)
        patterns = sample_pattern_library_from_sequences(
            [sequence.tokens for batch in raw_candidate_batches for sequence in batch],
            seed=pattern_seed,
            **sampler_kwargs,
        )

    manifest, patterns = _load_or_create_manifest(
        paths,
        run_name=run_name,
        version=version,
        seed=seed,
        grammar_config=grammar_config,
        experiment_config=experiment_config,
        policies=policies,
        patterns=patterns,
        resume=resume and not force_rebuild,
    )
    grammar = SyntheticBackoffGrammar(grammar_config)
    atomic_save_json(paths.patterns_path, manifest["patterns"])

    if raw_candidate_batches is None:
        candidate_batches = generate_candidate_batches(
            grammar,
            patterns,
            experiment_config,
            seed=seed,
            progress_bar=progress_bar,
        )
    else:
        candidate_batches = [
            classify_candidate_batch(raw_batch, patterns, round_index=round_index)
            for round_index, raw_batch in enumerate(raw_candidate_batches, start=1)
        ]

    trajectories: dict[str, PolicyTrajectory] = {}
    for policy_index, policy in enumerate(policies):
        published_sequences, rows, start_round = _load_policy_progress(paths, policy.name) if (resume and not force_rebuild) else ([], [], 1)
        learner = CountBackoffModel(vocab_size=grammar_config.vocab_size, order=experiment_config.learner_order)
        if published_sequences:
            learner.fit_sequences(published_sequences)

        iterator = range(start_round, experiment_config.rounds + 1)
        iterator = tqdm(
            iterator,
            desc=f"policy:{policy.name}",
            unit="round",
            disable=not progress_bar,
        )
        rng = random.Random(seed + 1000 * (policy_index + 1))
        # Advance the RNG so resumed publication decisions match a fresh run.
        for prior_batch in candidate_batches[: start_round - 1]:
            for _record in prior_batch.records:
                rng.random()

        for round_index in iterator:
            batch = candidate_batches[round_index - 1]
            new_sequences, published_records = apply_publication_policy(batch, policy, rng)
            published_sequences.extend(new_sequences)
            learner.fit_sequences(published_sequences)

            candidate_summary = summarize_records(batch.records)
            published_summary = summarize_records(published_records)
            learner_records = sample_from_model(
                learner,
                patterns,
                count=experiment_config.evaluation_count,
                sequence_length=experiment_config.sequence_length,
                seed=seed + 10000 * (policy_index + 1) + batch.round_index,
            )
            learner_summary = summarize_records(learner_records)
            prefixes = [record.tokens[:experiment_config.prefix_length] for record in batch.records[:experiment_config.search_trials]]
            search_summary = evaluate_reasoner(learner, patterns, experiment_config, prefixes)

            row = {
                "policy": policy.name,
                "round": batch.round_index,
                "candidate_count": int(candidate_summary["count"]),
                "candidate_desirable_rate": candidate_summary["desirable_rate"],
                "candidate_undesirable_rate": candidate_summary["undesirable_rate"],
                "candidate_mixed_rate": candidate_summary["mixed_rate"],
                "candidate_neutral_rate": candidate_summary["neutral_rate"],
                "candidate_trigram_rate": candidate_summary["trigram_rate"],
                "published_count": int(published_summary["count"]),
                "published_rate": published_summary["count"] / float(max(1, len(batch.records))),
                "published_desirable_rate": published_summary["desirable_rate"],
                "published_undesirable_rate": published_summary["undesirable_rate"],
                "published_mixed_rate": published_summary["mixed_rate"],
                "published_neutral_rate": published_summary["neutral_rate"],
                "published_avg_desirable_hits": published_summary["avg_desirable_hits"],
                "published_avg_undesirable_hits": published_summary["avg_undesirable_hits"],
                "learner_desirable_rate": learner_summary["desirable_rate"],
                "learner_undesirable_rate": learner_summary["undesirable_rate"],
                "learner_mixed_rate": learner_summary["mixed_rate"],
                "learner_neutral_rate": learner_summary["neutral_rate"],
                "learner_avg_desirable_hits": learner_summary["avg_desirable_hits"],
                "learner_avg_undesirable_hits": learner_summary["avg_undesirable_hits"],
                "learner_unigram_rate": learner_summary["unigram_rate"],
                "learner_bigram_rate": learner_summary["bigram_rate"],
                "learner_trigram_rate": learner_summary["trigram_rate"],
                "cumulative_published": len(published_sequences),
                "search_success_rate": search_summary["search_success_rate"],
                "greedy_success_rate": search_summary["greedy_success_rate"],
            }
            rows = [existing for existing in rows if int(existing["round"]) != batch.round_index]
            rows.append(row)
            rows.sort(key=lambda item: int(item["round"]))

            _save_policy_progress(
                paths,
                policy.name,
                published_sequences=published_sequences,
                rows=rows,
                next_round=batch.round_index + 1,
            )

            partial_result = ComparisonResult(
                patterns=patterns,
                candidate_batches=candidate_batches,
                trajectories={
                    **trajectories,
                    policy.name: PolicyTrajectory(policy=policy, rows=rows, published_sequences=list(published_sequences), learner=learner),
                },
            )
            manifest["updated_at"] = timestamp()
            manifest["status"] = "running"
            atomic_save_json(paths.manifest_path, _json_ready(manifest))
            atomic_save_json(
                paths.checkpoint_path,
                _json_ready(
                    {
                        "run_name": run_name,
                        "version": version,
                        "status": "running",
                        "current_policy": policy.name,
                        "last_completed_round": batch.round_index,
                        "next_round": batch.round_index + 1,
                        "updated_at": manifest["updated_at"],
                    }
                ),
            )
            _write_run_artifacts(paths, manifest, partial_result, final=False)
            iterator.set_postfix(
                published=int(row["published_count"]),
                desir=f"{row['learner_desirable_rate']:.3f}",
                search=f"{row['search_success_rate']:.3f}",
            )

        trajectories[policy.name] = PolicyTrajectory(
            policy=policy,
            rows=rows,
            published_sequences=list(published_sequences),
            learner=learner,
        )

    result = ComparisonResult(patterns=patterns, candidate_batches=candidate_batches, trajectories=trajectories)
    manifest["updated_at"] = timestamp()
    manifest["status"] = "finished"
    atomic_save_json(paths.manifest_path, _json_ready(manifest))
    atomic_save_json(
        paths.checkpoint_path,
        _json_ready(
            {
                "run_name": run_name,
                "version": version,
                "status": "finished",
                "last_completed_round": experiment_config.rounds,
                "updated_at": manifest["updated_at"],
            }
        ),
    )
    _write_sample_sequences(
        paths,
        result,
        prefix_length=experiment_config.prefix_length,
        sequence_length=experiment_config.sequence_length,
        branch_factor=experiment_config.branch_factor,
        max_expansions=experiment_config.max_expansions,
        sample_seed=seed,
    )
    _write_run_artifacts(paths, manifest, result, final=True)
    return ExperimentRun(version=version, run_name=run_name, paths=paths, manifest=manifest, result=result)
