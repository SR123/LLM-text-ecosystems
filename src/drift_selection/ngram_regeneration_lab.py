from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Sequence

from tqdm.auto import tqdm

from .checkpoints import atomic_save_json, load_pickle_checkpoint, save_pickle_checkpoint
from .theorem1_vocab_drift import CORPUS_REGISTRY, load_clean_corpus
from .theorem2_selected_publication import CountBackoffModel, stable_hash64, stable_u01
from .utils import ensure_dir, stable_slug, timestamp


@dataclass(frozen=True)
class UrtextConfig:
    mode: str = "synthetic_latent"  # synthetic_iid | synthetic_latent | author_corpus
    vocab_size: int = 100
    author_key: str | None = None
    vocab_cap: int | None = None
    lowercase: bool = True
    source_slice: str = "head"  # head | random_block

    def __post_init__(self) -> None:
        valid_modes = {"synthetic_iid", "synthetic_latent", "author_corpus"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.mode == "author_corpus":
            if not self.author_key or self.author_key not in CORPUS_REGISTRY:
                raise ValueError("author_key must be one of the registered corpora for author_corpus mode")
        if self.vocab_cap is not None and self.vocab_cap < 2:
            raise ValueError("vocab_cap must be >= 2 when provided")
        if self.source_slice not in {"head", "random_block"}:
            raise ValueError("source_slice must be 'head' or 'random_block'")


@dataclass(frozen=True)
class LatentGrammarConfig:
    max_order: int = 3
    order_keep_probs: dict[int, float] | None = None
    exact_support: bool = False
    support_cache_limit: int = 4096
    base_weight_seed: int = 11
    support_seed: int = 29
    edge_weight_seed: int = 37

    def __post_init__(self) -> None:
        if self.max_order < 1:
            raise ValueError("max_order must be >= 1")
        if self.support_cache_limit < 1:
            raise ValueError("support_cache_limit must be >= 1")
        probs = self.order_keep_probs or {}
        for order, prob in probs.items():
            if order < 1 or order > self.max_order:
                raise ValueError(f"order_keep_probs contains invalid order {order}")
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"order_keep_probs[{order}] must be in [0, 1]")

    def keep_prob(self, order: int) -> float:
        if order <= 1:
            return 1.0
        probs = self.order_keep_probs or {}
        return float(probs.get(order, 0.0))


@dataclass(frozen=True)
class SelectionConfig:
    mode: str = "none"  # none | hash_rgram | reference_rgram | reference_frequency_rgram
    span: int = 5
    lookahead_samples: int = 64
    desirable_prob: float = 0.25
    undesirable_prob: float = 0.25
    reference_min_count: int = 1
    reference_max_patterns: int | None = None
    reference_unseen_category: str = "neutral"  # neutral | undesirable
    reference_unseen_score: float = 0.0

    def __post_init__(self) -> None:
        valid_modes = {"none", "hash_rgram", "reference_rgram", "reference_frequency_rgram"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}")
        if self.mode != "none" and self.span < 2:
            raise ValueError("span must be >= 2 when selection is enabled")
        if self.lookahead_samples < 1:
            raise ValueError("lookahead_samples must be >= 1")
        if not 0.0 <= self.desirable_prob <= 1.0:
            raise ValueError("desirable_prob must be in [0, 1]")
        if not 0.0 <= self.undesirable_prob <= 1.0:
            raise ValueError("undesirable_prob must be in [0, 1]")
        if self.desirable_prob + self.undesirable_prob > 1.0:
            raise ValueError("desirable_prob + undesirable_prob must be <= 1")
        if self.reference_min_count < 1:
            raise ValueError("reference_min_count must be >= 1")
        if self.reference_max_patterns is not None and self.reference_max_patterns < 1:
            raise ValueError("reference_max_patterns must be >= 1 when provided")
        if self.reference_unseen_category not in {"neutral", "undesirable"}:
            raise ValueError("reference_unseen_category must be 'neutral' or 'undesirable'")


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "sampled_lookahead"  # sampled_lookahead | tree_search
    publish_strategy: str = "desirable_then_random"  # desirable_then_random | best_score | softmax_score
    lookahead_depth: int | None = None
    candidate_trials: int = 64
    branch_factor: int = 6
    max_expansions: int = 2500
    publish_horizon: int = 1
    utility_weight: float = 1.0
    logprob_weight: float = 0.0
    rollback_penalty: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        valid_modes = {"sampled_lookahead", "tree_search"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}")
        valid_strategies = {"desirable_then_random", "best_score", "softmax_score"}
        if self.publish_strategy not in valid_strategies:
            raise ValueError(f"publish_strategy must be one of {sorted(valid_strategies)}")
        if self.lookahead_depth is not None and self.lookahead_depth < 1:
            raise ValueError("lookahead_depth must be >= 1 when provided")
        if self.candidate_trials < 1:
            raise ValueError("candidate_trials must be >= 1")
        if self.branch_factor < 1:
            raise ValueError("branch_factor must be >= 1")
        if self.max_expansions < 1:
            raise ValueError("max_expansions must be >= 1")
        if self.publish_horizon < 1:
            raise ValueError("publish_horizon must be >= 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


@dataclass(frozen=True)
class RegenerationLabConfig:
    text_length: int = 1000
    generations: int = 20
    alpha: float = 0.25
    max_order: int = 3
    restart_probability: float = 0.0
    sample_retained_block: bool = True

    def __post_init__(self) -> None:
        if self.text_length < max(8, self.max_order + 2):
            raise ValueError("text_length is too small for the requested max_order")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.max_order < 1:
            raise ValueError("max_order must be >= 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= self.restart_probability <= 1.0:
            raise ValueError("restart_probability must be in [0, 1]")


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
    sample_text_path: Path


@dataclass
class UrtextBundle:
    token_ids: list[int]
    id_to_label: dict[int, str]
    source_summary: dict[str, Any]


@dataclass
class RegenerationLabRun:
    version: str
    run_name: str
    paths: RunPaths
    manifest: dict[str, Any]
    metrics_rows: list[dict[str, Any]]
    final_tokens: tuple[int, ...]
    id_to_label: dict[int, str]


@dataclass(frozen=True)
class HashRGramUtility:
    span: int = 5
    desirable_prob: float = 0.25
    undesirable_prob: float = 0.25
    seed: int = 123

    def __post_init__(self) -> None:
        if self.span < 1:
            raise ValueError("span must be >= 1")
        if not 0.0 <= self.desirable_prob <= 1.0:
            raise ValueError("desirable_prob must be in [0, 1]")
        if not 0.0 <= self.undesirable_prob <= 1.0:
            raise ValueError("undesirable_prob must be in [0, 1]")
        if self.desirable_prob + self.undesirable_prob > 1.0:
            raise ValueError("desirable_prob + undesirable_prob must be <= 1")

    def classify_window(self, window: Sequence[int]) -> str:
        if len(window) != self.span:
            raise ValueError("window length must equal span")
        score = stable_u01(self.seed, self.span, *[int(token) for token in window])
        if score < self.desirable_prob:
            return "desirable"
        if score < self.desirable_prob + self.undesirable_prob:
            return "undesirable"
        return "neutral"

    def score_window(self, window: Sequence[int]) -> float:
        category = self.classify_window(window)
        if category == "desirable":
            return 1.0
        if category == "undesirable":
            return -1.0
        return 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "hash_rgram",
            "span": int(self.span),
            "desirable_prob": float(self.desirable_prob),
            "undesirable_prob": float(self.undesirable_prob),
            "seed": int(self.seed),
        }


@dataclass
class ReferenceRGramUtility:
    span: int
    desirable: tuple[tuple[int, ...], ...]
    undesirable: tuple[tuple[int, ...], ...] = ()
    label: str = "reference_rgram"
    unseen_category: str = "neutral"

    def __post_init__(self) -> None:
        if self.span < 1:
            raise ValueError("span must be >= 1")
        desirable = tuple(tuple(int(x) for x in gram) for gram in self.desirable)
        undesirable = tuple(tuple(int(x) for x in gram) for gram in self.undesirable)
        if any(len(gram) != self.span for gram in desirable + undesirable):
            raise ValueError("all reference grams must have length equal to span")
        desirable_set = set(desirable)
        undesirable_set = set(undesirable)
        if desirable_set & undesirable_set:
            raise ValueError("desirable and undesirable reference grams must be disjoint")
        if self.unseen_category not in {"neutral", "undesirable"}:
            raise ValueError("unseen_category must be 'neutral' or 'undesirable'")
        self.desirable = desirable
        self.undesirable = undesirable
        self.desirable_set = desirable_set
        self.undesirable_set = undesirable_set

    def classify_window(self, window: Sequence[int]) -> str:
        gram = tuple(int(x) for x in window)
        if len(gram) != self.span:
            raise ValueError("window length must equal span")
        if gram in self.desirable_set:
            return "desirable"
        if gram in self.undesirable_set:
            return "undesirable"
        return self.unseen_category

    def score_window(self, window: Sequence[int]) -> float:
        category = self.classify_window(window)
        if category == "desirable":
            return 1.0
        if category == "undesirable":
            return -1.0
        return 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "reference_rgram",
            "span": int(self.span),
            "desirable_count": int(len(self.desirable)),
            "undesirable_count": int(len(self.undesirable)),
            "label": self.label,
            "unseen_category": self.unseen_category,
        }


@dataclass
class FrequencyRGramUtility:
    span: int
    scores: dict[tuple[int, ...], float]
    unseen_score: float = 0.0
    label: str = "reference_frequency_rgram"

    def __post_init__(self) -> None:
        if self.span < 1:
            raise ValueError("span must be >= 1")
        normalized = {tuple(int(x) for x in gram): float(score) for gram, score in self.scores.items()}
        if any(len(gram) != self.span for gram in normalized):
            raise ValueError("all scored grams must have length equal to span")
        self.scores = normalized

    def score_window(self, window: Sequence[int]) -> float:
        gram = tuple(int(x) for x in window)
        if len(gram) != self.span:
            raise ValueError("window length must equal span")
        return float(self.scores.get(gram, self.unseen_score))

    def classify_window(self, window: Sequence[int]) -> str:
        score = self.score_window(window)
        if score > 0.0:
            return "desirable"
        if score < 0.0:
            return "undesirable"
        return "neutral"

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "reference_frequency_rgram",
            "span": int(self.span),
            "scored_count": int(len(self.scores)),
            "unseen_score": float(self.unseen_score),
            "label": self.label,
        }


@dataclass(frozen=True)
class ExtensionCandidate:
    extension: tuple[int, ...]
    category: str
    utility_score: float
    model_logprob: float
    search_cost: float
    total_score: float
    source: str


class SyntheticOrderGrammar:
    def __init__(self, vocab_size: int, config: LatentGrammarConfig):
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        self.vocab_size = int(vocab_size)
        self.config = config
        self.tokens = tuple(range(1, self.vocab_size + 1))
        self._cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._base_weights = {
            token: 0.25 + stable_u01(config.base_weight_seed, token) + stable_u01(config.base_weight_seed + 1, token)
            for token in self.tokens
        }

    def _clear_cache_if_needed(self) -> None:
        if len(self._cache) >= self.config.support_cache_limit:
            self._cache.clear()

    def _sampled_successors(self, order: int, context: tuple[int, ...], fanout: int) -> tuple[int, ...]:
        if fanout <= 0:
            return tuple()
        if fanout >= self.vocab_size:
            return self.tokens
        seed = stable_hash64(self.config.support_seed, order, *context)
        rng = random.Random(seed)
        return tuple(int(x) for x in rng.sample(range(1, self.vocab_size + 1), fanout))

    def _exact_successors(self, order: int, context: tuple[int, ...], keep_prob: float) -> tuple[int, ...]:
        if keep_prob <= 0.0:
            return tuple()
        return tuple(
            token for token in self.tokens
            if stable_u01(self.config.support_seed, order, *context, token) < keep_prob
        )

    def successors(self, context: Sequence[int]) -> tuple[tuple[int, ...], int]:
        ctx = tuple(int(x) for x in context[-(self.config.max_order - 1):])
        for order in range(min(self.config.max_order, len(ctx) + 1), 1, -1):
            use_context = ctx[-(order - 1):]
            key = (order, *use_context)
            if key not in self._cache:
                keep_prob = self.config.keep_prob(order)
                if self.config.exact_support:
                    successors = self._exact_successors(order, use_context, keep_prob)
                else:
                    fanout = int(round(keep_prob * self.vocab_size))
                    successors = self._sampled_successors(order, use_context, fanout)
                self._clear_cache_if_needed()
                self._cache[key] = successors
            successors = self._cache[key]
            if successors:
                return successors, order
        return self.tokens, 1

    def _edge_weight(self, order: int, context: Sequence[int], token: int) -> float:
        return self._base_weights[int(token)] * (0.5 + stable_u01(self.config.edge_weight_seed, order, *context, int(token)))

    def sample_next(self, context: Sequence[int], rng: random.Random) -> tuple[int, int]:
        successors, order = self.successors(context)
        weights = [self._edge_weight(order, tuple(context[-(order - 1):]), token) for token in successors]
        total = sum(weights) or 1.0
        draw = rng.random() * total
        acc = 0.0
        for token, weight in zip(successors, weights):
            acc += weight
            if draw <= acc:
                return int(token), order
        return int(successors[-1]), order

    def generate(self, length: int, rng: random.Random, prefix: Sequence[int] | None = None) -> list[int]:
        out = [int(x) for x in (prefix or [])]
        while len(out) < length:
            token, _ = self.sample_next(out, rng)
            out.append(token)
        return out


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
    return ensure_dir(root / "GitHub" / "data" / "outputs" / "ngram_regeneration_lab")


def ensure_run_paths(*, output_root: Path, run_name: str, version: str) -> RunPaths:
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
        sample_text_path=run_dir / "sample_texts.txt",
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


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _remove_tree_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def snapshot_path(state_dir: Path, generation: int) -> Path:
    return state_dir / f"tokens_gen_{generation:03d}.pkl"


def save_generation_snapshot(state_dir: Path, generation: int, tokens: Sequence[int]) -> Path:
    path = snapshot_path(state_dir, generation)
    save_pickle_checkpoint(path, [int(token) for token in tokens])
    return path


def load_generation_snapshot(state_dir: Path, generation: int) -> list[int]:
    return [int(token) for token in load_pickle_checkpoint(snapshot_path(state_dir, generation))]


def build_reference_rgram_utility(
    tokens: Sequence[int],
    *,
    span: int,
    min_count: int = 1,
    max_patterns: int | None = None,
    label: str = "urtext_reference",
    unseen_category: str = "neutral",
) -> ReferenceRGramUtility:
    if span < 1:
        raise ValueError("span must be >= 1")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    counts: dict[tuple[int, ...], int] = {}
    seq = [int(token) for token in tokens]
    if len(seq) < span:
        raise ValueError("tokens are too short for the requested reference span")
    for i in range(len(seq) - span + 1):
        gram = tuple(seq[i:i + span])
        counts[gram] = counts.get(gram, 0) + 1
    ranked = [
        gram for gram, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ]
    if max_patterns is not None:
        ranked = ranked[:max_patterns]
    if not ranked:
        raise ValueError("reference utility did not find any grams meeting the requested threshold")
    return ReferenceRGramUtility(
        span=span,
        desirable=tuple(ranked),
        label=label,
        unseen_category=unseen_category,
    )


def build_frequency_rgram_utility(
    tokens: Sequence[int],
    *,
    span: int,
    min_count: int = 1,
    max_patterns: int | None = None,
    label: str = "urtext_frequency",
    unseen_score: float = 0.0,
) -> FrequencyRGramUtility:
    if span < 1:
        raise ValueError("span must be >= 1")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    counts: dict[tuple[int, ...], int] = {}
    seq = [int(token) for token in tokens]
    if len(seq) < span:
        raise ValueError("tokens are too short for the requested frequency span")
    for i in range(len(seq) - span + 1):
        gram = tuple(seq[i:i + span])
        counts[gram] = counts.get(gram, 0) + 1
    items = [
        (gram, count) for gram, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ]
    if max_patterns is not None:
        items = items[:max_patterns]
    if not items:
        raise ValueError("frequency utility did not find any grams meeting the requested threshold")
    max_log = max(math.log1p(count) for _, count in items)
    scores = {
        gram: (math.log1p(count) / max_log if max_log > 0.0 else 0.0)
        for gram, count in items
    }
    return FrequencyRGramUtility(
        span=span,
        scores=scores,
        unseen_score=unseen_score,
        label=label,
    )


def default_agent_config(selection_config: SelectionConfig | None) -> AgentConfig:
    trials = 64
    if selection_config is not None:
        trials = int(selection_config.lookahead_samples)
    return AgentConfig(candidate_trials=trials)


def build_selection_utility(
    selection_config: SelectionConfig | None,
    *,
    urtext_tokens: Sequence[int],
    seed: int,
) -> tuple[HashRGramUtility | ReferenceRGramUtility | FrequencyRGramUtility | None, dict[str, Any]]:
    if selection_config is None or selection_config.mode == "none":
        return None, {"mode": "none"}
    if selection_config.mode == "hash_rgram":
        utility = HashRGramUtility(
            span=selection_config.span,
            desirable_prob=selection_config.desirable_prob,
            undesirable_prob=selection_config.undesirable_prob,
            seed=seed,
        )
        return utility, utility.describe()
    if selection_config.mode == "reference_rgram":
        utility = build_reference_rgram_utility(
            urtext_tokens,
            span=selection_config.span,
            min_count=selection_config.reference_min_count,
            max_patterns=selection_config.reference_max_patterns,
            label="urtext_reference",
            unseen_category=selection_config.reference_unseen_category,
        )
        return utility, utility.describe()
    utility = build_frequency_rgram_utility(
        urtext_tokens,
        span=selection_config.span,
        min_count=selection_config.reference_min_count,
        max_patterns=selection_config.reference_max_patterns,
        label="urtext_frequency",
        unseen_score=selection_config.reference_unseen_score,
    )
    return utility, utility.describe()


def _build_author_bundle(config: UrtextConfig, text_length: int, seed: int) -> UrtextBundle:
    sentences = load_clean_corpus(config.author_key, lowercase=config.lowercase, progress_bar=False)
    tokens = [token for sentence in sentences for token in sentence]
    if not tokens:
        raise ValueError(f"No tokens found for author {config.author_key}")
    if config.vocab_cap is not None and config.vocab_cap > 1:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        top_vocab = {
            token
            for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:config.vocab_cap]
        }
        mapped_tokens = [token if token in top_vocab else "<UNK>" for token in tokens]
    else:
        mapped_tokens = list(tokens)
    if len(mapped_tokens) >= text_length:
        if config.source_slice == "random_block":
            rng = random.Random(seed)
            start = rng.randint(0, len(mapped_tokens) - text_length)
            mapped_tokens = mapped_tokens[start:start + text_length]
        else:
            mapped_tokens = mapped_tokens[:text_length]
    else:
        repeats = (text_length + len(mapped_tokens) - 1) // len(mapped_tokens)
        mapped_tokens = (mapped_tokens * repeats)[:text_length]
    vocab = sorted(set(mapped_tokens))
    label_to_id = {label: idx + 1 for idx, label in enumerate(vocab)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return UrtextBundle(
        token_ids=[label_to_id[token] for token in mapped_tokens],
        id_to_label=id_to_label,
        source_summary={
            "mode": "author_corpus",
            "author_key": config.author_key,
            "actual_vocab_size": len(vocab),
            "vocab_cap": config.vocab_cap,
            "source_slice": config.source_slice,
        },
    )


def build_urtext(
    urtext_config: UrtextConfig,
    latent_config: LatentGrammarConfig,
    text_length: int,
    *,
    seed: int,
) -> UrtextBundle:
    if urtext_config.mode == "synthetic_latent" and latent_config.max_order < 1:
        raise ValueError("latent_config.max_order must be >= 1")
    rng = random.Random(seed)
    if urtext_config.mode == "synthetic_iid":
        labels = {idx: str(idx) for idx in range(1, urtext_config.vocab_size + 1)}
        token_ids = [rng.randint(1, urtext_config.vocab_size) for _ in range(text_length)]
        return UrtextBundle(
            token_ids=token_ids,
            id_to_label=labels,
            source_summary={"mode": "synthetic_iid", "vocab_size": urtext_config.vocab_size},
        )
    if urtext_config.mode == "synthetic_latent":
        grammar = SyntheticOrderGrammar(urtext_config.vocab_size, latent_config)
        token_ids = grammar.generate(text_length, rng=rng)
        labels = {idx: str(idx) for idx in range(1, urtext_config.vocab_size + 1)}
        return UrtextBundle(
            token_ids=token_ids,
            id_to_label=labels,
            source_summary={
                "mode": "synthetic_latent",
                "vocab_size": urtext_config.vocab_size,
                "max_order": latent_config.max_order,
            },
        )
    return _build_author_bundle(urtext_config, text_length, seed)


def distinct_ngram_count(tokens: Sequence[int], order: int) -> int:
    if order == 1:
        return len(set(int(token) for token in tokens))
    if len(tokens) < order:
        return 0
    return len({tuple(int(x) for x in tokens[i:i + order]) for i in range(len(tokens) - order + 1)})


def token_entropy_bits(tokens: Sequence[int]) -> float:
    if not tokens:
        return 0.0
    counts: dict[int, int] = {}
    for token in tokens:
        counts[int(token)] = counts.get(int(token), 0) + 1
    total = float(len(tokens))
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def evaluate_window(
    utility: HashRGramUtility | ReferenceRGramUtility | FrequencyRGramUtility,
    window: Sequence[int],
) -> tuple[float, str]:
    score = float(utility.score_window(window))
    category = utility.classify_window(window)
    return score, category


def finalize_agent_stats(raw: dict[str, float | int]) -> dict[str, float | int]:
    decisions = int(raw.get("agent_decisions", 0))
    if decisions <= 0:
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
            "agent_tokens_published": int(raw.get("agent_tokens_published", 0)),
        }
    desirable = int(raw.get("agent_desirable_choices", 0))
    neutral = int(raw.get("agent_neutral_choices", 0))
    undesirable = int(raw.get("agent_undesirable_choices", 0))
    return {
        "agent_decisions": decisions,
        "agent_desirable_choices": desirable,
        "agent_neutral_choices": neutral,
        "agent_undesirable_choices": undesirable,
        "agent_desirable_rate": desirable / float(decisions),
        "agent_neutral_rate": neutral / float(decisions),
        "agent_undesirable_rate": undesirable / float(decisions),
        "agent_avg_utility_score": float(raw.get("agent_selected_utility_score_sum", 0.0)) / float(decisions),
        "agent_avg_total_score": float(raw.get("agent_selected_total_score_sum", 0.0)) / float(decisions),
        "agent_avg_model_logprob": float(raw.get("agent_selected_logprob_sum", 0.0)) / float(decisions),
        "agent_avg_search_cost": float(raw.get("agent_selected_search_cost_sum", 0.0)) / float(decisions),
        "agent_tokens_published": int(raw.get("agent_tokens_published", 0)),
    }


def utility_window_counts(
    tokens: Sequence[int],
    utility: HashRGramUtility | ReferenceRGramUtility | FrequencyRGramUtility | None,
) -> dict[str, float | int]:
    if utility is None:
        return {}
    total = max(0, len(tokens) - utility.span + 1)
    desirable = 0
    undesirable = 0
    neutral = 0
    for i in range(total):
        category = utility.classify_window(tokens[i:i + utility.span])
        if category == "desirable":
            desirable += 1
        elif category == "undesirable":
            undesirable += 1
        else:
            neutral += 1
    return {
        "utility_window_total": int(total),
        "desirable_windows": int(desirable),
        "undesirable_windows": int(undesirable),
        "neutral_windows": int(neutral),
        "desirable_window_share": float(desirable / total) if total else 0.0,
        "undesirable_window_share": float(undesirable / total) if total else 0.0,
        "neutral_window_share": float(neutral / total) if total else 0.0,
    }


def metrics_for_generation(
    tokens: Sequence[int],
    config: RegenerationLabConfig,
    *,
    generation: int,
    baseline: dict[str, float] | None = None,
    utility: HashRGramUtility | ReferenceRGramUtility | FrequencyRGramUtility | None = None,
    agent_stats: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "generation": int(generation),
        "text_length": int(len(tokens)),
        "alpha": float(config.alpha),
        "max_order": int(config.max_order),
        "token_entropy_bits": float(token_entropy_bits(tokens)),
    }
    for order in range(1, config.max_order + 1):
        row[f"distinct_{order}grams"] = int(distinct_ngram_count(tokens, order))
    row["vocab_size"] = int(row["distinct_1grams"])
    if baseline:
        for order in range(1, config.max_order + 1):
            key = f"distinct_{order}grams"
            row[f"{key}_ratio_vs_gen0"] = float(row[key]) / float(baseline[key] or 1.0)
    else:
        for order in range(1, config.max_order + 1):
            row[f"distinct_{order}grams_ratio_vs_gen0"] = 1.0
    row["vocab_ratio_vs_gen0"] = row["distinct_1grams_ratio_vs_gen0"]
    row.update(utility_window_counts(tokens, utility))
    if agent_stats:
        row.update(agent_stats)
    return row


def sample_retained_tokens(
    tokens: Sequence[int],
    retain_length: int,
    rng: random.Random,
    contiguous: bool = True,
) -> list[int]:
    retain_length = max(0, min(int(retain_length), len(tokens)))
    if retain_length == 0:
        return []
    if retain_length == len(tokens):
        return [int(token) for token in tokens]
    if contiguous:
        start = rng.randint(0, len(tokens) - retain_length)
        return [int(token) for token in tokens[start:start + retain_length]]
    indices = sorted(rng.sample(range(len(tokens)), retain_length))
    return [int(tokens[idx]) for idx in indices]


def build_empirical_model(tokens: Sequence[int], max_order: int) -> CountBackoffModel:
    model = CountBackoffModel(vocab_size=max(int(max(tokens, default=1)), 1), order=max_order)
    model.fit_sequences([tuple(int(token) for token in tokens)])
    return model


def random_seed_ngram(tokens: Sequence[int], seed_length: int, rng: random.Random) -> list[int]:
    seq = [int(token) for token in tokens]
    if len(seq) <= seed_length:
        return list(seq)
    start = rng.randint(0, len(seq) - seed_length)
    return seq[start:start + seed_length]


def _sample_from_distribution(dist: dict[int, float], rng: random.Random) -> tuple[int, float]:
    items = list(dist.items())
    tokens = [int(token) for token, _ in items]
    weights = [float(weight) for _, weight in items]
    total = sum(weights)
    if total <= 0.0:
        token = int(rng.choice(tokens))
        return token, 1.0 / float(len(tokens))
    draw = rng.random() * total
    acc = 0.0
    for token, weight in zip(tokens, weights):
        acc += weight
        if draw <= acc:
            return int(token), float(weight)
    return int(tokens[-1]), float(weights[-1])


def sample_lookahead_candidates(
    model: CountBackoffModel,
    prefix: Sequence[int],
    *,
    depth: int,
    n_samples: int,
    rng: random.Random,
) -> list[tuple[list[int], float]]:
    samples: list[tuple[list[int], float]] = []
    for _ in range(max(1, n_samples)):
        seq = [int(x) for x in prefix]
        logprob = 0.0
        while len(seq) < len(prefix) + depth:
            ctx = seq[-(model.order - 1):] if model.order > 1 else tuple()
            dist, _ = model.distribution(ctx)
            token, prob = _sample_from_distribution(dist, rng)
            logprob += math.log(max(prob, 1e-12))
            seq.append(int(token))
        samples.append((seq[len(prefix):], logprob))
    return samples


def tree_search_candidates(
    model: CountBackoffModel,
    prefix: Sequence[int],
    *,
    depth: int,
    branch_factor: int,
    max_expansions: int,
) -> list[tuple[list[int], float, float]]:
    candidates: list[tuple[list[int], float, float]] = []
    expansions = 0

    def dfs(current_prefix: list[int], extension: list[int], logprob: float, rollback_cost: float) -> None:
        nonlocal expansions
        if len(extension) >= depth or expansions >= max_expansions:
            if extension:
                candidates.append((list(extension), float(logprob), float(rollback_cost)))
            return
        ctx = current_prefix[-(model.order - 1):] if model.order > 1 else tuple()
        dist, _ = model.distribution(ctx)
        ranked = sorted(dist.items(), key=lambda item: item[1], reverse=True)[:branch_factor]
        if not ranked:
            return
        for rank, (token, prob) in enumerate(ranked):
            if expansions >= max_expansions:
                break
            expansions += 1
            next_prefix = current_prefix + [int(token)]
            dfs(
                next_prefix,
                extension + [int(token)],
                logprob + math.log(max(float(prob), 1e-12)),
                rollback_cost + float(rank),
            )

    dfs([int(x) for x in prefix], [], 0.0, 0.0)
    return candidates


def choose_agent_extension(
    model: CountBackoffModel,
    prefix: Sequence[int],
    utility: HashRGramUtility | ReferenceRGramUtility | FrequencyRGramUtility,
    *,
    agent_config: AgentConfig,
    rng: random.Random,
) -> ExtensionCandidate:
    depth = int(agent_config.lookahead_depth or utility.span)
    if depth < utility.span:
        depth = utility.span
    if agent_config.mode == "tree_search":
        raw_candidates = [
            (extension, logprob, search_cost, "tree_search")
            for extension, logprob, search_cost in tree_search_candidates(
                model,
                prefix,
                depth=depth,
                branch_factor=agent_config.branch_factor,
                max_expansions=agent_config.max_expansions,
            )
            if len(extension) >= utility.span
        ]
    else:
        raw_candidates = [
            (extension, logprob, 0.0, "sampled_lookahead")
            for extension, logprob in sample_lookahead_candidates(
                model,
                prefix,
                depth=depth,
                n_samples=agent_config.candidate_trials,
                rng=rng,
            )
            if len(extension) >= utility.span
        ]
    if not raw_candidates:
        ctx = prefix[-(model.order - 1):] if model.order > 1 else tuple()
        dist, _ = model.distribution(ctx)
        token, prob = _sample_from_distribution(dist, rng)
        score, category = evaluate_window(utility, [token] * utility.span)
        total = agent_config.utility_weight * score + agent_config.logprob_weight * math.log(max(prob, 1e-12))
        return ExtensionCandidate(
            extension=(int(token),),
            category=category,
            utility_score=float(score),
            model_logprob=float(math.log(max(prob, 1e-12))),
            search_cost=0.0,
            total_score=float(total),
            source="fallback_one_step",
        )

    candidates: list[ExtensionCandidate] = []
    for extension, logprob, search_cost, source in raw_candidates:
        utility_score, category = evaluate_window(utility, extension[:utility.span])
        total_score = (
            agent_config.utility_weight * utility_score
            + agent_config.logprob_weight * float(logprob)
            - agent_config.rollback_penalty * float(search_cost)
        )
        candidates.append(
            ExtensionCandidate(
                extension=tuple(int(x) for x in extension),
                category=category,
                utility_score=float(utility_score),
                model_logprob=float(logprob),
                search_cost=float(search_cost),
                total_score=float(total_score),
                source=source,
            )
        )

    if agent_config.publish_strategy == "desirable_then_random":
        desirable = [cand for cand in candidates if cand.category == "desirable"]
        if desirable:
            return rng.choice(desirable)
        neutral = [cand for cand in candidates if cand.category == "neutral"]
        if neutral:
            return rng.choice(neutral)
        return rng.choice(candidates)

    if agent_config.publish_strategy == "softmax_score":
        scaled = [cand.total_score / agent_config.temperature for cand in candidates]
        max_scaled = max(scaled)
        weights = [math.exp(value - max_scaled) for value in scaled]
        total = sum(weights) or 1.0
        draw = rng.random() * total
        acc = 0.0
        for candidate, weight in zip(candidates, weights):
            acc += weight
            if draw <= acc:
                return candidate
        return candidates[-1]

    return max(candidates, key=lambda cand: (cand.total_score, cand.utility_score, cand.model_logprob))


def generate_replacement_tokens(
    tokens: Sequence[int],
    config: RegenerationLabConfig,
    *,
    rng: random.Random,
    selection_utility: HashRGramUtility | ReferenceRGramUtility | FrequencyRGramUtility | None = None,
    selection_config: SelectionConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> tuple[list[int], dict[str, float | int]]:
    replacement_length = int(round(config.alpha * config.text_length))
    if replacement_length <= 0:
        return [], {}
    seq = [int(token) for token in tokens]
    model = build_empirical_model(seq, config.max_order)
    seed_length = min(config.max_order, replacement_length)
    out = random_seed_ngram(seq, seed_length, rng)
    raw_agent_stats: dict[str, float | int] = {
        "agent_decisions": 0,
        "agent_desirable_choices": 0,
        "agent_neutral_choices": 0,
        "agent_undesirable_choices": 0,
        "agent_selected_utility_score_sum": 0.0,
        "agent_selected_total_score_sum": 0.0,
        "agent_selected_logprob_sum": 0.0,
        "agent_selected_search_cost_sum": 0.0,
        "agent_tokens_published": 0,
    }
    active_agent = agent_config or default_agent_config(selection_config)
    while len(out) < replacement_length:
        if config.restart_probability > 0.0 and len(out) >= seed_length and rng.random() < config.restart_probability:
            remaining = replacement_length - len(out)
            out.extend(random_seed_ngram(seq, min(config.max_order, remaining), rng))
            continue
        remaining = replacement_length - len(out)
        if (
            selection_utility is not None
            and selection_config is not None
            and selection_config.mode != "none"
            and remaining >= selection_utility.span
        ):
            candidate = choose_agent_extension(
                model,
                out,
                selection_utility,
                agent_config=active_agent,
                rng=rng,
            )
            append_count = min(int(active_agent.publish_horizon), remaining, len(candidate.extension))
            out.extend(int(token) for token in candidate.extension[:append_count])
            raw_agent_stats["agent_decisions"] = int(raw_agent_stats["agent_decisions"]) + 1
            raw_agent_stats[f"agent_{candidate.category}_choices"] = int(raw_agent_stats[f"agent_{candidate.category}_choices"]) + 1
            raw_agent_stats["agent_selected_utility_score_sum"] = float(raw_agent_stats["agent_selected_utility_score_sum"]) + float(candidate.utility_score)
            raw_agent_stats["agent_selected_total_score_sum"] = float(raw_agent_stats["agent_selected_total_score_sum"]) + float(candidate.total_score)
            raw_agent_stats["agent_selected_logprob_sum"] = float(raw_agent_stats["agent_selected_logprob_sum"]) + float(candidate.model_logprob)
            raw_agent_stats["agent_selected_search_cost_sum"] = float(raw_agent_stats["agent_selected_search_cost_sum"]) + float(candidate.search_cost)
            raw_agent_stats["agent_tokens_published"] = int(raw_agent_stats["agent_tokens_published"]) + int(append_count)
            continue
        ctx = out[-(config.max_order - 1):] if config.max_order > 1 else tuple()
        token, _ = model.sample_next(ctx, rng=rng)
        out.append(int(token))
    return out[:replacement_length], finalize_agent_stats(raw_agent_stats)


def next_generation_text(
    tokens: Sequence[int],
    config: RegenerationLabConfig,
    *,
    rng: random.Random,
    selection_utility: HashRGramUtility | ReferenceRGramUtility | FrequencyRGramUtility | None = None,
    selection_config: SelectionConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> tuple[list[int], dict[str, float | int]]:
    retain_length = int(round((1.0 - config.alpha) * config.text_length))
    retained = sample_retained_tokens(tokens, retain_length, rng, contiguous=config.sample_retained_block)
    generated, agent_stats = generate_replacement_tokens(
        tokens,
        config,
        rng=rng,
        selection_utility=selection_utility,
        selection_config=selection_config,
        agent_config=agent_config,
    )
    combined = retained + generated
    if len(combined) < config.text_length:
        pad_cfg = RegenerationLabConfig(
            text_length=config.text_length - len(combined),
            generations=1,
            alpha=1.0,
            max_order=config.max_order,
            restart_probability=config.restart_probability,
            sample_retained_block=config.sample_retained_block,
        )
        pad_tokens, _ = generate_replacement_tokens(
            tokens,
            pad_cfg,
            rng=rng,
            selection_utility=selection_utility,
            selection_config=selection_config,
            agent_config=agent_config,
        )
        combined.extend(pad_tokens)
    return [int(token) for token in combined[:config.text_length]], agent_stats


def _title_suffix(manifest: dict[str, Any]) -> str:
    return f"{manifest['version']} | {manifest['run_name']} | {manifest.get('updated_at', manifest['created_at'])}"


def _save_metric_figures(
    paths: RunPaths,
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    max_order: int,
) -> list[str]:
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    generations = [int(row["generation"]) for row in rows]
    saved: list[str] = []

    fig, axes = plt.subplots(1, max_order, figsize=(5 * max_order, 4), sharex=True)
    if max_order == 1:
        axes = [axes]
    for order, ax in enumerate(axes, start=1):
        ax.plot(generations, [float(row[f"distinct_{order}grams"]) for row in rows], marker="o")
        ax.set_title(f"Distinct {order}-grams")
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Count")
    fig.suptitle(_title_suffix(manifest))
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    counts_path = paths.figures_dir / f"{stable_slug(manifest['version'])}_support_counts.png"
    fig.savefig(counts_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(counts_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for order in range(1, max_order + 1):
        axes[0].plot(generations, [float(row[f"distinct_{order}grams_ratio_vs_gen0"]) for row in rows], marker="o")
    axes[0].legend([f"{order}-grams/gen0" for order in range(1, max_order + 1)])
    axes[0].set_title("Relative support size")
    axes[1].plot(generations, [float(row["token_entropy_bits"]) for row in rows], marker="o")
    axes[1].set_title("Token entropy")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Ratio")
    axes[1].set_ylabel("Bits")
    fig.suptitle(_title_suffix(manifest))
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    ratio_path = paths.figures_dir / f"{stable_slug(manifest['version'])}_relative_support.png"
    fig.savefig(ratio_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(ratio_path))

    if "desirable_windows" in rows[0]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        axes[0].plot(generations, [float(row["desirable_windows"]) for row in rows], marker="o")
        axes[0].plot(generations, [float(row["neutral_windows"]) for row in rows], marker="o")
        axes[0].plot(generations, [float(row["undesirable_windows"]) for row in rows], marker="o")
        axes[0].legend(["desirable", "neutral", "undesirable"])
        axes[0].set_title("Utility window counts")
        axes[1].plot(generations, [float(row["desirable_window_share"]) for row in rows], marker="o")
        axes[1].plot(generations, [float(row["neutral_window_share"]) for row in rows], marker="o")
        axes[1].plot(generations, [float(row["undesirable_window_share"]) for row in rows], marker="o")
        axes[1].legend(["desirable", "neutral", "undesirable"])
        axes[1].set_title("Utility window shares")
        for ax in axes:
            ax.set_xlabel("Generation")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Count")
        axes[1].set_ylabel("Share")
        fig.suptitle(_title_suffix(manifest))
        fig.tight_layout()
        fig.subplots_adjust(top=0.82)
        utility_path = paths.figures_dir / f"{stable_slug(manifest['version'])}_utility_windows.png"
        fig.savefig(utility_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(utility_path))

    if "agent_decisions" in rows[0]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        axes[0].plot(generations, [float(row["agent_desirable_rate"]) for row in rows], marker="o")
        axes[0].plot(generations, [float(row["agent_neutral_rate"]) for row in rows], marker="o")
        axes[0].plot(generations, [float(row["agent_undesirable_rate"]) for row in rows], marker="o")
        axes[0].legend(["desirable", "neutral", "undesirable"])
        axes[0].set_title("Agent publication mix")
        axes[1].plot(generations, [float(row["agent_avg_total_score"]) for row in rows], marker="o")
        axes[1].plot(generations, [float(row["agent_avg_search_cost"]) for row in rows], marker="o")
        axes[1].legend(["avg total score", "avg search cost"])
        axes[1].set_title("Agent search summary")
        for ax in axes:
            ax.set_xlabel("Generation")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Rate")
        axes[1].set_ylabel("Value")
        fig.suptitle(_title_suffix(manifest))
        fig.tight_layout()
        fig.subplots_adjust(top=0.82)
        agent_path = paths.figures_dir / f"{stable_slug(manifest['version'])}_agent_policy.png"
        fig.savefig(agent_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(agent_path))

    return saved


def _decode_tokens(tokens: Sequence[int], id_to_label: dict[int, str], limit: int = 80) -> str:
    return " ".join(id_to_label.get(int(token), str(int(token))) for token in tokens[:limit])


def _write_sample_texts(paths: RunPaths, snapshots: dict[int, Sequence[int]], id_to_label: dict[int, str]) -> None:
    generations = sorted(snapshots)
    picks = []
    if generations:
        picks.append(generations[0])
        if len(generations) > 2:
            picks.append(generations[len(generations) // 2])
        if len(generations) > 1:
            picks.append(generations[-1])
    unique_picks = []
    for generation in picks:
        if generation not in unique_picks:
            unique_picks.append(generation)
    lines: list[str] = []
    for generation in unique_picks:
        lines.append(f"[generation {generation}]")
        lines.append(_decode_tokens(snapshots[generation], id_to_label))
        lines.append("")
    paths.sample_text_path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_artifacts(
    paths: RunPaths,
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    snapshots: dict[int, Sequence[int]],
    id_to_label: dict[int, str],
    *,
    final: bool,
    max_order: int,
) -> None:
    _write_rows_csv(paths.metrics_partial_path, rows)
    if final:
        _write_rows_csv(paths.metrics_final_path, rows)
    figure_paths = _save_metric_figures(paths, rows, manifest, max_order=max_order)
    _write_sample_texts(paths, snapshots, id_to_label)
    summary = {
        "run_name": manifest["run_name"],
        "version": manifest["version"],
        "created_at": manifest["created_at"],
        "updated_at": manifest["updated_at"],
        "figure_paths": figure_paths,
        "metrics_partial_path": str(paths.metrics_partial_path),
        "metrics_final_path": str(paths.metrics_final_path if final else paths.metrics_partial_path),
        "sample_text_path": str(paths.sample_text_path),
        "final_generation": max(int(row["generation"]) for row in rows) if rows else 0,
        "urtext_source": manifest["urtext_config"]["mode"],
        "selection_summary": manifest.get("selection_summary", {"mode": "none"}),
        "agent_config": manifest.get("agent_config"),
    }
    atomic_save_json(paths.summary_path, _json_ready(summary))


def _parse_metric_value(key: str, value: str) -> int | float:
    if value == "" or value is None:
        if key in {"generation", "text_length", "max_order", "vocab_size", "utility_window_total", "agent_decisions", "agent_tokens_published"}:
            return 0
        if key.startswith("distinct_") and key.endswith("grams"):
            return 0
        if key.endswith("_windows"):
            return 0
        if key.endswith("_choices"):
            return 0
        return 0.0
    if key in {"generation", "text_length", "max_order", "vocab_size", "utility_window_total", "agent_decisions", "agent_tokens_published"}:
        return int(float(value))
    if key.startswith("distinct_") and key.endswith("grams"):
        return int(float(value))
    if key.endswith("_windows"):
        return int(float(value))
    if key.endswith("_choices"):
        return int(float(value))
    return float(value)


def _load_metrics_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                parsed[key] = _parse_metric_value(key, value)
            rows.append(parsed)
    return rows


def _baseline_metrics(rows: Sequence[dict[str, Any]], max_order: int) -> dict[str, float]:
    first = min(rows, key=lambda row: int(row["generation"]))
    return {f"distinct_{order}grams": float(first[f"distinct_{order}grams"]) for order in range(1, max_order + 1)}


def _build_manifest(
    run_name: str,
    version: str,
    seed: int,
    urtext_config: UrtextConfig,
    latent_config: LatentGrammarConfig,
    lab_config: RegenerationLabConfig,
    selection_config: SelectionConfig | None,
    agent_config: AgentConfig | None,
    bundle: UrtextBundle,
    selection_summary: dict[str, Any],
) -> dict[str, Any]:
    now = timestamp()
    return {
        "run_name": run_name,
        "version": version,
        "seed": int(seed),
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "urtext_config": _json_ready(asdict(urtext_config)),
        "latent_config": _json_ready(asdict(latent_config)),
        "lab_config": _json_ready(asdict(lab_config)),
        "selection_config": _json_ready(asdict(selection_config)) if selection_config is not None else None,
        "agent_config": _json_ready(asdict(agent_config)) if agent_config is not None else None,
        "selection_summary": _json_ready(selection_summary),
        "source_summary": _json_ready(bundle.source_summary),
        "id_to_label": _json_ready(bundle.id_to_label),
    }


def _load_or_create_manifest(
    paths: RunPaths,
    *,
    run_name: str,
    version: str,
    seed: int,
    urtext_config: UrtextConfig,
    latent_config: LatentGrammarConfig,
    lab_config: RegenerationLabConfig,
    selection_config: SelectionConfig | None,
    agent_config: AgentConfig | None,
    bundle: UrtextBundle,
    selection_summary: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    expected = {
        "run_name": run_name,
        "version": version,
        "seed": int(seed),
        "urtext_config": _json_ready(asdict(urtext_config)),
        "latent_config": _json_ready(asdict(latent_config)),
        "lab_config": _json_ready(asdict(lab_config)),
        "selection_config": _json_ready(asdict(selection_config)) if selection_config is not None else None,
        "agent_config": _json_ready(asdict(agent_config)) if agent_config is not None else None,
    }
    if resume and paths.manifest_path.exists():
        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                raise ValueError(
                    f"Existing run manifest at {paths.manifest_path} does not match the requested {key}. "
                    "Use FORCE_REBUILD=True or change the run name/version."
                )
        return manifest
    manifest = _build_manifest(
        run_name,
        version,
        seed,
        urtext_config,
        latent_config,
        lab_config,
        selection_config,
        agent_config,
        bundle,
        selection_summary,
    )
    atomic_save_json(paths.manifest_path, _json_ready(manifest))
    return manifest


def run_ngram_regeneration_lab(
    *,
    version: str,
    run_name: str,
    urtext_config: UrtextConfig,
    latent_config: LatentGrammarConfig,
    lab_config: RegenerationLabConfig,
    selection_config: SelectionConfig | None = None,
    agent_config: AgentConfig | None = None,
    seed: int = 123,
    output_root: Path | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
    progress_bar: bool = True,
) -> RegenerationLabRun:
    if urtext_config.mode == "synthetic_latent" and latent_config.max_order < lab_config.max_order:
        raise ValueError("latent_config.max_order must be >= lab_config.max_order for synthetic_latent urtexts")

    effective_agent_config = (
        agent_config
        if (selection_config is not None and selection_config.mode != "none")
        else None
    )
    if effective_agent_config is None and selection_config is not None and selection_config.mode != "none":
        effective_agent_config = default_agent_config(selection_config)

    output_root = default_output_root(output_root) if output_root is None else ensure_dir(output_root)
    paths = ensure_run_paths(output_root=output_root, run_name=run_name, version=version)
    if force_rebuild:
        _remove_tree_contents(paths.run_dir)
        paths = ensure_run_paths(output_root=output_root, run_name=run_name, version=version)

    bundle = build_urtext(urtext_config, latent_config, lab_config.text_length, seed=seed)
    selection_utility, selection_summary = build_selection_utility(
        selection_config,
        urtext_tokens=bundle.token_ids,
        seed=seed,
    )
    manifest = _load_or_create_manifest(
        paths,
        run_name=run_name,
        version=version,
        seed=seed,
        urtext_config=urtext_config,
        latent_config=latent_config,
        lab_config=lab_config,
        selection_config=selection_config,
        agent_config=effective_agent_config,
        bundle=bundle,
        selection_summary=selection_summary,
        resume=resume and not force_rebuild,
    )
    id_to_label = {int(k): str(v) for k, v in manifest["id_to_label"].items()}

    metrics_rows: list[dict[str, Any]] = []
    snapshots: dict[int, Sequence[int]] = {}
    start_generation = 0
    current_tokens: list[int] = []

    if resume and not force_rebuild and paths.metrics_partial_path.exists() and paths.checkpoint_path.exists():
        metrics_rows = _load_metrics_rows(paths.metrics_partial_path)
        checkpoint = json.loads(paths.checkpoint_path.read_text(encoding="utf-8"))
        last_completed_generation = int(checkpoint.get("last_completed_generation", -1))
        if last_completed_generation >= 0:
            current_tokens = load_generation_snapshot(
                paths.state_dir,
                min(last_completed_generation, lab_config.generations),
            )
            start_generation = last_completed_generation + 1
            for generation in {0, last_completed_generation, lab_config.generations}:
                snap = snapshot_path(paths.state_dir, generation)
                if snap.exists():
                    snapshots[generation] = load_generation_snapshot(paths.state_dir, generation)

    if not current_tokens:
        current_tokens = list(bundle.token_ids[:lab_config.text_length])
        save_generation_snapshot(paths.state_dir, 0, current_tokens)
        baseline_row = metrics_for_generation(
            current_tokens,
            lab_config,
            generation=0,
            baseline=None,
            utility=selection_utility,
            agent_stats=finalize_agent_stats({}) if effective_agent_config is not None else None,
        )
        metrics_rows = [baseline_row]
        snapshots[0] = list(current_tokens)
        manifest["updated_at"] = timestamp()
        atomic_save_json(paths.manifest_path, _json_ready(manifest))
        atomic_save_json(
            paths.checkpoint_path,
            _json_ready(
                {
                    "run_name": run_name,
                    "version": version,
                    "status": "running",
                    "last_completed_generation": 0,
                    "updated_at": manifest["updated_at"],
                }
            ),
        )
        _write_run_artifacts(
            paths,
            manifest,
            metrics_rows,
            snapshots,
            id_to_label,
            final=False,
            max_order=lab_config.max_order,
        )
        start_generation = 1

    if start_generation > lab_config.generations:
        manifest["status"] = "finished"
        atomic_save_json(paths.manifest_path, _json_ready(manifest))
        return RegenerationLabRun(
            version=version,
            run_name=run_name,
            paths=paths,
            manifest=manifest,
            metrics_rows=metrics_rows,
            final_tokens=tuple(current_tokens),
            id_to_label=id_to_label,
        )

    baseline = _baseline_metrics(metrics_rows, lab_config.max_order)
    iterator = range(start_generation, lab_config.generations + 1)
    iterator = tqdm(iterator, desc=f"ngram-lab:{run_name}", unit="gen", disable=not progress_bar)
    for generation in iterator:
        generation_rng = random.Random(seed + 1009 * generation)
        current_tokens, agent_stats = next_generation_text(
            current_tokens,
            lab_config,
            rng=generation_rng,
            selection_utility=selection_utility,
            selection_config=selection_config,
            agent_config=effective_agent_config,
        )
        save_generation_snapshot(paths.state_dir, generation, current_tokens)
        row = metrics_for_generation(
            current_tokens,
            lab_config,
            generation=generation,
            baseline=baseline,
            utility=selection_utility,
            agent_stats=agent_stats,
        )
        metrics_rows = [existing for existing in metrics_rows if int(existing["generation"]) != generation]
        metrics_rows.append(row)
        metrics_rows.sort(key=lambda item: int(item["generation"]))
        if generation in {0, lab_config.generations // 2, lab_config.generations}:
            snapshots[generation] = list(current_tokens)
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
                    "last_completed_generation": generation,
                    "updated_at": manifest["updated_at"],
                }
            ),
        )
        _write_run_artifacts(
            paths,
            manifest,
            metrics_rows,
            snapshots,
            id_to_label,
            final=False,
            max_order=lab_config.max_order,
        )
        postfix = {"vocab": int(row["vocab_size"])}
        postfix[f"{lab_config.max_order}gram_ratio"] = f"{row[f'distinct_{lab_config.max_order}grams_ratio_vs_gen0']:.3f}"
        if "desirable_window_share" in row:
            postfix["desirable"] = f"{row['desirable_window_share']:.3f}"
        iterator.set_postfix(postfix)

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
                "last_completed_generation": lab_config.generations,
                "updated_at": manifest["updated_at"],
            }
        ),
    )
    for generation in (0, lab_config.generations // 2, lab_config.generations):
        snap = snapshot_path(paths.state_dir, generation)
        if snap.exists():
            snapshots[generation] = load_generation_snapshot(paths.state_dir, generation)
    _write_run_artifacts(
        paths,
        manifest,
        metrics_rows,
        snapshots,
        id_to_label,
        final=True,
        max_order=lab_config.max_order,
    )
    return RegenerationLabRun(
        version=version,
        run_name=run_name,
        paths=paths,
        manifest=manifest,
        metrics_rows=metrics_rows,
        final_tokens=tuple(current_tokens),
        id_to_label=id_to_label,
    )
