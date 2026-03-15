from __future__ import annotations

import gzip
import json
import math
import random
import re
import sqlite3
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

BOS = "<BOS>"
EOS = "<EOS>"


@dataclass
class CorpusSource:
    author_key: str
    label: str
    gutenberg_ids: list[int]
    preferred_filename: str


CORPUS_REGISTRY: dict[str, CorpusSource] = {
    "conan_doyle": CorpusSource(
        author_key="conan_doyle",
        label="Arthur Conan Doyle",
        gutenberg_ids=[1661, 834, 108, 244, 2097, 2852],
        preferred_filename="conan_doyle_combined_clean.txt",
    ),
    "jane_austen": CorpusSource(
        author_key="jane_austen",
        label="Jane Austen",
        gutenberg_ids=[1342, 158, 161, 141, 105, 121],
        preferred_filename="jane_austen_combined_clean.txt",
    ),
    "charles_darwin": CorpusSource(
        author_key="charles_darwin",
        label="Charles Darwin",
        gutenberg_ids=[1228, 2300, 1227, 944],
        preferred_filename="charles_darwin_combined_clean.txt",
    ),
}


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def locate_project_root(start: str | Path | None = None) -> Path:
    if start is None:
        start = Path.cwd()
    path = Path(start).resolve()
    for candidate in [path, *path.parents]:
        if candidate.name == "Drift_and_selection":
            return candidate
        if (candidate / "GitHub").exists() and (candidate / "Nat_Paper").exists():
            return candidate
    # fallback: create a local pseudo-root in cwd for ad hoc use
    fallback = path / "Drift_and_selection"
    ensure_dir(fallback)
    return fallback


def default_paths(project_root: str | Path | None = None) -> dict[str, Path]:
    root = locate_project_root(project_root)
    github = ensure_dir(root / "GitHub")
    corpora = ensure_dir(github / "corpora")
    raw = ensure_dir(corpora / "raw")
    cleaned = ensure_dir(corpora / "cleaned")
    notebooks = ensure_dir(github / "notebooks" / "active")
    outputs = ensure_dir(github / "data" / "outputs" / "theorem1_vocabulary_drift")
    runs = ensure_dir(outputs / "runs")
    metrics = ensure_dir(outputs / "metrics")
    counts = ensure_dir(outputs / "counts")
    figures = ensure_dir(github / "figures" / "appendix")
    db_dir = ensure_dir(github / "data" / "databases")
    return {
        "root": root,
        "github": github,
        "corpora": corpora,
        "raw": raw,
        "cleaned": cleaned,
        "notebooks": notebooks,
        "outputs": outputs,
        "runs": runs,
        "metrics": metrics,
        "counts": counts,
        "figures": figures,
        "db": db_dir / "theorem1_vocabulary_drift.sqlite",
    }


GUTENBERG_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt.utf-8",
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
]


def fetch_gutenberg_text(gutenberg_id: int, timeout: int = 30) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; theorem1-vocab-drift/1.0)"}
    last_error: Exception | None = None
    for pattern in GUTENBERG_PATTERNS:
        url = pattern.format(gid=gutenberg_id)
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as response:
                raw = response.read()
            # utf-8 first, then latin-1 fallback
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    text = raw.decode(enc)
                    if "Project Gutenberg" in text and len(text) > 1000:
                        return text
                except UnicodeDecodeError:
                    continue
        except Exception as exc:  # pragma: no cover - network variability
            last_error = exc
            continue
    raise RuntimeError(f"Failed to fetch Gutenberg text for id={gutenberg_id}") from last_error


_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg_header_footer(text: str) -> str:
    start = _START_RE.search(text)
    if start:
        text = text[start.end():]
    end = _END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2014", " -- ")
    text = text.replace("\u2013", " - ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def combined_clean_path(author_key: str, project_root: str | Path | None = None) -> Path:
    paths = default_paths(project_root)
    info = CORPUS_REGISTRY[author_key]
    return ensure_dir(paths["cleaned"] / author_key) / info.preferred_filename


def raw_author_dir(author_key: str, project_root: str | Path | None = None) -> Path:
    paths = default_paths(project_root)
    return ensure_dir(paths["raw"] / author_key)


def build_or_download_author_corpus(
    author_key: str,
    project_root: str | Path | None = None,
    force_download_missing: bool = True,
    force_rebuild_clean: bool = False,
) -> Path:
    clean_path = combined_clean_path(author_key, project_root)
    if clean_path.exists() and not force_rebuild_clean:
        print(f"[corpus] Found existing cleaned corpus, reusing: {clean_path}")
        return clean_path

    info = CORPUS_REGISTRY[author_key]
    raw_dir = raw_author_dir(author_key, project_root)
    has_any_local_raw = any(p.suffix.lower() == ".txt" and p.stat().st_size > 1024 for p in raw_dir.glob("*.txt"))
    texts: list[str] = []
    for gid in info.gutenberg_ids:
        raw_path = raw_dir / f"pg{gid}.txt"
        if not raw_path.exists() and force_download_missing and not has_any_local_raw:
            print(f"[corpus] Downloading Project Gutenberg id={gid} for {author_key}")
            raw_text = fetch_gutenberg_text(gid)
            raw_path.write_text(raw_text, encoding="utf-8")
        if raw_path.exists():
            raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
            clean = normalize_text(strip_gutenberg_header_footer(raw_text))
            texts.append(clean)
    if not texts:
        # Fallback for already-downloaded corpora that do not use pg<ID>.txt naming.
        fallback_raw = [
            p for p in sorted(raw_dir.glob("*.txt"))
            if p.name.lower() != "readme.txt" and p.stat().st_size > 1024
        ]
        if fallback_raw:
            print(f"[corpus] Found existing raw corpus files, reusing {len(fallback_raw)} files from {raw_dir}")
            for raw_path in fallback_raw:
                raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
                clean = normalize_text(strip_gutenberg_header_footer(raw_text))
                texts.append(clean)
    if not texts:
        raise FileNotFoundError(f"No raw texts available for {author_key}; either supply them manually or enable downloads.")

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    combined = "\n\n".join(texts)
    clean_path.write_text(combined, encoding="utf-8")
    print(f"[corpus] Wrote cleaned combined corpus: {clean_path}")
    return clean_path


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'\(\[]?[A-Z0-9])")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")


def sentence_tokenize(text: str, lowercase: bool = True, progress_bar: bool = False) -> list[list[str]]:
    text = normalize_text(text)
    raw_sentences = []
    blocks = text.split("\n\n")
    iterator = tqdm(blocks, desc="sentence-parse", unit="block", disable=not progress_bar)
    for block in iterator:
        block = block.strip()
        if not block:
            continue
        raw_sentences.extend(_SENTENCE_SPLIT_RE.split(block))
    tokenized: list[list[str]] = []
    for sent in raw_sentences:
        tokens = _TOKEN_RE.findall(sent)
        if lowercase:
            tokens = [tok.lower() for tok in tokens]
        if len(tokens) >= 3:
            tokenized.append(tokens)
    return tokenized


def flatten_sentences(sentences: Sequence[Sequence[str]]) -> list[str]:
    return [tok for sent in sentences for tok in sent]


def build_ngram_counts(sentences: Sequence[Sequence[str]], n: int) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for sent in sentences:
        padded = [BOS] * (n - 1) + list(sent) + [EOS]
        for i in range(len(padded) - n + 1):
            counts[tuple(padded[i : i + n])] += 1
    return counts


def build_trigram_model(sentences: Sequence[Sequence[str]]) -> dict:
    trigram_counts = build_ngram_counts(sentences, 3)
    bigram_counts = build_ngram_counts(sentences, 2)
    unigram_counts = Counter(flatten_sentences(sentences) + [EOS])
    starts = Counter()
    for sent in sentences:
        if len(sent) >= 1:
            starts[(BOS, BOS)] += 1
    context_to_next: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for (w1, w2, w3), c in trigram_counts.items():
        context_to_next[(w1, w2)][w3] += c
    bigram_to_next: dict[str, Counter[str]] = defaultdict(Counter)
    for (w1, w2), c in bigram_counts.items():
        bigram_to_next[w1][w2] += c
    return {
        "trigram_counts": trigram_counts,
        "bigram_counts": bigram_counts,
        "unigram_counts": unigram_counts,
        "context_to_next": context_to_next,
        "bigram_to_next": bigram_to_next,
    }


def weighted_choice(counter: Counter[str], rng: random.Random) -> str:
    items = list(counter.items())
    total = sum(v for _, v in items)
    r = rng.random() * total
    acc = 0.0
    for item, weight in items:
        acc += weight
        if acc >= r:
            return item
    return items[-1][0]


def sample_sentence(
    model: dict,
    rng: random.Random,
    max_tokens: int = 60,
    allow_backoff: bool = True,
) -> list[str]:
    sent: list[str] = []
    w1, w2 = BOS, BOS
    for _ in range(max_tokens):
        next_counter = model["context_to_next"].get((w1, w2))
        if next_counter:
            nxt = weighted_choice(next_counter, rng)
        elif allow_backoff and w2 in model["bigram_to_next"]:
            nxt = weighted_choice(model["bigram_to_next"][w2], rng)
        elif allow_backoff:
            nxt = weighted_choice(model["unigram_counts"], rng)
        else:
            break
        if nxt == EOS:
            break
        sent.append(nxt)
        w1, w2 = w2, nxt
    return sent


def generate_sentences_by_token_budget(
    model: dict,
    target_tokens: int,
    rng: random.Random,
    min_sent_len: int = 3,
    max_sent_len: int = 60,
    progress_bar: bool = False,
) -> list[list[str]]:
    sentences: list[list[str]] = []
    produced = 0
    attempts = 0
    pbar = tqdm(total=max(0, target_tokens), desc="generate-synthetic", unit="tok", disable=not progress_bar)
    while produced < target_tokens and attempts < target_tokens * 20:
        sent = sample_sentence(model, rng=rng, max_tokens=max_sent_len)
        attempts += 1
        if len(sent) < min_sent_len:
            continue
        remaining = target_tokens - produced
        if len(sent) > remaining + 5:
            sent = sent[:remaining]
        if not sent:
            continue
        sentences.append(sent)
        produced += len(sent)
        pbar.update(len(sent))
    pbar.close()
    return sentences


def sample_original_sentences_by_token_budget(
    anchor_sentences: Sequence[Sequence[str]],
    target_tokens: int,
    rng: random.Random,
    progress_bar: bool = False,
) -> list[list[str]]:
    sampled: list[list[str]] = []
    produced = 0
    pbar = tqdm(total=max(0, target_tokens), desc="sample-anchor", unit="tok", disable=not progress_bar)
    while produced < target_tokens:
        sent = list(rng.choice(anchor_sentences))
        remaining = target_tokens - produced
        if len(sent) > remaining + 5:
            sent = sent[:remaining]
        if len(sent) == 0:
            continue
        sampled.append(sent)
        produced += len(sent)
        pbar.update(len(sent))
    pbar.close()
    return sampled


def corpus_metrics(
    sentences: Sequence[Sequence[str]],
    original_token_count: int | None = None,
) -> dict[str, float | int]:
    tokens = flatten_sentences(sentences)
    token_count = len(tokens)
    word_counts = Counter(tokens)
    trigram_counts = build_ngram_counts(sentences, 3)
    bigram_counts = build_ngram_counts(sentences, 2)
    head_word_counter: Counter[str] = Counter()
    for (w1, w2, _w3), c in trigram_counts.items():
        if w1 not in {BOS, EOS}:
            head_word_counter[w1] += c
        if w2 not in {BOS, EOS}:
            head_word_counter[w2] += c
    probs = np.array(list(word_counts.values()), dtype=float)
    probs = probs / probs.sum() if probs.size else probs
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    metrics = {
        "token_count": token_count,
        "vocab_size": len(word_counts),
        "head_vocab_size": len(head_word_counter),
        "distinct_bigram_contexts": len({k for k in bigram_counts if k[0] != BOS}),
        "trigram_type_count": len({k for k in trigram_counts if k[0] != BOS and k[1] != BOS}),
        "hapax_count": sum(1 for c in word_counts.values() if c == 1),
        "token_entropy_bits": entropy,
    }
    if original_token_count is not None and original_token_count > 0:
        threshold = 1.0 / original_token_count
        metrics["active_vocab_gt_1_over_M0"] = sum((c / token_count) > threshold for c in word_counts.values()) if token_count else 0
        metrics["active_count_threshold"] = math.floor(token_count / original_token_count) + 1 if token_count else 0
    return metrics


def top_counts(counter: Counter, k: int = 100) -> pd.DataFrame:
    rows = []
    for rank, (item, count) in enumerate(counter.most_common(k), start=1):
        if isinstance(item, tuple):
            row = {f"item_{i+1}": x for i, x in enumerate(item)}
        else:
            row = {"item": item}
        row.update({"rank": rank, "count": count})
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class SimulationConfig:
    corpus_name: str
    mode: str  # fixed_size_mix | growing_corpus
    alpha: float
    generations: int
    seed: int = 12345
    synthetic_fraction_mode: str = "mix_with_original_anchor"
    lowercase: bool = True
    save_full_trigram_tables: bool = False
    notes: str = ""


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def slugify(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    out = re.sub(r"_+", "_", out)
    return out.strip("_") or "run"


def ensure_run_paths(paths: dict[str, Path], run_name: str) -> dict[str, Path]:
    run_dir = ensure_dir(paths["outputs"] / "runs" / slugify(run_name))
    counts_partial = ensure_dir(run_dir / "counts_partial")
    plots_partial = ensure_dir(run_dir / "plots_partial")
    state_dir = ensure_dir(run_dir / "state")
    return {
        "run_dir": run_dir,
        "counts_partial": counts_partial,
        "plots_partial": plots_partial,
        "state_dir": state_dir,
        "manifest": run_dir / "run_manifest.json",
        "checkpoint": run_dir / "checkpoint_state.json",
        "metrics_partial": run_dir / "metrics_partial.csv",
    }


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path, default: dict | None = None) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {} if default is None else default


def save_sentence_snapshot(state_dir: Path, generation: int, sentences: Sequence[Sequence[str]]) -> Path:
    ensure_dir(state_dir)
    out_path = state_dir / f"sentences_gen_{generation:03d}.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump([list(sent) for sent in sentences], f)
    return out_path


def load_sentence_snapshot(path: Path) -> list[list[str]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return [list(sent) for sent in data]


def init_db(db_path: str | Path) -> None:
    db_path = Path(db_path)
    ensure_dir(db_path.parent)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                corpus_name TEXT,
                mode TEXT,
                alpha REAL,
                generations INTEGER,
                seed INTEGER,
                notes TEXT,
                config_json TEXT
            );
            CREATE TABLE IF NOT EXISTS generation_metrics (
                run_id INTEGER,
                generation INTEGER,
                token_count INTEGER,
                vocab_size INTEGER,
                head_vocab_size INTEGER,
                distinct_bigram_contexts INTEGER,
                trigram_type_count INTEGER,
                hapax_count INTEGER,
                token_entropy_bits REAL,
                active_vocab_gt_1_over_M0 INTEGER,
                active_count_threshold INTEGER,
                PRIMARY KEY (run_id, generation)
            );
            CREATE TABLE IF NOT EXISTS word_counts (
                run_id INTEGER,
                generation INTEGER,
                word TEXT,
                count INTEGER,
                rel_freq REAL,
                PRIMARY KEY (run_id, generation, word)
            );
            CREATE TABLE IF NOT EXISTS head_word_counts (
                run_id INTEGER,
                generation INTEGER,
                word TEXT,
                count INTEGER,
                PRIMARY KEY (run_id, generation, word)
            );
            """
        )


def register_run(db_path: str | Path, config: SimulationConfig) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs(created_at, corpus_name, mode, alpha, generations, seed, notes, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp(),
                config.corpus_name,
                config.mode,
                config.alpha,
                config.generations,
                config.seed,
                config.notes,
                json.dumps(asdict(config), sort_keys=True),
            ),
        )
        return int(cur.lastrowid)


def save_generation_to_db(
    db_path: str | Path,
    run_id: int,
    generation: int,
    sentences: Sequence[Sequence[str]],
    original_token_count: int,
) -> dict[str, float | int]:
    metrics = corpus_metrics(sentences, original_token_count=original_token_count)
    tokens = flatten_sentences(sentences)
    word_counts = Counter(tokens)
    trigram_counts = build_ngram_counts(sentences, 3)
    head_word_counter: Counter[str] = Counter()
    for (w1, w2, _w3), c in trigram_counts.items():
        if w1 not in {BOS, EOS}:
            head_word_counter[w1] += c
        if w2 not in {BOS, EOS}:
            head_word_counter[w2] += c

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO generation_metrics(
                run_id, generation, token_count, vocab_size, head_vocab_size,
                distinct_bigram_contexts, trigram_type_count, hapax_count,
                token_entropy_bits, active_vocab_gt_1_over_M0, active_count_threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                generation,
                metrics["token_count"],
                metrics["vocab_size"],
                metrics["head_vocab_size"],
                metrics["distinct_bigram_contexts"],
                metrics["trigram_type_count"],
                metrics["hapax_count"],
                metrics["token_entropy_bits"],
                metrics.get("active_vocab_gt_1_over_M0", 0),
                metrics.get("active_count_threshold", 0),
            ),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO word_counts(run_id, generation, word, count, rel_freq) VALUES (?, ?, ?, ?, ?)",
            [
                (run_id, generation, word, count, count / metrics["token_count"])
                for word, count in word_counts.items()
            ],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO head_word_counts(run_id, generation, word, count) VALUES (?, ?, ?, ?)",
            [(run_id, generation, word, count) for word, count in head_word_counter.items()],
        )
    return metrics


def export_generation_counts(
    counts_dir: str | Path,
    run_id: int,
    generation: int,
    sentences: Sequence[Sequence[str]],
    save_full_trigrams: bool = False,
) -> None:
    counts_dir = ensure_dir(counts_dir)
    tokens = flatten_sentences(sentences)
    word_counts = Counter(tokens)
    trigram_counts = build_ngram_counts(sentences, 3)
    bigram_counts = build_ngram_counts(sentences, 2)

    word_df = pd.DataFrame(sorted(word_counts.items(), key=lambda x: (-x[1], x[0])), columns=["word", "count"])
    word_df.to_csv(counts_dir / f"run_{run_id:04d}_gen_{generation:03d}_word_counts.csv.gz", index=False)

    context_rows = []
    for (w1, w2), count in sorted(bigram_counts.items(), key=lambda x: (-x[1], x[0])):
        context_rows.append({"w1": w1, "w2": w2, "count": count})
    pd.DataFrame(context_rows).to_csv(counts_dir / f"run_{run_id:04d}_gen_{generation:03d}_bigram_context_counts.csv.gz", index=False)

    trigram_path = counts_dir / f"run_{run_id:04d}_gen_{generation:03d}_trigram_counts.csv.gz"
    if save_full_trigrams:
        tri_rows = []
        for (w1, w2, w3), count in sorted(trigram_counts.items(), key=lambda x: (-x[1], x[0])):
            tri_rows.append({"w1": w1, "w2": w2, "w3": w3, "count": count})
        pd.DataFrame(tri_rows).to_csv(trigram_path, index=False)
    else:
        tri_rows = []
        for (w1, w2, w3), count in trigram_counts.most_common(5000):
            tri_rows.append({"w1": w1, "w2": w2, "w3": w3, "count": count})
        pd.DataFrame(tri_rows).to_csv(trigram_path, index=False)


def simulate_fixed_size_mix(
    anchor_sentences: Sequence[Sequence[str]],
    config: SimulationConfig,
    paths: dict[str, Path] | None = None,
    run_name: str | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
    progress_bar: bool = True,
) -> tuple[int, pd.DataFrame]:
    assert 0.0 <= config.alpha <= 1.0
    assert config.mode == "fixed_size_mix"
    paths = default_paths() if paths is None else paths
    init_db(paths["db"])
    run_name = run_name or f"{config.corpus_name}_{config.mode}_alpha_{alpha_tag(config.alpha)}_seed_{config.seed}"
    rp = ensure_run_paths(paths, run_name)
    manifest = read_json(rp["manifest"], default={})
    checkpoint = read_json(rp["checkpoint"], default={})

    if force_rebuild:
        for p in rp["counts_partial"].glob("*"):
            if p.is_file():
                p.unlink()
        for p in rp["state_dir"].glob("sentences_gen_*.json.gz"):
            p.unlink()
        if rp["metrics_partial"].exists():
            rp["metrics_partial"].unlink()
        checkpoint = {}
        print(f"[run] force_rebuild=True, cleared previous checkpoint state for {run_name}")

    run_id = int(manifest.get("run_id", 0) or 0)
    if run_id == 0:
        run_id = register_run(paths["db"], config)
        manifest = {
            "run_name": run_name,
            "run_id": run_id,
            "created_at": timestamp(),
            "config": asdict(config),
        }
        write_json(rp["manifest"], manifest)
        print(f"[run] Started new run_id={run_id} ({run_name})")
    else:
        print(f"[run] Found existing run manifest for {run_name} (run_id={run_id})")

    original_token_count = len(flatten_sentences(anchor_sentences))
    save_sentence_snapshot(rp["state_dir"], 0, anchor_sentences)

    start_gen = 0
    metrics_rows: list[dict] = []
    current_sentences = [list(sent) for sent in anchor_sentences]
    if rp["metrics_partial"].exists() and resume and not force_rebuild:
        partial_df = pd.read_csv(rp["metrics_partial"])
        metrics_rows = partial_df.to_dict("records")
        if not partial_df.empty:
            last_completed_generation = int(partial_df["generation"].max())
            start_gen = last_completed_generation + 1
            if start_gen <= config.generations:
                snap = rp["state_dir"] / f"sentences_gen_{start_gen:03d}.json.gz"
                if snap.exists():
                    current_sentences = load_sentence_snapshot(snap)
                    print(f"[run] Resuming simulation from generation {start_gen}")
                else:
                    print(f"[run] Missing snapshot for generation {start_gen}; restarting from generation 0")
                    start_gen = 0
                    metrics_rows = []
    if start_gen > config.generations:
        print(f"[run] Run already complete for {run_name}; reusing existing outputs.")
        df = pd.DataFrame(metrics_rows)
        return run_id, df

    iterator = tqdm(
        range(start_gen, config.generations + 1),
        desc=f"{config.mode}:{config.corpus_name}:alpha={config.alpha:g}",
        unit="gen",
        disable=not progress_bar,
    )
    for gen in iterator:
        checkpoint.update(
            {
                "run_name": run_name,
                "run_id": run_id,
                "mode": config.mode,
                "corpus_name": config.corpus_name,
                "alpha": config.alpha,
                "last_completed_stage": "generation_metrics",
                "last_completed_generation": gen - 1,
                "last_completed_alpha": config.alpha,
                "updated_at": timestamp(),
            }
        )
        write_json(rp["checkpoint"], checkpoint)
        metrics = save_generation_to_db(paths["db"], run_id, gen, current_sentences, original_token_count)
        export_generation_counts(rp["counts_partial"], run_id, gen, current_sentences, save_full_trigrams=config.save_full_trigram_tables)
        row = {"run_id": run_id, "generation": gen, **metrics, "alpha": config.alpha, "mode": config.mode, "corpus_name": config.corpus_name}
        metrics_rows = [r for r in metrics_rows if int(r.get("generation", -1)) != gen]
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).sort_values("generation").to_csv(rp["metrics_partial"], index=False)
        checkpoint.update(
            {
                "last_completed_stage": "generation_complete",
                "last_completed_generation": gen,
                "updated_at": timestamp(),
            }
        )
        write_json(rp["checkpoint"], checkpoint)
        if gen == config.generations:
            break
        checkpoint.update({"last_completed_stage": "trigram_count_build", "updated_at": timestamp()})
        write_json(rp["checkpoint"], checkpoint)
        model = build_trigram_model(current_sentences)
        rng_gen = random.Random(config.seed + 10007 * gen)
        synthetic_tokens = int(round(config.alpha * original_token_count))
        anchor_tokens = max(0, original_token_count - synthetic_tokens)
        original_part = sample_original_sentences_by_token_budget(anchor_sentences, anchor_tokens, rng_gen, progress_bar=progress_bar) if anchor_tokens > 0 else []
        synthetic_part = generate_sentences_by_token_budget(model, synthetic_tokens, rng_gen, progress_bar=progress_bar) if synthetic_tokens > 0 else []
        current_sentences = original_part + synthetic_part
        rng_gen.shuffle(current_sentences)
        save_sentence_snapshot(rp["state_dir"], gen + 1, current_sentences)
        checkpoint.update(
            {
                "last_completed_stage": "next_generation_prepared",
                "next_generation_ready": gen + 1,
                "updated_at": timestamp(),
            }
        )
        write_json(rp["checkpoint"], checkpoint)
    df = pd.DataFrame(metrics_rows)
    df = df.sort_values("generation")
    df.to_csv(paths["metrics"] / f"run_{run_id:04d}_metrics.csv", index=False)
    checkpoint.update(
        {
            "last_completed_stage": "finished",
            "last_completed_generation": config.generations,
            "updated_at": timestamp(),
        }
    )
    write_json(rp["checkpoint"], checkpoint)
    return run_id, df


def simulate_growing_corpus(
    anchor_sentences: Sequence[Sequence[str]],
    config: SimulationConfig,
    paths: dict[str, Path] | None = None,
    run_name: str | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
    progress_bar: bool = True,
) -> tuple[int, pd.DataFrame]:
    assert config.mode == "growing_corpus"
    assert config.alpha >= 0.0
    paths = default_paths() if paths is None else paths
    init_db(paths["db"])
    run_name = run_name or f"{config.corpus_name}_{config.mode}_alpha_{alpha_tag(config.alpha)}_seed_{config.seed}"
    rp = ensure_run_paths(paths, run_name)
    manifest = read_json(rp["manifest"], default={})
    checkpoint = read_json(rp["checkpoint"], default={})

    if force_rebuild:
        for p in rp["counts_partial"].glob("*"):
            if p.is_file():
                p.unlink()
        for p in rp["state_dir"].glob("sentences_gen_*.json.gz"):
            p.unlink()
        if rp["metrics_partial"].exists():
            rp["metrics_partial"].unlink()
        checkpoint = {}
        print(f"[run] force_rebuild=True, cleared previous checkpoint state for {run_name}")

    run_id = int(manifest.get("run_id", 0) or 0)
    if run_id == 0:
        run_id = register_run(paths["db"], config)
        manifest = {
            "run_name": run_name,
            "run_id": run_id,
            "created_at": timestamp(),
            "config": asdict(config),
        }
        write_json(rp["manifest"], manifest)
        print(f"[run] Started new run_id={run_id} ({run_name})")
    else:
        print(f"[run] Found existing run manifest for {run_name} (run_id={run_id})")

    original_token_count = len(flatten_sentences(anchor_sentences))
    current_sentences = [list(sent) for sent in anchor_sentences]
    save_sentence_snapshot(rp["state_dir"], 0, anchor_sentences)
    metrics_rows: list[dict] = []
    start_gen = 0
    if rp["metrics_partial"].exists() and resume and not force_rebuild:
        partial_df = pd.read_csv(rp["metrics_partial"])
        metrics_rows = partial_df.to_dict("records")
        if not partial_df.empty:
            last_completed_generation = int(partial_df["generation"].max())
            start_gen = last_completed_generation + 1
            if start_gen <= config.generations:
                snap = rp["state_dir"] / f"sentences_gen_{start_gen:03d}.json.gz"
                if snap.exists():
                    current_sentences = load_sentence_snapshot(snap)
                    print(f"[run] Resuming simulation from generation {start_gen}")
                else:
                    print(f"[run] Missing snapshot for generation {start_gen}; restarting from generation 0")
                    start_gen = 0
                    metrics_rows = []
    if start_gen > config.generations:
        print(f"[run] Run already complete for {run_name}; reusing existing outputs.")
        df = pd.DataFrame(metrics_rows)
        return run_id, df

    add_tokens_each_gen = int(round(config.alpha * original_token_count))

    iterator = tqdm(
        range(start_gen, config.generations + 1),
        desc=f"{config.mode}:{config.corpus_name}:alpha={config.alpha:g}",
        unit="gen",
        disable=not progress_bar,
    )
    for gen in iterator:
        checkpoint.update(
            {
                "run_name": run_name,
                "run_id": run_id,
                "mode": config.mode,
                "corpus_name": config.corpus_name,
                "alpha": config.alpha,
                "last_completed_stage": "generation_metrics",
                "last_completed_generation": gen - 1,
                "last_completed_alpha": config.alpha,
                "updated_at": timestamp(),
            }
        )
        write_json(rp["checkpoint"], checkpoint)
        metrics = save_generation_to_db(paths["db"], run_id, gen, current_sentences, original_token_count)
        export_generation_counts(rp["counts_partial"], run_id, gen, current_sentences, save_full_trigrams=config.save_full_trigram_tables)
        row = {"run_id": run_id, "generation": gen, **metrics, "alpha": config.alpha, "mode": config.mode, "corpus_name": config.corpus_name}
        metrics_rows = [r for r in metrics_rows if int(r.get("generation", -1)) != gen]
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).sort_values("generation").to_csv(rp["metrics_partial"], index=False)
        checkpoint.update(
            {
                "last_completed_stage": "generation_complete",
                "last_completed_generation": gen,
                "updated_at": timestamp(),
            }
        )
        write_json(rp["checkpoint"], checkpoint)
        if gen == config.generations:
            break
        checkpoint.update({"last_completed_stage": "trigram_count_build", "updated_at": timestamp()})
        write_json(rp["checkpoint"], checkpoint)
        model = build_trigram_model(current_sentences)
        rng_gen = random.Random(config.seed + 10007 * gen)
        synthetic_part = generate_sentences_by_token_budget(model, add_tokens_each_gen, rng_gen, progress_bar=progress_bar) if add_tokens_each_gen > 0 else []
        current_sentences = current_sentences + synthetic_part
        save_sentence_snapshot(rp["state_dir"], gen + 1, current_sentences)
        checkpoint.update(
            {
                "last_completed_stage": "next_generation_prepared",
                "next_generation_ready": gen + 1,
                "updated_at": timestamp(),
            }
        )
        write_json(rp["checkpoint"], checkpoint)
    df = pd.DataFrame(metrics_rows)
    df = df.sort_values("generation")
    df.to_csv(paths["metrics"] / f"run_{run_id:04d}_metrics.csv", index=False)
    checkpoint.update(
        {
            "last_completed_stage": "finished",
            "last_completed_generation": config.generations,
            "updated_at": timestamp(),
        }
    )
    write_json(rp["checkpoint"], checkpoint)
    return run_id, df


def run_alpha_grid(
    anchor_sentences: Sequence[Sequence[str]],
    corpus_name: str,
    alphas: Sequence[float],
    generations: int,
    mode: str,
    base_seed: int = 12345,
    save_full_trigram_tables: bool = False,
    paths: dict[str, Path] | None = None,
    run_name: str | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
    progress_bar: bool = True,
) -> pd.DataFrame:
    paths = default_paths() if paths is None else paths
    grid_run_name = run_name or f"{corpus_name}_{mode}_alpha_grid"
    grid_rp = ensure_run_paths(paths, grid_run_name)
    grid_manifest = read_json(grid_rp["manifest"], default={})
    if not grid_manifest:
        write_json(
            grid_rp["manifest"],
            {
                "run_name": grid_run_name,
                "created_at": timestamp(),
                "corpus_name": corpus_name,
                "mode": mode,
                "alphas": [float(a) for a in alphas],
                "generations": int(generations),
            },
        )
    grid_state = read_json(grid_rp["checkpoint"], default={})
    completed_alpha_tags = set(grid_state.get("completed_alpha_tags", []))
    frames = []
    alpha_iter = tqdm(alphas, desc=f"alpha-grid:{corpus_name}:{mode}", unit="alpha", disable=not progress_bar)
    for i, alpha in enumerate(alpha_iter):
        tag = alpha_tag(float(alpha))
        if resume and not force_rebuild and tag in completed_alpha_tags:
            print(f"[grid] Skipping completed alpha={alpha:g} for {grid_run_name}")
            alpha_run_name = f"{grid_run_name}__alpha_{tag}"
            alpha_rp = ensure_run_paths(paths, alpha_run_name)
            if alpha_rp["metrics_partial"].exists():
                frames.append(pd.read_csv(alpha_rp["metrics_partial"]))
            continue
        config = SimulationConfig(
            corpus_name=corpus_name,
            mode=mode,
            alpha=float(alpha),
            generations=generations,
            seed=base_seed + i,
            save_full_trigram_tables=save_full_trigram_tables,
            notes=f"alpha-grid run for {mode}",
        )
        alpha_run_name = f"{grid_run_name}__alpha_{tag}"
        if mode == "fixed_size_mix":
            _run_id, df = simulate_fixed_size_mix(
                anchor_sentences,
                config,
                paths=paths,
                run_name=alpha_run_name,
                resume=resume,
                force_rebuild=force_rebuild,
                progress_bar=progress_bar,
            )
        elif mode == "growing_corpus":
            _run_id, df = simulate_growing_corpus(
                anchor_sentences,
                config,
                paths=paths,
                run_name=alpha_run_name,
                resume=resume,
                force_rebuild=force_rebuild,
                progress_bar=progress_bar,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        frames.append(df)
        completed_alpha_tags.add(tag)
        grid_state.update(
            {
                "last_completed_stage": "alpha_complete",
                "last_completed_alpha": float(alpha),
                "completed_alpha_tags": sorted(completed_alpha_tags),
                "updated_at": timestamp(),
            }
        )
        write_json(grid_rp["checkpoint"], grid_state)
    if not frames:
        raise RuntimeError(f"No alpha-grid runs produced output for {grid_run_name}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(paths["metrics"] / f"{corpus_name}_{mode}_alpha_grid_metrics.csv", index=False)
    out.to_csv(grid_rp["run_dir"] / "metrics_partial.csv", index=False)
    grid_state.update(
        {
            "last_completed_stage": "finished",
            "completed_alpha_tags": sorted(completed_alpha_tags),
            "updated_at": timestamp(),
        }
    )
    write_json(grid_rp["checkpoint"], grid_state)
    return out


def load_clean_corpus(
    author_key: str,
    project_root: str | Path | None = None,
    lowercase: bool = True,
    progress_bar: bool = True,
) -> list[list[str]]:
    clean_path = build_or_download_author_corpus(author_key, project_root=project_root)
    text = clean_path.read_text(encoding="utf-8", errors="ignore")
    print(f"[corpus] Parsing cleaned corpus into sentences: {clean_path}")
    return sentence_tokenize(text, lowercase=lowercase, progress_bar=progress_bar)


def plot_metric_grid(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path_base: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for alpha, sub in sorted(df.groupby("alpha"), key=lambda x: x[0]):
        ax.plot(sub["generation"], sub[metric], marker="o", label=f"alpha={alpha:g}")
    ax.set_xlabel("generation")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend(title="mixture")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path_base is not None:
        out_path_base = Path(out_path_base)
        fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_path_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    return fig


def summarize_corpus(sentences: Sequence[Sequence[str]], top_k: int = 25) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    word_counts = Counter(flatten_sentences(sentences))
    trigram_counts = build_ngram_counts(sentences, 3)
    metrics = pd.DataFrame([corpus_metrics(sentences, original_token_count=len(flatten_sentences(sentences)))])
    return metrics, top_counts(word_counts, top_k), top_counts(trigram_counts, top_k)


def notebook_instructions_for_codex() -> str:
    return textwrap.dedent(
        """
        Place the notebook at `Drift_and_selection/GitHub/notebooks/active/17_theorem1_vocabulary_drift_trigrams.ipynb`
        and the helper module at `Drift_and_selection/GitHub/src/drift_selection/theorem1_vocab_drift.py`.
        Ensure the notebook discovers the project root by walking upward until it finds `Drift_and_selection/`.
        Corpora should live under `Drift_and_selection/GitHub/corpora/{raw,cleaned}/<author>/`.
        Outputs should go under `Drift_and_selection/GitHub/data/outputs/theorem1_vocabulary_drift/`.
        The SQLite database should be `Drift_and_selection/GitHub/data/databases/theorem1_vocabulary_drift.sqlite`.
        """
    ).strip()
