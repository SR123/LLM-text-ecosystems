
from __future__ import annotations

import gzip
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    # expected repo placement
    from drift_selection.theorem1_vocab_drift import (
        BOS,
        EOS,
        build_or_download_author_corpus,
        build_ngram_counts,
        build_trigram_model,
        corpus_metrics,
        default_paths as base_default_paths,
        flatten_sentences,
        generate_sentences_by_token_budget,
        locate_project_root,
        normalize_text,
        sample_original_sentences_by_token_budget,
        sentence_tokenize,
        timestamp,
        ensure_dir,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    from theorem1_vocab_drift import (
        BOS,
        EOS,
        build_or_download_author_corpus,
        build_ngram_counts,
        build_trigram_model,
        corpus_metrics,
        default_paths as base_default_paths,
        flatten_sentences,
        generate_sentences_by_token_budget,
        locate_project_root,
        normalize_text,
        sample_original_sentences_by_token_budget,
        sentence_tokenize,
        timestamp,
        ensure_dir,
    )


AUSTEN_KEY = "jane_austen"


def theorem1_noise_paths(project_root: str | Path | None = None) -> dict[str, Path]:
    base = base_default_paths(project_root)
    root = locate_project_root(project_root)
    github = base["github"]
    out_root = ensure_dir(github / "data" / "outputs" / "theorem1_standardization")
    derived = ensure_dir(out_root / "derived_corpora")
    runs = ensure_dir(out_root / "runs")
    metrics = ensure_dir(out_root / "metrics")
    figures = ensure_dir(github / "figures" / "appendix" / "theorem1_standardization")
    db = ensure_dir(github / "data" / "databases") / "theorem1_standardization.sqlite"
    return {
        **base,
        "root": root,
        "out_root": out_root,
        "derived": derived,
        "runs": runs,
        "metrics": metrics,
        "figures_std": figures,
        "db_std": db,
    }


def load_austen_sentences(project_root: str | Path | None = None, lowercase: bool = True) -> list[list[str]]:
    clean_path = build_or_download_author_corpus(AUSTEN_KEY, project_root=project_root)
    text = clean_path.read_text(encoding="utf-8", errors="ignore")
    return sentence_tokenize(text, lowercase=lowercase)


def join_sentences(sentences: Sequence[Sequence[str]]) -> str:
    return "\n".join(" ".join(sent) for sent in sentences)


def save_sentence_state(path: str | Path, sentences: Sequence[Sequence[str]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")
    return path


def load_sentence_state(path: str | Path) -> list[list[str]]:
    sentences: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                sentences.append(toks)
    return sentences


def downsample_sentences_to_token_budget(
    anchor_sentences: Sequence[Sequence[str]],
    token_budget: int,
    seed: int = 12345,
) -> list[list[str]]:
    rng = random.Random(seed)
    idxs = list(range(len(anchor_sentences)))
    rng.shuffle(idxs)
    sampled: list[list[str]] = []
    total = 0
    for idx in idxs:
        sent = list(anchor_sentences[idx])
        if total + len(sent) > token_budget and total > int(0.9 * token_budget):
            break
        sampled.append(sent)
        total += len(sent)
        if total >= token_budget:
            break
    if total < int(0.8 * token_budget):
        # fallback: cycle until close
        while total < token_budget:
            sent = list(anchor_sentences[rng.choice(idxs)])
            sampled.append(sent)
            total += len(sent)
    return sampled


# ---------- Noise family 1: OCR / artifact corruption ----------

def _ocr_variants(token: str) -> list[list[str]]:
    vars_out: list[list[str]] = []
    if len(token) < 4:
        return vars_out
    if "'" in token:
        vars_out.append([token.replace("'", "")])
    if "-" in token:
        vars_out.append(token.replace("-", " ").split())
        vars_out.append([token.replace("-", "")])
    if "m" in token:
        vars_out.append([token.replace("m", "rn", 1)])
    if "rn" in token:
        vars_out.append([token.replace("rn", "m", 1)])
    if "w" in token:
        vars_out.append([token.replace("w", "vv", 1)])
    if "d" in token:
        vars_out.append([token.replace("d", "cl", 1)])
    if "cl" in token:
        vars_out.append([token.replace("cl", "d", 1)])
    if token.endswith("e") and len(token) > 5:
        vars_out.append([token[:-1]])  # dropped final letter
    return [v for v in vars_out if v and v != [token]]


def inject_ocr_artifacts(
    sentences: Sequence[Sequence[str]],
    rate: float,
    seed: int = 12345,
) -> tuple[list[list[str]], dict]:
    rng = random.Random(seed)
    out_sentences: list[list[str]] = []
    changed_positions = 0
    candidate_positions = 0
    injected_tokens: Counter[str] = Counter()
    original_to_noisy: dict[str, Counter[str]] = defaultdict(Counter)

    for sent in sentences:
        new_sent: list[str] = []
        for tok in sent:
            variants = _ocr_variants(tok)
            if variants:
                candidate_positions += 1
            if variants and rng.random() < rate:
                replacement = rng.choice(variants)
                changed_positions += 1
                for rtok in replacement:
                    new_sent.append(rtok)
                    injected_tokens[rtok] += 1
                    original_to_noisy[tok][rtok] += 1
            else:
                new_sent.append(tok)
        out_sentences.append(new_sent)

    meta = {
        "family": "ocr_artifacts",
        "rate": rate,
        "seed": seed,
        "changed_positions": changed_positions,
        "candidate_positions": candidate_positions,
        "changed_fraction_of_candidates": changed_positions / candidate_positions if candidate_positions else 0.0,
        "noise_token_set": sorted(injected_tokens.keys()),
        "original_to_noisy": {k: dict(v) for k, v in original_to_noisy.items()},
    }
    return out_sentences, meta


# ---------- Noise family 3: orthographic standardisation ----------

ORTHO_VARIANTS = {
    "today": "to-day",
    "tomorrow": "to-morrow",
    "tonight": "to-night",
    "goodbye": "good-bye",
    "everyone": "every-one",
    "everybody": "every-body",
    "someone": "some-one",
    "something": "some-thing",
    "anyone": "any-one",
    "anything": "any-thing",
    "everything": "every-thing",
    "myself": "my-self",
    "yourself": "your-self",
    "himself": "him-self",
    "herself": "her-self",
    "themselves": "them-selves",
}

ORTHO_FAMILIES = {canon: {canon, variant} for canon, variant in ORTHO_VARIANTS.items()}


def inject_orthographic_variants(
    sentences: Sequence[Sequence[str]],
    rate: float,
    seed: int = 12345,
    variant_map: dict[str, str] | None = None,
) -> tuple[list[list[str]], dict]:
    rng = random.Random(seed)
    variant_map = ORTHO_VARIANTS if variant_map is None else variant_map
    out_sentences: list[list[str]] = []
    changed_positions = 0
    candidate_positions = 0
    used_families: Counter[str] = Counter()

    for sent in sentences:
        new_sent: list[str] = []
        for tok in sent:
            if tok in variant_map:
                candidate_positions += 1
                if rng.random() < rate:
                    tok = variant_map[tok]
                    changed_positions += 1
                    used_families[tok] += 1
            new_sent.append(tok)
        out_sentences.append(new_sent)

    meta = {
        "family": "orthographic_variants",
        "rate": rate,
        "seed": seed,
        "changed_positions": changed_positions,
        "candidate_positions": candidate_positions,
        "changed_fraction_of_candidates": changed_positions / candidate_positions if candidate_positions else 0.0,
        "variant_map": variant_map,
        "variant_families": {canon: sorted(list({canon, variant})) for canon, variant in variant_map.items()},
    }
    return out_sentences, meta


# ---------- Base/reference metrics ----------

def compute_clean_reference_stats(
    clean_sentences: Sequence[Sequence[str]],
    rare_threshold: int = 3,
) -> dict:
    tokens = flatten_sentences(clean_sentences)
    counts = Counter(tokens)
    vocab = set(counts)
    rare_vocab = {w for w, c in counts.items() if c <= rare_threshold}
    return {
        "clean_token_count": len(tokens),
        "clean_vocab": vocab,
        "clean_vocab_size": len(vocab),
        "clean_rare_vocab": rare_vocab,
        "clean_rare_vocab_size": len(rare_vocab),
        "clean_counts": counts,
    }


def noise_metrics(sentences: Sequence[Sequence[str]], noise_token_set: set[str]) -> dict[str, float | int]:
    tokens = flatten_sentences(sentences)
    counts = Counter(tokens)
    noise_mass = sum(counts.get(tok, 0) for tok in noise_token_set)
    noise_types_active = sum(1 for tok in noise_token_set if counts.get(tok, 0) > 0)
    return {
        "noise_token_fraction": noise_mass / len(tokens) if tokens else 0.0,
        "noise_token_count": noise_mass,
        "noise_type_count_active": noise_types_active,
    }


def orthographic_metrics(sentences: Sequence[Sequence[str]], variant_map: dict[str, str] | None = None) -> dict[str, float | int]:
    variant_map = ORTHO_VARIANTS if variant_map is None else variant_map
    tokens = flatten_sentences(sentences)
    counts = Counter(tokens)
    total_family_mass = 0
    canonical_mass = 0
    active_variant_forms = 0
    active_families = 0
    for canon, variant in variant_map.items():
        cm = counts.get(canon, 0)
        vm = counts.get(variant, 0)
        fam = cm + vm
        if fam > 0:
            active_families += 1
            total_family_mass += fam
            canonical_mass += cm
            if vm > 0:
                active_variant_forms += 1
    return {
        "orthographic_family_mass": total_family_mass,
        "orthographic_canonical_share": canonical_mass / total_family_mass if total_family_mass else 1.0,
        "orthographic_variant_share": 1.0 - (canonical_mass / total_family_mass if total_family_mass else 1.0),
        "orthographic_active_variant_forms": active_variant_forms,
        "orthographic_active_families": active_families,
    }


def clean_vocab_metrics(sentences: Sequence[Sequence[str]], clean_ref: dict) -> dict[str, float | int]:
    tokens = flatten_sentences(sentences)
    vocab = set(tokens)
    clean_vocab = clean_ref["clean_vocab"]
    clean_rare_vocab = clean_ref["clean_rare_vocab"]
    retained_clean = len(vocab & clean_vocab)
    retained_rare = len(vocab & clean_rare_vocab)
    return {
        "retained_clean_vocab_count": retained_clean,
        "retained_clean_vocab_share": retained_clean / len(clean_vocab) if clean_vocab else 0.0,
        "retained_rare_clean_vocab_count": retained_rare,
        "retained_rare_clean_vocab_share": retained_rare / len(clean_rare_vocab) if clean_rare_vocab else 0.0,
    }


def combined_metrics(
    sentences: Sequence[Sequence[str]],
    clean_ref: dict,
    family: str,
    family_meta: dict,
) -> dict[str, float | int]:
    base = corpus_metrics(sentences, original_token_count=clean_ref["clean_token_count"])
    out = dict(base)
    out.update(clean_vocab_metrics(sentences, clean_ref))
    if family == "ocr_artifacts":
        out.update(noise_metrics(sentences, set(family_meta.get("noise_token_set", []))))
    elif family == "orthographic_variants":
        out.update(orthographic_metrics(sentences, family_meta.get("variant_map")))
    return out


# ---------- Derived corpus creation ----------

def safe_rate_label(rate: float) -> str:
    return f"{100*rate:.1f}".replace(".", "p").replace("p0", "")


def build_derived_corpus_family(
    base_sentences: Sequence[Sequence[str]],
    family: str,
    rates: Sequence[float] = (0.005, 0.01, 0.02),
    base_seed: int = 12345,
    project_root: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    paths = theorem1_noise_paths(project_root)
    family_dir = ensure_dir(paths["derived"] / family)
    rows = []

    for idx, rate in enumerate(rates):
        rate_dir = ensure_dir(family_dir / f"rate_{safe_rate_label(rate)}")
        sent_path = rate_dir / "tokenized_sentences.txt.gz"
        meta_path = rate_dir / "manifest.json"

        if sent_path.exists() and meta_path.exists() and not overwrite:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(meta)
            continue

        if family == "ocr_artifacts":
            noisy_sentences, meta = inject_ocr_artifacts(base_sentences, rate=rate, seed=base_seed + idx)
        elif family == "orthographic_variants":
            noisy_sentences, meta = inject_orthographic_variants(base_sentences, rate=rate, seed=base_seed + idx)
        else:
            raise ValueError(f"Unknown family: {family}")

        save_sentence_state(sent_path, noisy_sentences)
        meta = {
            **meta,
            "family": family,
            "rate": rate,
            "seed": base_seed + idx,
            "sentence_count": len(noisy_sentences),
            "token_count": len(flatten_sentences(noisy_sentences)),
            "sentences_path": str(sent_path),
            "created_at": timestamp(),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(meta)

    df = pd.DataFrame(rows)
    df.to_csv(family_dir / "derived_corpora_summary.csv", index=False)
    return df


def load_derived_sentences(
    family: str,
    rate: float,
    project_root: str | Path | None = None,
) -> tuple[list[list[str]], dict]:
    paths = theorem1_noise_paths(project_root)
    rate_dir = paths["derived"] / family / f"rate_{safe_rate_label(rate)}"
    meta = json.loads((rate_dir / "manifest.json").read_text(encoding="utf-8"))
    sentences = load_sentence_state(meta["sentences_path"])
    return sentences, meta


# ---------- Checkpointed drift simulation ----------

@dataclass
class DriftRunConfig:
    corpus_label: str
    family: str
    rate: float
    alpha: float
    mode: str  # fixed_size_mix | growing_corpus
    generations: int = 10
    seed: int = 12345
    token_budget: int = 150_000
    export_top_k: int = 250
    notes: str = ""


def init_noise_db(db_path: str | Path) -> None:
    ensure_dir(Path(db_path).parent)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                corpus_label TEXT,
                family TEXT,
                rate REAL,
                alpha REAL,
                mode TEXT,
                generations INTEGER,
                seed INTEGER,
                token_budget INTEGER,
                notes TEXT,
                config_json TEXT
            );
            CREATE TABLE IF NOT EXISTS generation_metrics (
                run_id INTEGER,
                generation INTEGER,
                family TEXT,
                rate REAL,
                alpha REAL,
                mode TEXT,
                metric_json TEXT,
                PRIMARY KEY (run_id, generation)
            );
            """
        )


def register_noise_run(db_path: str | Path, config: DriftRunConfig) -> int:
    init_noise_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs(created_at, corpus_label, family, rate, alpha, mode, generations, seed, token_budget, notes, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp(),
                config.corpus_label,
                config.family,
                config.rate,
                config.alpha,
                config.mode,
                config.generations,
                config.seed,
                config.token_budget,
                config.notes,
                json.dumps(asdict(config), sort_keys=True),
            ),
        )
        return int(cur.lastrowid)


def _run_name(config: DriftRunConfig) -> str:
    rate = f"{config.rate:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    alpha = f"{config.alpha:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{config.corpus_label}__{config.family}__rate_{rate}__alpha_{alpha}__{config.mode}"


def _write_checkpoint(run_dir: Path, state: dict) -> None:
    (run_dir / "checkpoint_state.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _load_checkpoint(run_dir: Path) -> dict | None:
    path = run_dir / "checkpoint_state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_generation_artifacts(
    run_dir: Path,
    run_id: int,
    generation: int,
    sentences: Sequence[Sequence[str]],
    metric_row: dict,
    export_top_k: int = 250,
) -> None:
    state_path = run_dir / "generation_states" / f"generation_{generation:03d}.txt.gz"
    save_sentence_state(state_path, sentences)

    counts_dir = ensure_dir(run_dir / "counts_partial")
    tokens = flatten_sentences(sentences)
    word_counts = Counter(tokens)
    pd.DataFrame(word_counts.most_common(export_top_k), columns=["word", "count"]).to_csv(
        counts_dir / f"generation_{generation:03d}_top_words.csv", index=False
    )

    metrics_path = run_dir / "generation_metrics_partial.csv"
    frame = pd.DataFrame([metric_row])
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        old = old[old["generation"] != generation]
        frame = pd.concat([old, frame], ignore_index=True).sort_values("generation")
    frame.to_csv(metrics_path, index=False)


def _load_latest_generation_state(run_dir: Path) -> tuple[int, list[list[str]]] | tuple[None, None]:
    states_dir = run_dir / "generation_states"
    if not states_dir.exists():
        return None, None
    candidates = sorted(states_dir.glob("generation_*.txt.gz"))
    if not candidates:
        return None, None
    latest = candidates[-1]
    m = re.search(r"generation_(\d+)\.txt\.gz$", latest.name)
    if not m:
        return None, None
    gen = int(m.group(1))
    return gen, load_sentence_state(latest)


def simulate_drift_run(
    clean_anchor_sentences: Sequence[Sequence[str]],
    initial_sentences: Sequence[Sequence[str]],
    family_meta: dict,
    config: DriftRunConfig,
    project_root: str | Path | None = None,
    force_rebuild: bool = False,
) -> tuple[int, pd.DataFrame]:
    paths = theorem1_noise_paths(project_root)
    init_noise_db(paths["db_std"])
    clean_ref = compute_clean_reference_stats(clean_anchor_sentences)
    run_name = _run_name(config)
    run_dir = ensure_dir(paths["runs"] / run_name)
    ensure_dir(run_dir / "generation_states")
    ensure_dir(run_dir / "counts_partial")

    if force_rebuild:
        for p in run_dir.glob("*"):
            if p.is_file():
                p.unlink()

    checkpoint = _load_checkpoint(run_dir)
    if checkpoint and not force_rebuild:
        run_id = checkpoint["run_id"]
        last_gen, current_sentences = _load_latest_generation_state(run_dir)
        start_gen = (last_gen or 0) + 1
    else:
        run_id = register_noise_run(paths["db_std"], config)
        current_sentences = [list(sent) for sent in initial_sentences]
        start_gen = 0
        checkpoint = {
            "run_id": run_id,
            "run_name": run_name,
            "last_completed_generation": -1,
            "last_completed_stage": "initialized",
            "config": asdict(config),
        }
        _write_checkpoint(run_dir, checkpoint)

    base_token_count = len(flatten_sentences(clean_anchor_sentences))
    metrics_rows: list[dict] = []
    metrics_path = run_dir / "generation_metrics_partial.csv"
    if metrics_path.exists():
        try:
            metrics_rows = pd.read_csv(metrics_path).to_dict(orient="records")
        except Exception:
            metrics_rows = []

    gen_iter = tqdm(
        range(start_gen, config.generations + 1),
        desc=f"{config.family}:rate={config.rate:g}:alpha={config.alpha:g}",
        unit="gen",
    )

    for gen in gen_iter:
        metric = combined_metrics(current_sentences, clean_ref, config.family, family_meta)
        metric_row = {
            "run_id": run_id,
            "generation": gen,
            "family": config.family,
            "rate": config.rate,
            "alpha": config.alpha,
            "mode": config.mode,
            **metric,
        }
        _save_generation_artifacts(run_dir, run_id, gen, current_sentences, metric_row, export_top_k=config.export_top_k)

        with sqlite3.connect(paths["db_std"]) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO generation_metrics(run_id, generation, family, rate, alpha, mode, metric_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, gen, config.family, config.rate, config.alpha, config.mode, json.dumps(metric_row, sort_keys=True)),
            )

        checkpoint["last_completed_generation"] = gen
        checkpoint["last_completed_stage"] = "generation_metrics_saved"
        _write_checkpoint(run_dir, checkpoint)

        if gen == config.generations:
            break

        model = build_trigram_model(current_sentences)
        rng = random.Random(config.seed + gen + int(1000 * config.alpha))

        if config.mode == "fixed_size_mix":
            synthetic_tokens = int(round(config.alpha * base_token_count))
            anchor_tokens = max(0, base_token_count - synthetic_tokens)
            original_part = (
                sample_original_sentences_by_token_budget(clean_anchor_sentences, anchor_tokens, rng)
                if anchor_tokens > 0
                else []
            )
            synthetic_part = (
                generate_sentences_by_token_budget(model, synthetic_tokens, rng)
                if synthetic_tokens > 0
                else []
            )
            current_sentences = original_part + synthetic_part
            rng.shuffle(current_sentences)
        elif config.mode == "growing_corpus":
            add_tokens = int(round(config.alpha * base_token_count))
            synthetic_part = generate_sentences_by_token_budget(model, add_tokens, rng) if add_tokens > 0 else []
            current_sentences = [list(sent) for sent in current_sentences] + synthetic_part
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        checkpoint["last_completed_stage"] = "generation_advanced"
        _write_checkpoint(run_dir, checkpoint)

    final_df = pd.read_csv(metrics_path)
    final_df.to_csv(paths["metrics"] / f"{run_name}_metrics.csv", index=False)
    return run_id, final_df


def run_family_experiment_grid(
    clean_anchor_sentences: Sequence[Sequence[str]],
    family: str,
    rates: Sequence[float],
    alphas: Sequence[float],
    mode: str = "fixed_size_mix",
    generations: int = 10,
    token_budget: int = 150_000,
    base_seed: int = 12345,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    all_frames = []
    for ridx, rate in enumerate(tqdm(rates, desc=f"family={family}:rates", unit="rate")):
        initial_sentences, meta = load_derived_sentences(family, rate, project_root=project_root)
        for aidx, alpha in enumerate(tqdm(alphas, desc=f"{family}:alpha-grid", unit="alpha", leave=False)):
            cfg = DriftRunConfig(
                corpus_label="jane_austen",
                family=family,
                rate=float(rate),
                alpha=float(alpha),
                mode=mode,
                generations=generations,
                seed=base_seed + ridx * 100 + aidx,
                token_budget=token_budget,
                notes="Theorem 1 positive/negative family run",
            )
            run_id, df = simulate_drift_run(clean_anchor_sentences, initial_sentences, meta, cfg, project_root=project_root)
            df["run_id"] = run_id
            all_frames.append(df)
    out = pd.concat(all_frames, ignore_index=True)
    paths = theorem1_noise_paths(project_root)
    out.to_csv(paths["metrics"] / f"jane_austen__{family}__{mode}_grid_metrics.csv", index=False)
    return out


# ---------- Plotting ----------

def save_fig(fig, path_base: str | Path) -> None:
    path_base = Path(path_base)
    ensure_dir(path_base.parent)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=180, bbox_inches="tight")


def plot_family_grid(
    df: pd.DataFrame,
    family: str,
    out_path_base: str | Path | None = None,
) -> plt.Figure:
    rates = sorted(df["rate"].unique())
    metrics = (
        ["vocab_size", "retained_rare_clean_vocab_share", "noise_token_fraction", "noise_type_count_active"]
        if family == "ocr_artifacts"
        else ["vocab_size", "retained_rare_clean_vocab_share", "orthographic_canonical_share", "orthographic_active_variant_forms"]
    )
    titles = {
        "vocab_size": "Vocabulary size",
        "retained_rare_clean_vocab_share": "Rare clean-vocab retention",
        "noise_token_fraction": "Artifact token fraction",
        "noise_type_count_active": "Active artifact types",
        "orthographic_canonical_share": "Canonical orthography share",
        "orthographic_active_variant_forms": "Active variant forms",
    }

    fig, axes = plt.subplots(len(rates), len(metrics), figsize=(4.2 * len(metrics), 3.2 * len(rates)), squeeze=False)
    for r, rate in enumerate(rates):
        sub_r = df[df["rate"] == rate]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            for alpha, sub in sorted(sub_r.groupby("alpha"), key=lambda x: x[0]):
                ax.plot(sub["generation"], sub[metric], marker="o", label=f"α={alpha:g}")
            if r == 0:
                ax.set_title(titles.get(metric, metric))
            if c == 0:
                ax.set_ylabel(f"rate={100*rate:.1f}%")
            ax.set_xlabel("generation")
            ax.grid(alpha=0.3)
            if r == 0 and c == len(metrics) - 1:
                ax.legend(loc="best", title="mix")
    fig.suptitle(f"Theorem 1 family: {family.replace('_', ' ')}", y=1.02, fontsize=13)
    fig.tight_layout()
    if out_path_base is not None:
        save_fig(fig, out_path_base)
    return fig


def plot_combined_vocabulary(df_list: list[pd.DataFrame], labels: list[str], out_path_base: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for df, label in zip(df_list, labels):
        grouped = df.groupby("generation", as_index=False)["vocab_size"].mean()
        ax.plot(grouped["generation"], grouped["vocab_size"], marker="o", label=label)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean vocabulary size")
    ax.set_title("Vocabulary reduction across theorem-1 positive/negative families")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path_base is not None:
        save_fig(fig, out_path_base)
    return fig


def codex_instructions() -> str:
    return (
        "Place the notebook at `Drift_and_selection/GitHub/notebooks/active/19_theorem1_austen_noise_and_standardization.ipynb` "
        "and the helper at `Drift_and_selection/GitHub/src/drift_selection/theorem1_noise_standardization.py`. "
        "Keep original cleaned Austen text untouched. Save derived corpora under "
        "`Drift_and_selection/GitHub/data/outputs/theorem1_standardization/derived_corpora/` with separate folders for "
        "`ocr_artifacts` and `orthographic_variants` at rates 0.5%, 1%, and 2%. "
        "Save run checkpoints under `.../runs/`, metrics under `.../metrics/`, figures under "
        "`Drift_and_selection/GitHub/figures/appendix/theorem1_standardization/`, and SQLite metadata in "
        "`Drift_and_selection/GitHub/data/databases/theorem1_standardization.sqlite`."
    )
