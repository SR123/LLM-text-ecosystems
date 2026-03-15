from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any, Sequence

from tqdm.auto import tqdm

from .checkpoints import atomic_save_json, load_pickle_checkpoint, save_pickle_checkpoint
from .theorem2_selected_publication import (
    SyntheticBackoffGrammar,
    SyntheticGrammarConfig,
    CountBackoffModel,
)
from .utils import ensure_dir, stable_slug, timestamp


@dataclass(frozen=True)
class RegenerationConfig:
    text_length: int = 1000
    generations: int = 20
    alpha: float = 0.25
    restart_probability: float = 0.0
    sample_retained_block: bool = True

    def __post_init__(self) -> None:
        if self.text_length < 8:
            raise ValueError("text_length must be >= 8")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
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
class SyntheticRegenerationRun:
    version: str
    run_name: str
    paths: RunPaths
    manifest: dict[str, Any]
    metrics_rows: list[dict[str, Any]]
    final_tokens: tuple[int, ...]


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
    return ensure_dir(root / "GitHub" / "data" / "outputs" / "theorem1_synthetic_regeneration")


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
    fieldnames = list(rows[0].keys())
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
    path = snapshot_path(state_dir, generation)
    return [int(token) for token in load_pickle_checkpoint(path)]


def distinct_ngram_count(tokens: Sequence[int], n: int) -> int:
    if n == 1:
        return len(set(int(token) for token in tokens))
    if len(tokens) < n:
        return 0
    return len({tuple(int(x) for x in tokens[i:i + n]) for i in range(len(tokens) - n + 1)})


def token_entropy_bits(tokens: Sequence[int]) -> float:
    if not tokens:
        return 0.0
    counts: dict[int, int] = {}
    for token in tokens:
        counts[int(token)] = counts.get(int(token), 0) + 1
    total = float(len(tokens))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * (0.0 if p <= 0.0 else __import__("math").log2(p))
    return float(entropy)


def metrics_for_generation(
    tokens: Sequence[int],
    *,
    generation: int,
    alpha: float,
    baseline_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    row = {
        "generation": int(generation),
        "text_length": int(len(tokens)),
        "alpha": float(alpha),
        "vocab_size": int(distinct_ngram_count(tokens, 1)),
        "distinct_2grams": int(distinct_ngram_count(tokens, 2)),
        "distinct_3grams": int(distinct_ngram_count(tokens, 3)),
        "token_entropy_bits": float(token_entropy_bits(tokens)),
    }
    if baseline_metrics:
        row["vocab_ratio_vs_gen0"] = row["vocab_size"] / float(baseline_metrics["vocab_size"] or 1.0)
        row["distinct_2gram_ratio_vs_gen0"] = row["distinct_2grams"] / float(baseline_metrics["distinct_2grams"] or 1.0)
        row["distinct_3gram_ratio_vs_gen0"] = row["distinct_3grams"] / float(baseline_metrics["distinct_3grams"] or 1.0)
    else:
        row["vocab_ratio_vs_gen0"] = 1.0
        row["distinct_2gram_ratio_vs_gen0"] = 1.0
        row["distinct_3gram_ratio_vs_gen0"] = 1.0
    return row


def sample_retained_tokens(tokens: Sequence[int], retain_length: int, rng: random.Random, contiguous: bool = True) -> list[int]:
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


def random_seed_trigram(tokens: Sequence[int], rng: random.Random) -> list[int]:
    seq = [int(token) for token in tokens]
    if not seq:
        raise ValueError("tokens must not be empty")
    if len(seq) <= 3:
        return list(seq)
    start = rng.randint(0, len(seq) - 3)
    return seq[start:start + 3]


def build_empirical_backoff_model(tokens: Sequence[int]) -> CountBackoffModel:
    model = CountBackoffModel(vocab_size=max(int(max(tokens, default=1)), 1), order=3)
    model.fit_sequences([tuple(int(token) for token in tokens)])
    return model


def generate_replacement_tokens(
    source_tokens: Sequence[int],
    target_length: int,
    *,
    rng: random.Random,
    restart_probability: float = 0.0,
) -> list[int]:
    target_length = int(target_length)
    if target_length <= 0:
        return []
    seq = [int(token) for token in source_tokens]
    if not seq:
        raise ValueError("source_tokens must not be empty")
    model = build_empirical_backoff_model(seq)
    seed = random_seed_trigram(seq, rng)
    out = seed[:min(3, target_length)]
    while len(out) < target_length:
        if restart_probability > 0.0 and len(out) >= 3 and rng.random() < restart_probability:
            seed = random_seed_trigram(seq, rng)
            remaining = target_length - len(out)
            out.extend(seed[:min(3, remaining)])
            continue
        token, _ = model.sample_next(out[-2:], rng=rng)
        out.append(int(token))
    return out


def next_generation_text(
    current_tokens: Sequence[int],
    config: RegenerationConfig,
    *,
    rng: random.Random,
) -> list[int]:
    retain_length = int(round((1.0 - config.alpha) * config.text_length))
    retain_length = max(0, min(retain_length, config.text_length))
    replacement_length = config.text_length - retain_length
    retained = sample_retained_tokens(
        current_tokens,
        retain_length,
        rng,
        contiguous=config.sample_retained_block,
    )
    generated = generate_replacement_tokens(
        current_tokens,
        replacement_length,
        rng=rng,
        restart_probability=config.restart_probability,
    )
    combined = retained + generated
    if len(combined) < config.text_length:
        combined.extend(generate_replacement_tokens(current_tokens, config.text_length - len(combined), rng=rng, restart_probability=config.restart_probability))
    return [int(token) for token in combined[:config.text_length]]


def _title_suffix(manifest: dict[str, Any]) -> str:
    return f"{manifest['version']} | {manifest['run_name']} | {manifest.get('updated_at', manifest['created_at'])}"


def _save_metric_figures(paths: RunPaths, rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    generations = [int(row["generation"]) for row in rows]
    saved: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    axes[0].plot(generations, [float(row["vocab_size"]) for row in rows], marker="o")
    axes[1].plot(generations, [float(row["distinct_2grams"]) for row in rows], marker="o")
    axes[2].plot(generations, [float(row["distinct_3grams"]) for row in rows], marker="o")
    axes[0].set_title("Vocabulary size")
    axes[1].set_title("Distinct 2-grams")
    axes[2].set_title("Distinct 3-grams")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Count")
    fig.suptitle(_title_suffix(manifest))
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    main_path = paths.figures_dir / f"{stable_slug(manifest['version'])}_support_counts.png"
    fig.savefig(main_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(main_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(generations, [float(row["vocab_ratio_vs_gen0"]) for row in rows], marker="o")
    axes[0].plot(generations, [float(row["distinct_2gram_ratio_vs_gen0"]) for row in rows], marker="o")
    axes[0].plot(generations, [float(row["distinct_3gram_ratio_vs_gen0"]) for row in rows], marker="o")
    axes[0].legend(["vocab/gen0", "2-grams/gen0", "3-grams/gen0"])
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
    return saved


def _write_sample_texts(paths: RunPaths, snapshots: dict[int, Sequence[int]]) -> None:
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
        sample = " ".join(str(int(token)) for token in snapshots[generation][:80])
        lines.append(f"[generation {generation}]")
        lines.append(sample)
        lines.append("")
    paths.sample_text_path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_artifacts(paths: RunPaths, manifest: dict[str, Any], rows: Sequence[dict[str, Any]], snapshots: dict[int, Sequence[int]], *, final: bool) -> None:
    _write_rows_csv(paths.metrics_partial_path, rows)
    if final:
        _write_rows_csv(paths.metrics_final_path, rows)
    figure_paths = _save_metric_figures(paths, rows, manifest)
    _write_sample_texts(paths, snapshots)
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
    }
    atomic_save_json(paths.summary_path, _json_ready(summary))


def _build_manifest(
    *,
    run_name: str,
    version: str,
    seed: int,
    grammar_config: SyntheticGrammarConfig,
    regeneration_config: RegenerationConfig,
) -> dict[str, Any]:
    now = timestamp()
    return {
        "run_name": run_name,
        "version": version,
        "seed": int(seed),
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "grammar_config": _json_ready(asdict(grammar_config)),
        "regeneration_config": _json_ready(asdict(regeneration_config)),
    }


def _load_or_create_manifest(
    paths: RunPaths,
    *,
    run_name: str,
    version: str,
    seed: int,
    grammar_config: SyntheticGrammarConfig,
    regeneration_config: RegenerationConfig,
    resume: bool,
) -> dict[str, Any]:
    expected = {
        "run_name": run_name,
        "version": version,
        "seed": int(seed),
        "grammar_config": _json_ready(asdict(grammar_config)),
        "regeneration_config": _json_ready(asdict(regeneration_config)),
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
        run_name=run_name,
        version=version,
        seed=seed,
        grammar_config=grammar_config,
        regeneration_config=regeneration_config,
    )
    atomic_save_json(paths.manifest_path, _json_ready(manifest))
    return manifest


def _load_metrics_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "generation": int(row["generation"]),
                    "text_length": int(row["text_length"]),
                    "alpha": float(row["alpha"]),
                    "vocab_size": int(row["vocab_size"]),
                    "distinct_2grams": int(row["distinct_2grams"]),
                    "distinct_3grams": int(row["distinct_3grams"]),
                    "token_entropy_bits": float(row["token_entropy_bits"]),
                    "vocab_ratio_vs_gen0": float(row["vocab_ratio_vs_gen0"]),
                    "distinct_2gram_ratio_vs_gen0": float(row["distinct_2gram_ratio_vs_gen0"]),
                    "distinct_3gram_ratio_vs_gen0": float(row["distinct_3gram_ratio_vs_gen0"]),
                }
            )
    return rows


def _baseline_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    first = min(rows, key=lambda row: int(row["generation"]))
    return {
        "vocab_size": float(first["vocab_size"]),
        "distinct_2grams": float(first["distinct_2grams"]),
        "distinct_3grams": float(first["distinct_3grams"]),
    }


def run_synthetic_regeneration(
    *,
    version: str,
    run_name: str,
    grammar_config: SyntheticGrammarConfig,
    regeneration_config: RegenerationConfig,
    seed: int = 123,
    output_root: Path | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
    progress_bar: bool = True,
) -> SyntheticRegenerationRun:
    output_root = default_output_root(output_root) if output_root is None else ensure_dir(output_root)
    paths = ensure_run_paths(output_root=output_root, run_name=run_name, version=version)
    if force_rebuild:
        _remove_tree_contents(paths.run_dir)
        paths = ensure_run_paths(output_root=output_root, run_name=run_name, version=version)

    manifest = _load_or_create_manifest(
        paths,
        run_name=run_name,
        version=version,
        seed=seed,
        grammar_config=grammar_config,
        regeneration_config=regeneration_config,
        resume=resume and not force_rebuild,
    )

    grammar = SyntheticBackoffGrammar(grammar_config)
    rng = random.Random(seed)

    metrics_rows: list[dict[str, Any]] = []
    snapshots: dict[int, Sequence[int]] = {}
    start_generation = 0

    if resume and not force_rebuild and paths.metrics_partial_path.exists() and paths.checkpoint_path.exists():
        metrics_rows = _load_metrics_rows(paths.metrics_partial_path)
        checkpoint = json.loads(paths.checkpoint_path.read_text(encoding="utf-8"))
        last_completed_generation = int(checkpoint.get("last_completed_generation", -1))
        if last_completed_generation >= 0:
            start_generation = last_completed_generation + 1
            if start_generation <= regeneration_config.generations:
                current_tokens = load_generation_snapshot(paths.state_dir, last_completed_generation)
            else:
                current_tokens = load_generation_snapshot(paths.state_dir, regeneration_config.generations)
            for generation in {0, last_completed_generation, regeneration_config.generations}:
                snap = snapshot_path(paths.state_dir, generation)
                if snap.exists():
                    snapshots[generation] = load_generation_snapshot(paths.state_dir, generation)
        else:
            current_tokens = []
    else:
        current_tokens = []

    if not current_tokens:
        initial_tokens, _ = grammar.generate(regeneration_config.text_length, rng=rng)
        current_tokens = [int(token) for token in initial_tokens]
        save_generation_snapshot(paths.state_dir, 0, current_tokens)
        baseline_row = metrics_for_generation(current_tokens, generation=0, alpha=regeneration_config.alpha, baseline_metrics=None)
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
        _write_run_artifacts(paths, manifest, metrics_rows, snapshots, final=False)
        start_generation = 1

    if start_generation > regeneration_config.generations:
        manifest["status"] = "finished"
        atomic_save_json(paths.manifest_path, _json_ready(manifest))
        return SyntheticRegenerationRun(
            version=version,
            run_name=run_name,
            paths=paths,
            manifest=manifest,
            metrics_rows=metrics_rows,
            final_tokens=tuple(int(token) for token in current_tokens),
        )

    baseline = _baseline_metrics(metrics_rows)
    iterator = range(start_generation, regeneration_config.generations + 1)
    iterator = tqdm(iterator, desc=f"regen:{run_name}", unit="gen", disable=not progress_bar)
    for generation in iterator:
        generation_rng = random.Random(seed + 1009 * generation)
        current_tokens = next_generation_text(current_tokens, regeneration_config, rng=generation_rng)
        save_generation_snapshot(paths.state_dir, generation, current_tokens)
        row = metrics_for_generation(
            current_tokens,
            generation=generation,
            alpha=regeneration_config.alpha,
            baseline_metrics=baseline,
        )
        metrics_rows = [existing for existing in metrics_rows if int(existing["generation"]) != generation]
        metrics_rows.append(row)
        metrics_rows.sort(key=lambda item: int(item["generation"]))
        if generation in {0, regeneration_config.generations} or generation == regeneration_config.generations // 2:
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
        _write_run_artifacts(paths, manifest, metrics_rows, snapshots, final=False)
        iterator.set_postfix(
            vocab=int(row["vocab_size"]),
            tri=int(row["distinct_3grams"]),
            vratio=f"{row['vocab_ratio_vs_gen0']:.3f}",
        )

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
                "last_completed_generation": regeneration_config.generations,
                "updated_at": manifest["updated_at"],
            }
        ),
    )

    for generation in (0, regeneration_config.generations // 2, regeneration_config.generations):
        snap = snapshot_path(paths.state_dir, generation)
        if snap.exists():
            snapshots[generation] = load_generation_snapshot(paths.state_dir, generation)

    _write_run_artifacts(paths, manifest, metrics_rows, snapshots, final=True)
    return SyntheticRegenerationRun(
        version=version,
        run_name=run_name,
        paths=paths,
        manifest=manifest,
        metrics_rows=metrics_rows,
        final_tokens=tuple(int(token) for token in current_tokens),
    )
