
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from theorem2_process_learning import (
    find_project_root,
    ensure_dir,
    timestamp,
    save_json,
    set_seed,
    ModelConfig,
    TrainConfig,
    SimpleTokenVocab,
    collect_vocab_tokens,
    train_style_models,
    sample_generation_examples,
    plot_training_histories,
    save_plot,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class TilingTaskConfig:
    train_min_n: int = 6
    train_max_n: int = 10
    test_min_n: int = 11
    test_max_n: int = 14
    n_train: int = 9000
    n_val: int = 1200
    n_test: int = 2200
    seed: int = 123
    n_failed_attempts: int = 2
    equalize_examples_to_longest: bool = True
    deterministic_success: bool = True


PIECES = {
    'V': 1,   # vertical domino (fills one column)
    'HH': 2,  # two horizontal dominoes stacked
    'SQ': 2,  # 2x2 square
}
VALID_PIECES = list(PIECES.keys())
SPECIAL_STYLE_TOKENS = ['TRY', 'FAIL', 'SOL', 'OVER', 'SHORT', 'REM']


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def default_output_roots(project_root: Optional[Path] = None) -> dict:
    root = find_project_root(project_root)
    gh = root / 'GitHub'
    return {
        'data_out': ensure_dir(gh / 'data' / 'outputs' / 'theorem2_artifact_learning'),
        'fig_out': ensure_dir(gh / 'figures' / 'appendix' / 'theorem2_artifact_learning'),
        'nb_out': ensure_dir(gh / 'notebooks' / 'active'),
        'src_out': ensure_dir(gh / 'src' / 'drift_selection'),
    }


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def digits_of(n: int) -> List[str]:
    return list(str(n))


def prompt_tokens_for_tiling(n: int) -> List[str]:
    return ['TASK', 'TILE', 'N', *digits_of(n), 'SOLVE']


def random_success_tiling(n: int) -> List[str]:
    pieces = []
    rem = n
    while rem > 0:
        options = [p for p, w in PIECES.items() if w <= rem]
        p = random.choice(options)
        pieces.append(p)
        rem -= PIECES[p]
    return pieces


def canonical_success_tiling(n: int) -> List[str]:
    # Deterministic decomposition so the model has a stable target per width.
    pieces = []
    rem = n
    while rem >= 2:
        pieces.append('HH')
        rem -= 2
    if rem == 1:
        pieces.append('V')
    return pieces


def random_failed_attempt(n: int) -> Tuple[List[str], str]:
    # Produce a plausible failed attempt that either overshoots or stops short.
    mode = random.choice(['over', 'short'])
    pieces = []
    total = 0
    if mode == 'over':
        while total <= n:
            p = random.choice(VALID_PIECES)
            pieces.append(p)
            total += PIECES[p]
        return pieces, 'OVER'
    else:
        # Stop early with a positive remainder.
        while total < n:
            valid = [p for p in VALID_PIECES if total + PIECES[p] < n]
            if not valid:
                break
            p = random.choice(valid)
            pieces.append(p)
            total += PIECES[p]
            if total > 0 and random.random() < 0.35:
                break
        rem = max(1, n - total)
        return pieces + ['REM', *digits_of(rem)], 'SHORT'


def make_tiling_example(
    n: int,
    style: str = 'artifact_only',
    n_failed_attempts: int = 2,
    deterministic_success: bool = True,
) -> dict:
    prompt = prompt_tokens_for_tiling(n)
    success = canonical_success_tiling(n) if deterministic_success else random_success_tiling(n)
    artifact = ['SOL', *success]

    if style == 'artifact_only':
        target = artifact
    elif style == 'mixed_failed':
        target = []
        for _ in range(n_failed_attempts):
            fail_tokens, fail_reason = random_failed_attempt(n)
            target.extend(['TRY', *fail_tokens, 'FAIL', fail_reason])
        target.extend(artifact)
    else:
        raise ValueError(style)

    return {
        'task_family': 'tiling_artifact_learning',
        'style': style,
        'n': n,
        'success_tokens': success,
        'prompt_tokens': prompt,
        'target_tokens': target,
    }


def sample_ns(n_examples: int, min_n: int, max_n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randint(min_n, max_n) for _ in range(n_examples)]


def _token_budget(examples: List[dict]) -> int:
    return sum(len(ex['target_tokens']) for ex in examples)


def build_tiling_datasets(styles: List[str], cfg: TilingTaskConfig) -> Dict[str, Dict[str, List[dict]]]:
    set_seed(cfg.seed)
    train_ns = sample_ns(cfg.n_train, cfg.train_min_n, cfg.train_max_n, cfg.seed)
    val_ns = sample_ns(cfg.n_val, cfg.train_min_n, cfg.train_max_n, cfg.seed + 1)
    test_ns = sample_ns(cfg.n_test, cfg.test_min_n, cfg.test_max_n, cfg.seed + 2)

    out: Dict[str, Dict[str, List[dict]]] = {s: {} for s in styles}
    for style in styles:
        out[style]['train'] = [
            make_tiling_example(
                n,
                style=style,
                n_failed_attempts=cfg.n_failed_attempts,
                deterministic_success=cfg.deterministic_success,
            )
            for n in train_ns
        ]
        out[style]['val'] = [
            make_tiling_example(
                n,
                style=style,
                n_failed_attempts=cfg.n_failed_attempts,
                deterministic_success=cfg.deterministic_success,
            )
            for n in val_ns
        ]
        out[style]['test'] = [
            make_tiling_example(
                n,
                style=style,
                n_failed_attempts=cfg.n_failed_attempts,
                deterministic_success=cfg.deterministic_success,
            )
            for n in test_ns
        ]

    if cfg.equalize_examples_to_longest:
        max_train_tokens = max(_token_budget(out[s]['train']) for s in styles)
        max_val_tokens = max(_token_budget(out[s]['val']) for s in styles)
        for split, budget in [('train', max_train_tokens), ('val', max_val_tokens)]:
            for style in styles:
                rows = out[style][split]
                tok_count = _token_budget(rows)
                idx = 0
                while tok_count < budget and rows:
                    clone = dict(rows[idx % len(rows)])
                    clone['replicated'] = True
                    rows.append(clone)
                    tok_count += len(clone['target_tokens'])
                    idx += 1
    return out


# ---------------------------------------------------------------------
# Save/load helpers
# ---------------------------------------------------------------------

def save_examples_jsonl(path: Path, examples: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')


def save_dataset_bundle(path: Path, datasets: Dict[str, Dict[str, List[dict]]], task_cfg: TilingTaskConfig) -> None:
    ensure_dir(path)
    for style, splits in datasets.items():
        for split_name, examples in splits.items():
            save_examples_jsonl(path / f'{style}_{split_name}.jsonl', examples)
    save_json(path / 'dataset_config.json', asdict(task_cfg))


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def parse_last_solution(tokens: List[str]) -> List[str]:
    if 'SOL' not in tokens:
        return []
    idx = max(i for i, tok in enumerate(tokens) if tok == 'SOL')
    out = []
    for tok in tokens[idx + 1:]:
        if tok in VALID_PIECES:
            out.append(tok)
        else:
            break
    return out


def tiling_width(piece_tokens: List[str]) -> int:
    return sum(PIECES.get(tok, 0) for tok in piece_tokens)


def is_valid_tiling(piece_tokens: List[str], n: int) -> bool:
    return len(piece_tokens) > 0 and all(tok in PIECES for tok in piece_tokens) and tiling_width(piece_tokens) == n


def is_pure_solution(tokens: List[str], n: int) -> bool:
    if not tokens:
        return False
    if any(tok in {'TRY', 'FAIL', 'OVER', 'SHORT', 'REM'} for tok in tokens):
        return False
    if tokens[0] == 'SOL':
        piece_tokens = [tok for tok in tokens[1:] if tok in VALID_PIECES]
        if len(piece_tokens) != len(tokens) - 1:
            return False
        return is_valid_tiling(piece_tokens, n)
    piece_tokens = [tok for tok in tokens if tok in VALID_PIECES]
    return len(piece_tokens) == len(tokens) and is_valid_tiling(piece_tokens, n)


def generate_response(model, vocab, prompt_tokens: List[str], max_new_tokens: int = 80) -> List[str]:
    from theorem2_process_learning import generate_response as _gen
    return _gen(model, vocab, prompt_tokens, max_new_tokens=max_new_tokens)


def evaluate_tiling_artifact_learning(model, vocab, test_examples: List[dict]) -> dict:
    rows = []
    for ex in tqdm(test_examples, desc='eval-tiling', leave=False):
        response = generate_response(model, vocab, ex['prompt_tokens'], max_new_tokens=80)
        last_sol = parse_last_solution(response)
        valid_last = is_valid_tiling(last_sol, ex['n'])
        pure_valid = is_pure_solution(response, ex['n'])
        contains_fail = any(tok in {'TRY', 'FAIL', 'OVER', 'SHORT'} for tok in response)
        rows.append({
            'n': ex['n'],
            'style': ex['style'],
            'prompt_text': ' '.join(ex['prompt_tokens']),
            'target_text': ' '.join(ex['target_tokens']),
            'response_text': ' '.join(response),
            'valid_last_solution': int(valid_last),
            'pure_valid_solution': int(pure_valid),
            'contains_failure_markers': int(contains_fail),
            'target_solution_text': ' '.join(ex['success_tokens']),
            'parsed_last_solution_text': ' '.join(last_sol),
        })
    df = pd.DataFrame(rows)
    return {
        'valid_last_solution_rate': float(df['valid_last_solution'].mean()) if not df.empty else 0.0,
        'pure_valid_solution_rate': float(df['pure_valid_solution'].mean()) if not df.empty else 0.0,
        'failure_marker_rate': float(df['contains_failure_markers'].mean()) if not df.empty else 0.0,
        'rows': df,
    }


def sample_generation_examples_for_style(model, vocab, test_examples: List[dict], n: int = 12) -> pd.DataFrame:
    rows = []
    for ex in test_examples[:n]:
        resp = generate_response(model, vocab, ex['prompt_tokens'], max_new_tokens=80)
        rows.append({
            'n': ex['n'],
            'prompt': ' '.join(ex['prompt_tokens']),
            'target': ' '.join(ex['target_tokens']),
            'response': ' '.join(resp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Vocab / training / evaluation wrappers
# ---------------------------------------------------------------------

def build_vocab_from_datasets(datasets: Dict[str, Dict[str, List[dict]]]) -> SimpleTokenVocab:
    tokens = []
    for style, splits in datasets.items():
        for split_name, examples in splits.items():
            tokens.extend(collect_vocab_tokens(examples))
    return SimpleTokenVocab(tokens + SPECIAL_STYLE_TOKENS + ['TASK', 'TILE', 'N', 'SOLVE'])


def evaluate_task_family(trained: Dict[str, dict], vocab: SimpleTokenVocab, datasets_by_style: Dict[str, Dict[str, List[dict]]], out_root: Path) -> pd.DataFrame:
    ensure_dir(out_root)
    rows = []
    for style, obj in trained.items():
        model = obj['model']
        test_examples = datasets_by_style[style]['test']
        ev = evaluate_tiling_artifact_learning(model, vocab, test_examples)
        metrics = {
            'style': style,
            'task_family': 'tiling_artifact_learning',
            'valid_last_solution_rate': ev['valid_last_solution_rate'],
            'pure_valid_solution_rate': ev['pure_valid_solution_rate'],
            'failure_marker_rate': ev['failure_marker_rate'],
        }
        ev['rows'].to_csv(out_root / f'{style}_test_predictions.csv', index=False)
        per_n = (
            ev['rows']
            .groupby('n', as_index=False)[['valid_last_solution', 'pure_valid_solution', 'contains_failure_markers']]
            .mean()
            .rename(columns={
                'valid_last_solution': 'valid_last_solution_rate',
                'pure_valid_solution': 'pure_valid_solution_rate',
                'contains_failure_markers': 'failure_marker_rate',
            })
        )
        per_n.to_csv(out_root / f'{style}_per_n_metrics.csv', index=False)
        sample_df = sample_generation_examples_for_style(model, vocab, test_examples, n=12)
        sample_df.to_csv(out_root / f'{style}_samples.csv', index=False)
        rows.append(metrics)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'summary_metrics.csv', index=False)
    return df


def plot_task_family_metrics(df: pd.DataFrame, out_path_base: Path, title: str):
    metric_cols = [c for c in ['valid_last_solution_rate', 'pure_valid_solution_rate', 'failure_marker_rate'] if c in df.columns]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4.6 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, metric_cols):
        ax.bar(df['style'], df[col])
        ax.set_ylim(0, 1.0)
        ax.set_title(col.replace('_', ' '))
        ax.tick_params(axis='x', rotation=20)
    fig.suptitle(title)
    fig.tight_layout()
    save_plot(fig, out_path_base)
    plt.close(fig)


def run_tiling_pipeline(
    styles: List[str],
    task_cfg: TilingTaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    run_name: str,
    project_root: Optional[Path] = None,
) -> dict:
    roots = default_output_roots(project_root)
    run_root = ensure_dir(roots['data_out'] / run_name)
    datasets_dir = ensure_dir(run_root / 'datasets')
    models_dir = ensure_dir(run_root / 'models')
    eval_dir = ensure_dir(run_root / 'evaluation')

    datasets = build_tiling_datasets(styles, task_cfg)
    save_dataset_bundle(datasets_dir, datasets, task_cfg)

    vocab = build_vocab_from_datasets(datasets)
    vocab.save(run_root / 'vocab.json')

    trained = train_style_models('tiling_artifact_learning', datasets, vocab, model_cfg, train_cfg, models_dir)
    metrics_df = evaluate_task_family(trained, vocab, datasets, eval_dir)

    plot_task_family_metrics(metrics_df, roots['fig_out'] / f'{run_name}_metrics', title=run_name)
    plot_training_histories(trained, roots['fig_out'] / f'{run_name}_training', title=f'{run_name} training')

    save_json(run_root / 'run_manifest.json', {
        'task_family': 'tiling_artifact_learning',
        'styles': styles,
        'task_cfg': asdict(task_cfg),
        'model_cfg': asdict(model_cfg),
        'train_cfg': asdict(train_cfg),
        'run_name': run_name,
        'timestamp': timestamp(),
    })

    return {
        'run_root': run_root,
        'metrics_df': metrics_df,
        'datasets': datasets,
        'trained': trained,
        'vocab': vocab,
    }


def pretty_print_examples(examples: List[dict], n: int = 3) -> str:
    rows = []
    for ex in examples[:n]:
        rows.append('PROMPT: ' + ' '.join(ex['prompt_tokens']))
        rows.append('TARGET: ' + ' '.join(ex['target_tokens']))
        rows.append('')
    return '\n'.join(rows)
