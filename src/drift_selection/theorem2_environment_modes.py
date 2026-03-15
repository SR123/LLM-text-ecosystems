from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .theorem2_process_learning import (
    ModelConfig,
    TrainConfig,
    SimpleTokenVocab,
    build_vocab_from_datasets,
    carry_needed,
    digits_of,
    direct_clean_sequence_ok,
    ensure_dir,
    evaluate_direct_answer_mode,
    extract_answer_digits_flexible,
    find_project_root,
    generate_response,
    get_device,
    is_structurally_valid_trace,
    no_carry_sum_digits,
    normalize_digit_string,
    plot_training_histories,
    repair_marker_structure_ok,
    save_json,
    save_plot,
    sequence_logprob_after_prompt,
    set_seed,
    timestamp,
    TinyCausalTransformer,
    train_style_models,
    unique_pairs,
    addition_steps,
)


STYLES = ["artifact_only", "worked_trace", "failed_then_repair"]
PROMPT_MODES = ["DIRECT", "WORK"]


@dataclass
class EnvironmentModesConfig:
    train_min_digits: int = 2
    train_max_digits: int = 3
    id_test_min_digits: int = 2
    id_test_max_digits: int = 3
    ood_test_min_digits: int = 4
    ood_test_max_digits: int = 4
    n_train: int = 9000
    n_val: int = 1200
    n_id_test: int = 2000
    n_ood_test: int = 1600
    seed: int = 321
    equalize_examples_to_longest: bool = True
    include_both_prompt_modes_in_train: bool = True


def default_paths(project_root: Optional[Path] = None) -> dict:
    root = find_project_root(project_root)
    gh = root / "GitHub"
    return {
        "project_root": root,
        "output_root": ensure_dir(gh / "data" / "outputs" / "theorem2_process_learning" / "taskB_environment_modes_v2"),
        "figure_root": ensure_dir(gh / "figures" / "appendix" / "theorem2_process_learning"),
        "notebook_root": ensure_dir(gh / "notebooks" / "active"),
        "src_root": ensure_dir(gh / "src" / "drift_selection"),
    }


def prompt_tokens_for_add_mode(a: int, b: int, prompt_mode: str) -> List[str]:
    return ["TASK", "TASKB_ENV_ADD", "A", *digits_of(a), "SEP", "B", *digits_of(b), "MODE", prompt_mode]


def _force_wrong_attempt(no_carry_digits: List[str], gold_digits: List[str]) -> List[str]:
    attempt = list(no_carry_digits)
    if attempt != gold_digits:
        return attempt
    if not attempt:
        return ["0"]
    attempt[-1] = str((int(attempt[-1]) + 1) % 10)
    return attempt


def make_environment_example(a: int, b: int, style: str, prompt_mode: str, split_name: str, example_id: str) -> dict:
    prompt = prompt_tokens_for_add_mode(a, b, prompt_mode)
    trace, ans_digits = addition_steps(a, b)
    need_carry = carry_needed(a, b)
    failed_attempt = _force_wrong_attempt(no_carry_sum_digits(a, b), ans_digits)

    if style == "artifact_only":
        target = ["ANS", *ans_digits]
    elif style == "worked_trace":
        target = ["WORK", *trace, "ANS", *ans_digits]
    elif style == "failed_then_repair":
        target = ["TRY", "NOCARRY", *failed_attempt, "FAIL", "REPAIR", *trace, "ANS", *ans_digits]
    else:
        raise ValueError(f"Unknown style={style!r}")

    return {
        "example_id": example_id,
        "split_name": split_name,
        "style": style,
        "prompt_mode": prompt_mode,
        "a": a,
        "b": b,
        "carry_needed": need_carry,
        "answer_digits": ans_digits,
        "failed_attempt_digits": failed_attempt,
        "prompt_tokens": prompt,
        "target_tokens": target,
    }


def build_problem_splits(cfg: EnvironmentModesConfig) -> dict:
    return {
        "train": unique_pairs(cfg.n_train, cfg.train_min_digits, cfg.train_max_digits, cfg.seed),
        "val": unique_pairs(cfg.n_val, cfg.train_min_digits, cfg.train_max_digits, cfg.seed + 1),
        "id_test": unique_pairs(cfg.n_id_test, cfg.id_test_min_digits, cfg.id_test_max_digits, cfg.seed + 2),
        "ood_test": unique_pairs(cfg.n_ood_test, cfg.ood_test_min_digits, cfg.ood_test_max_digits, cfg.seed + 3),
    }


def build_environment_datasets(cfg: EnvironmentModesConfig, styles: Optional[List[str]] = None) -> Tuple[dict, dict]:
    styles = styles or list(STYLES)
    problems = build_problem_splits(cfg)
    datasets: Dict[str, Dict[str, List[dict]]] = {style: {} for style in styles}
    train_modes = list(PROMPT_MODES) if cfg.include_both_prompt_modes_in_train else ["DIRECT"]

    for style in styles:
        train_examples: List[dict] = []
        val_examples: List[dict] = []
        id_direct: List[dict] = []
        id_process: List[dict] = []
        ood_direct: List[dict] = []
        ood_process: List[dict] = []

        for idx, (a, b) in enumerate(problems["train"]):
            for prompt_mode in train_modes:
                train_examples.append(make_environment_example(a, b, style, prompt_mode, "train", f"train_{idx:06d}_{prompt_mode.lower()}"))
        for idx, (a, b) in enumerate(problems["val"]):
            for prompt_mode in train_modes:
                val_examples.append(make_environment_example(a, b, style, prompt_mode, "val", f"val_{idx:06d}_{prompt_mode.lower()}"))
        for idx, (a, b) in enumerate(problems["id_test"]):
            id_direct.append(make_environment_example(a, b, style, "DIRECT", "id_test", f"id_{idx:06d}_direct"))
            id_process.append(make_environment_example(a, b, style, "WORK", "id_test", f"id_{idx:06d}_work"))
        for idx, (a, b) in enumerate(problems["ood_test"]):
            ood_direct.append(make_environment_example(a, b, style, "DIRECT", "ood_test", f"ood_{idx:06d}_direct"))
            ood_process.append(make_environment_example(a, b, style, "WORK", "ood_test", f"ood_{idx:06d}_work"))

        datasets[style]["train"] = train_examples
        datasets[style]["val"] = val_examples
        datasets[style]["id_test_direct"] = id_direct
        datasets[style]["id_test_process"] = id_process
        datasets[style]["ood_test_direct"] = ood_direct
        datasets[style]["ood_test_process"] = ood_process

    manifest = {
        "config": asdict(cfg),
        "train_prompt_modes": train_modes,
        "problem_splits": {
            split_name: [
                {
                    "problem_id": f"{split_name}_{idx:06d}",
                    "a": a,
                    "b": b,
                    "answer": a + b,
                    "carry_needed": carry_needed(a, b),
                }
                for idx, (a, b) in enumerate(pairs)
            ]
            for split_name, pairs in problems.items()
        },
    }
    return datasets, manifest


def save_examples_jsonl(path: Path, examples: List[dict]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(json.dumps(x) for x in examples), encoding="utf-8")


def save_dataset_bundle(base_dir: Path, datasets_by_style: Dict[str, Dict[str, List[dict]]], manifest: dict, cfg: EnvironmentModesConfig) -> None:
    ensure_dir(base_dir)
    save_json(base_dir / "taskB_problem_manifest.json", manifest)
    save_json(base_dir / "environment_modes_config.json", asdict(cfg))
    for style, splits in datasets_by_style.items():
        for split_name, examples in splits.items():
            save_examples_jsonl(base_dir / f"{style}_{split_name}.jsonl", examples)


def pretty_print_examples(examples: List[dict], n: int = 3) -> str:
    rows = []
    for ex in examples[:n]:
        rows.append("PROMPT: " + " ".join(ex["prompt_tokens"]))
        rows.append("TARGET: " + " ".join(ex["target_tokens"]))
        rows.append("")
    return "\n".join(rows)


def prompt_budget_summary(datasets_by_style: Dict[str, Dict[str, List[dict]]]) -> pd.DataFrame:
    rows = []
    for style, splits in datasets_by_style.items():
        for split_name, examples in splits.items():
            prompt_lens = [len(ex["prompt_tokens"]) for ex in examples]
            target_lens = [len(ex["target_tokens"]) for ex in examples]
            rows.append(
                {
                    "style": style,
                    "split": split_name,
                    "n_examples": len(examples),
                    "avg_prompt_len": float(np.mean(prompt_lens)) if prompt_lens else 0.0,
                    "avg_target_len": float(np.mean(target_lens)) if target_lens else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _augment_rows(ev_rows: pd.DataFrame, examples: List[dict]) -> pd.DataFrame:
    df = ev_rows.copy()
    df["prompt"] = [" ".join(ex["prompt_tokens"]) for ex in examples]
    df["target"] = [" ".join(ex["target_tokens"]) for ex in examples]
    df["prompt_mode"] = [ex["prompt_mode"] for ex in examples]
    df["split_name"] = [ex["split_name"] for ex in examples]
    return df


def evaluate_process_mode_from_prompt(
    model: TinyCausalTransformer,
    vocab: SimpleTokenVocab,
    examples: List[dict],
    max_new_tokens: int = 192,
) -> dict:
    rows = []
    final_correct = 0
    trace_valid = 0
    repair_acc = 0
    answer_present = 0

    for ex in tqdm(examples, desc="eval-process-mode", leave=False):
        pred_tokens = generate_response(model, vocab, ex["prompt_tokens"], max_new_tokens=max_new_tokens)
        gold = "".join(ex["answer_digits"])
        pred = extract_answer_digits_flexible(pred_tokens)
        expected_cols = max(len(str(ex["a"])), len(str(ex["b"])))

        final_ok = normalize_digit_string(pred) == normalize_digit_string(gold)
        trace_ok = is_structurally_valid_trace(pred_tokens, expected_columns=expected_cols) and final_ok
        repair_ok = repair_marker_structure_ok(pred_tokens)
        answer_in_completion = pred is not None

        final_correct += int(final_ok)
        trace_valid += int(trace_ok)
        repair_acc += int(repair_ok)
        answer_present += int(answer_in_completion)

        rows.append(
            {
                "example_id": ex["example_id"],
                "a": ex["a"],
                "b": ex["b"],
                "gold_answer": gold,
                "pred_answer": pred,
                "process_final_answer_accuracy": final_ok,
                "process_trace_validity": trace_ok,
                "repair_marker_accuracy": repair_ok,
                "process_answer_in_completion": answer_in_completion,
                "prediction_tokens": " ".join(pred_tokens),
            }
        )

    n = max(len(rows), 1)
    return {
        "process_final_answer_accuracy": final_correct / n,
        "process_trace_validity": trace_valid / n,
        "repair_marker_accuracy": repair_acc / n,
        "process_answer_in_completion": answer_present / n,
        "rows": pd.DataFrame(rows),
    }


def _write_sample_csv(path: Path, df: pd.DataFrame, n: int = 20, seed: int = 0) -> None:
    ensure_dir(path.parent)
    if df.empty:
        df.to_csv(path, index=False)
        return
    sample_n = min(n, len(df))
    df.sample(sample_n, random_state=seed).to_csv(path, index=False)


def _grouped_bars(ax, df: pd.DataFrame, value_col: str, title: str) -> None:
    styles = list(df["style"].drop_duplicates())
    splits = list(df["split"].drop_duplicates())
    x = np.arange(len(styles))
    width = 0.35 if len(splits) == 2 else 0.6
    for idx, split_name in enumerate(splits):
        split_df = df[df["split"] == split_name].set_index("style").reindex(styles).reset_index()
        offset = (idx - (len(splits) - 1) / 2) * width
        ax.bar(x + offset, split_df[value_col], width=width, label=split_name)
    ax.set_xticks(x)
    ax.set_xticklabels(styles, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    if len(splits) > 1:
        ax.legend(fontsize=8)


def plot_direct_metrics(summary_df: pd.DataFrame, figure_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _grouped_bars(axes[0], summary_df, "direct_exact_match", "direct_exact_match")
    _grouped_bars(axes[1], summary_df, "direct_clean_answer_rate", "direct_clean_answer_rate")
    fig.tight_layout()
    save_plot(fig, figure_base)
    plt.close(fig)


def plot_process_metrics(summary_df: pd.DataFrame, figure_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _grouped_bars(axes[0], summary_df, "process_final_answer_accuracy", "process_final_answer_accuracy")
    _grouped_bars(axes[1], summary_df, "process_trace_validity", "process_trace_validity")
    fig.tight_layout()
    save_plot(fig, figure_base)
    plt.close(fig)


def plot_direct_contamination(summary_df: pd.DataFrame, figure_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _grouped_bars(ax, summary_df, "direct_marker_contamination_rate", "direct_marker_contamination_rate")
    fig.tight_layout()
    save_plot(fig, figure_base)
    plt.close(fig)


def find_task_a_summary_csv(project_root: Optional[Path] = None) -> Optional[Path]:
    root = find_project_root(project_root)
    base = root / "GitHub" / "data" / "outputs" / "theorem2_process_learning"
    candidates = sorted(base.glob("theorem2_taskA_addition_process_vs_artifact*/evaluation/summary_metrics.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_appendix_assets(
    appendix_dir: Path,
    task_a_summary_csv: Optional[Path],
    direct_summary: pd.DataFrame,
    process_summary: pd.DataFrame,
) -> dict:
    ensure_dir(appendix_dir)

    task_a_rows = []
    if task_a_summary_csv and Path(task_a_summary_csv).exists():
        task_a_df = pd.read_csv(task_a_summary_csv)
        for _, row in task_a_df.iterrows():
            task_a_rows.append(
                {
                    "task": "Task A",
                    "mode": "process_vs_artifact",
                    "split": "test",
                    "style": row.get("style"),
                    "answer_accuracy": row.get("answer_accuracy"),
                    "answer_accuracy_strict": row.get("answer_accuracy_strict"),
                    "direct_exact_match": np.nan,
                    "direct_clean_answer_rate": np.nan,
                    "direct_marker_contamination_rate": np.nan,
                    "process_final_answer_accuracy": np.nan,
                    "process_trace_validity": np.nan,
                }
            )

    task_b_direct = direct_summary.copy()
    task_b_direct.insert(0, "task", "Task B")
    task_b_direct.insert(1, "mode", "direct")
    task_b_direct["answer_accuracy"] = np.nan
    task_b_direct["answer_accuracy_strict"] = np.nan
    task_b_direct["process_final_answer_accuracy"] = np.nan
    task_b_direct["process_trace_validity"] = np.nan

    task_b_process = process_summary.copy()
    task_b_process.insert(0, "task", "Task B")
    task_b_process.insert(1, "mode", "process")
    task_b_process["answer_accuracy"] = np.nan
    task_b_process["answer_accuracy_strict"] = np.nan
    task_b_process["direct_exact_match"] = np.nan
    task_b_process["direct_clean_answer_rate"] = np.nan
    task_b_process["direct_marker_contamination_rate"] = np.nan

    combined = pd.concat([pd.DataFrame(task_a_rows), task_b_direct, task_b_process], ignore_index=True, sort=False)
    combined_csv = appendix_dir / "theorem2_combined_summary.csv"
    combined.to_csv(combined_csv, index=False)

    display_cols = [
        "task",
        "mode",
        "split",
        "style",
        "answer_accuracy",
        "direct_exact_match",
        "direct_clean_answer_rate",
        "direct_marker_contamination_rate",
        "process_final_answer_accuracy",
        "process_trace_validity",
    ]
    latex_df = combined[display_cols].copy()
    latex_path = appendix_dir / "table_theorem2_combined_summary.tex"
    latex_path.write_text(
        latex_df.to_latex(index=False, na_rep="", float_format=lambda x: f"{x:.3f}"),
        encoding="utf-8",
    )
    return {"combined_csv": combined_csv, "table_tex": latex_path}


def run_environment_modes_workflow(
    env_cfg: EnvironmentModesConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    run_name: str = "theorem2_taskB_environment_modes_v2",
    project_root: Optional[Path] = None,
    task_a_summary_csv: Optional[Path] = None,
    build_appendix_assets_flag: bool = False,
) -> dict:
    roots = default_paths(project_root)
    run_root = ensure_dir(roots["output_root"] / run_name)
    datasets_dir = ensure_dir(run_root / "datasets")
    models_dir = ensure_dir(run_root / "models")
    evaluation_dir = ensure_dir(run_root / "evaluation")
    appendix_dir = ensure_dir(run_root / "appendix_assets")

    datasets, manifest = build_environment_datasets(env_cfg, styles=list(STYLES))
    save_dataset_bundle(datasets_dir, datasets, manifest, env_cfg)

    vocab = build_vocab_from_datasets(datasets)
    vocab.save(run_root / "vocab.json")

    trained = train_style_models(
        task_family="taskB_environment_modes_v2",
        datasets_by_style=datasets,
        vocab=vocab,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        out_root=models_dir,
    )

    direct_rows = []
    process_rows = []
    for style in STYLES:
        model = trained[style]["model"]
        for split_name in ["id_test", "ood_test"]:
            direct_examples = datasets[style][f"{split_name}_direct"]
            process_examples = datasets[style][f"{split_name}_process"]

            direct_ev = evaluate_direct_answer_mode(model, vocab, direct_examples, max_new_tokens=48)
            direct_df = _augment_rows(direct_ev["rows"], direct_examples)
            direct_path = evaluation_dir / f"{style}_{split_name}_direct_predictions.csv"
            direct_df.to_csv(direct_path, index=False)
            _write_sample_csv(evaluation_dir / f"sample_generations_{style}_{split_name}_direct.csv", direct_df, n=20, seed=11)
            direct_rows.append(
                {
                    "style": style,
                    "split": split_name,
                    "direct_exact_match": direct_ev["direct_exact_match"],
                    "direct_answer_in_prefix": direct_ev["direct_answer_in_prefix"],
                    "direct_clean_answer_rate": direct_ev["direct_clean_answer_rate"],
                    "direct_marker_contamination_rate": direct_ev["direct_marker_contamination_rate"],
                    "direct_answer_logprob": direct_ev["direct_answer_logprob"],
                }
            )

            process_ev = evaluate_process_mode_from_prompt(model, vocab, process_examples, max_new_tokens=192)
            process_df = _augment_rows(process_ev["rows"], process_examples)
            process_path = evaluation_dir / f"{style}_{split_name}_process_predictions.csv"
            process_df.to_csv(process_path, index=False)
            _write_sample_csv(evaluation_dir / f"sample_generations_{style}_{split_name}_process.csv", process_df, n=20, seed=17)
            process_rows.append(
                {
                    "style": style,
                    "split": split_name,
                    "process_final_answer_accuracy": process_ev["process_final_answer_accuracy"],
                    "process_trace_validity": process_ev["process_trace_validity"],
                    "repair_marker_accuracy": process_ev["repair_marker_accuracy"],
                    "process_answer_in_completion": process_ev["process_answer_in_completion"],
                }
            )

    direct_summary = pd.DataFrame(direct_rows)
    process_summary = pd.DataFrame(process_rows)
    direct_summary.to_csv(evaluation_dir / "summary_direct_mode.csv", index=False)
    process_summary.to_csv(evaluation_dir / "summary_process_mode.csv", index=False)

    direct_fig = roots["figure_root"] / f"{run_name}_direct_metrics"
    process_fig = roots["figure_root"] / f"{run_name}_process_metrics"
    contamination_fig = roots["figure_root"] / f"{run_name}_direct_contamination"
    plot_direct_metrics(direct_summary, direct_fig)
    plot_process_metrics(process_summary, process_fig)
    plot_direct_contamination(direct_summary, contamination_fig)
    plot_training_histories(trained, roots["figure_root"] / f"{run_name}_training", title=run_name)

    appendix_paths = {"combined_csv": None, "table_tex": None}
    task_a_path = Path(task_a_summary_csv) if task_a_summary_csv else find_task_a_summary_csv(project_root)
    if build_appendix_assets_flag:
        appendix_paths = build_appendix_assets(appendix_dir, task_a_path, direct_summary, process_summary)

    save_json(
        run_root / "run_manifest.json",
        {
            "run_name": run_name,
            "timestamp": timestamp(),
            "device": str(get_device()),
            "env_cfg": asdict(env_cfg),
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "styles": list(STYLES),
            "task_a_summary_csv": str(task_a_path) if task_a_path else None,
            "summary_direct_mode": str(evaluation_dir / "summary_direct_mode.csv"),
            "summary_process_mode": str(evaluation_dir / "summary_process_mode.csv"),
            "figure_direct_metrics": str(direct_fig.with_suffix(".pdf")),
            "figure_process_metrics": str(process_fig.with_suffix(".pdf")),
            "figure_direct_contamination": str(contamination_fig.with_suffix(".pdf")),
            "appendix_assets": {
                "combined_csv": str(appendix_paths["combined_csv"]) if appendix_paths["combined_csv"] else None,
                "table_tex": str(appendix_paths["table_tex"]) if appendix_paths["table_tex"] else None,
            },
        },
    )

    return {
        "run_root": run_root,
        "evaluation_dir": evaluation_dir,
        "appendix_dir": appendix_dir,
        "direct_summary": direct_summary,
        "process_summary": process_summary,
        "task_a_summary_csv": task_a_path,
        "figure_paths": {
            "direct_metrics": direct_fig.with_suffix(".pdf"),
            "process_metrics": process_fig.with_suffix(".pdf"),
            "direct_contamination": contamination_fig.with_suffix(".pdf"),
        },
        "appendix_paths": appendix_paths,
    }


def codex_instructions() -> str:
    return (
        "Task B environment modes workflow installed. "
        "Run preview first, then full pipeline with appendix assets enabled."
    )
