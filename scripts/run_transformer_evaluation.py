#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.database import export_registry_csv, init_experiment_registry, log_artifact, log_run_metric, register_run
from drift_selection.metrics import build_subword_trigram_table, evaluate_all_students, heldout_loss
from drift_selection.plots import plot_selected_decoding_example, plot_teacher_student_bar_chart, plot_training_curves
from drift_selection.selected_decoding import SelectedDecodingConfig, compare_greedy_vs_selected_next_token
from drift_selection.training import load_trained_model
from drift_selection.transformer import load_model_config
from drift_selection.transformer_pipeline import (
    ensure_tokenizer_and_encoded_splits,
    load_main_config,
    load_prompt_bank,
    load_selected_cfg,
    resolve_path,
    write_prompt_bank,
)
from drift_selection.utils import ensure_dir, timestamp


def _load_model_from_summary(summary_path: Path, device):
    info = json.loads(summary_path.read_text(encoding="utf-8"))
    ckpt = info["history"]["best_checkpoint_path"] or info["history"]["final_checkpoint_path"]
    model_cfg = load_model_config(summary_path.parent / "model_config.json")
    model = load_trained_model(Path(ckpt), model_cfg, device)
    return model, info, ckpt


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate teacher/student policy agreement and build figures")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_teacher_student.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_main_config(root, args.config)

    prep = ensure_tokenizer_and_encoded_splits(root, cfg, force=False)
    tokenizer = prep["tokenizer"]
    split_ids = prep["encoded_ids"]

    paths_cfg = cfg["paths"]
    out_root = resolve_path(root, paths_cfg["root_outputs_dir"])
    eval_root = ensure_dir(out_root / "evaluation")
    eval_csv_dir = ensure_dir(eval_root / "csv")
    eval_json_dir = ensure_dir(eval_root / "json")

    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    teacher_summary = out_root / "teacher" / "training_summary.json"
    if not teacher_summary.exists():
        raise FileNotFoundError("Teacher summary missing. Run teacher training first.")
    teacher, teacher_info, teacher_ckpt = _load_model_from_summary(teacher_summary, device)

    students = {}
    histories = {"teacher": teacher_info["history"]}
    for env in ["original", "neutral", "selected"]:
        s_summary = out_root / "students" / env / "training_summary.json"
        if not s_summary.exists():
            raise FileNotFoundError(f"Student summary missing: {s_summary}")
        model, info, _ = _load_model_from_summary(s_summary, device)
        students[env] = model
        histories[env] = info["history"]

    prompt_bank_dir = out_root / "manifests"
    prompts_path = prompt_bank_dir / f"evaluation_prompts_{args.mode}.json"
    legacy_prompts_path = prompt_bank_dir / f"evaluation_prompts_{args.mode}_prompt_bank.json"
    if not prompts_path.exists() and legacy_prompts_path.exists():
        prompts_path = legacy_prompts_path
    if not prompts_path.exists():
        eval_n_prompts = int(cfg["evaluation"]["pilot_prompt_count"] if args.mode == "pilot" else cfg["evaluation"]["full_prompt_count"])
        prompts_path = write_prompt_bank(
            out_dir=prompt_bank_dir,
            tokenizer=tokenizer,
            token_ids=split_ids["test"],
            prompt_len=int(cfg["environments"]["prompt_len"]),
            n_prompts=eval_n_prompts,
            seed=int(cfg["seeds"]["prompts"]) + 1,
            name=f"evaluation_prompts_{args.mode}",
        )
        print(f"Evaluation prompt bank was missing and has been generated at: {prompts_path}")
    prompt_bank = load_prompt_bank(prompts_path)

    selected_cfg = SelectedDecodingConfig(**load_selected_cfg(root, cfg["configs"]["selected_decoding_config_path"]))
    trigram_table = build_subword_trigram_table(split_ids["train"])

    metrics_df = evaluate_all_students(
        teacher=teacher,
        students=students,
        prompt_bank=prompt_bank,
        selected_cfg=selected_cfg,
        trigram_table=trigram_table,
        device=device,
    )

    # add heldout losses
    heldout_batches = int(cfg["evaluation"].get("heldout_eval_batches", 40))
    for i, row in metrics_df.iterrows():
        env = row["student"]
        h_loss = heldout_loss(
            model=students[env],
            test_ids=split_ids["test"],
            batch_size=int(cfg["student_training"]["batch_size"]),
            context_len=int(cfg["student_training"]["context_len"]),
            eval_batches=heldout_batches,
            device=device,
        )
        metrics_df.loc[i, "heldout_loss"] = float(h_loss)

    metrics_csv = eval_csv_dir / f"teacher_student_metrics_{args.mode}.csv"
    metrics_json = eval_json_dir / f"teacher_student_metrics_{args.mode}.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(metrics_df.to_json(orient="records", indent=2) + "\n", encoding="utf-8")

    # Main bar chart
    figure_base = resolve_path(root, paths_cfg["figures_paper_dir"]) / "transformer_teacher_student_bars"
    plot_teacher_student_bar_chart(metrics_df, figure_base, title=f"Teacher/Student Policy Metrics ({args.mode})")

    # Training curves appendix
    curve_base = resolve_path(root, paths_cfg["figures_appendix_dir"]) / "transformer_training_curves"
    plot_training_curves(histories, curve_base, title="Teacher and Student Training Curves")

    # Selected decoding diagnostic example
    demo = compare_greedy_vs_selected_next_token(
        model=teacher,
        prompt_ids=prompt_bank[0],
        cfg=selected_cfg,
        device=device,
    )
    demo_base = resolve_path(root, paths_cfg["figures_appendix_dir"]) / "selected_decoding_examples"
    plot_selected_decoding_example(demo, tokenizer, demo_base)

    # evaluation manifest
    manifest = {
        "created_at": timestamp(),
        "mode": args.mode,
        "teacher_checkpoint": teacher_ckpt,
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "paper_figure_base": str(figure_base),
        "appendix_curve_base": str(curve_base),
        "appendix_selected_demo_base": str(demo_base),
    }
    manifest_path = eval_root / f"evaluation_manifest_{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # registry logging
    db_path = resolve_path(root, paths_cfg["db_path"])
    init_experiment_registry(db_path)
    run_id = register_run(
        db_path=db_path,
        run_name=f"transformer_evaluation_{args.mode}",
        script_or_notebook="run_transformer_evaluation.py",
        config_path=args.config,
        metadata={
            "timestamp_start": manifest["created_at"],
            "timestamp_end": timestamp(),
            "corpus": "conan_doyle",
            "model_family": "tiny_gpt",
            "selection_rule": "evaluation",
            "seed": int(cfg["seeds"]["master"]),
            "output_dir": str(eval_root.relative_to(root)),
            "metrics_summary": json.dumps({"rows": int(len(metrics_df))}),
            "status": "completed",
        },
    )
    log_artifact(db_path, run_id, "metrics_csv", str(metrics_csv.relative_to(root)))
    log_artifact(db_path, run_id, "metrics_json", str(metrics_json.relative_to(root)))
    for _, row in metrics_df.iterrows():
        for col in ["mean_selected_probability", "top1_match_rate", "mean_branch_score", "heldout_loss"]:
            if col in row and pd.notna(row[col]):
                log_run_metric(db_path, run_id, f"{row['student']}.{col}", float(row[col]))

    export_registry_csv(db_path, resolve_path(root, "GitHub/data/outputs/csv/experiment_registry.csv"))

    print(f"Evaluation complete: {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
