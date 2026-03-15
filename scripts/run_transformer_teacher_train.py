#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.database import export_registry_csv, init_experiment_registry, log_artifact, log_run_metric, register_run
from drift_selection.transformer import TransformerConfig, build_tiny_gpt, save_model_config
from drift_selection.training import TrainingConfig, train_language_model
from drift_selection.transformer_pipeline import ensure_tokenizer_and_encoded_splits, load_main_config, resolve_path
from drift_selection.utils import ensure_dir, timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description="Train tiny transformer teacher on Conan Doyle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_teacher_student.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_main_config(root, args.config)

    paths_cfg = cfg["paths"]
    out_root = ensure_dir(resolve_path(root, paths_cfg["root_outputs_dir"]))
    teacher_dir = ensure_dir(out_root / "teacher")
    db_path = resolve_path(root, paths_cfg["db_path"])

    prep = ensure_tokenizer_and_encoded_splits(root, cfg, force=args.force)
    tokenizer = prep["tokenizer"]
    ids = prep["encoded_ids"]

    model_cfg = TransformerConfig(vocab_size=int(tokenizer.vocab_size), **cfg["model"])
    save_model_config(model_cfg, teacher_dir / "model_config.json")

    summary_path = teacher_dir / "training_summary.json"
    if summary_path.exists() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        hist = summary.get("history", {})
        print(f"Teacher training already exists at {summary_path}; skipping (use --force to retrain)")
        print(f"Best checkpoint: {hist.get('best_checkpoint_path')}")
        if hist.get("val_loss_history"):
            print(f"Final val loss: {hist['val_loss_history'][-1].get('val_loss')}")
        return 0

    model = build_tiny_gpt(model_cfg)
    train_cfg = TrainingConfig(**cfg["training"])
    history = train_language_model(
        model=model,
        train_ids=ids["train"],
        val_ids=ids["val"],
        train_cfg=train_cfg,
        out_dir=teacher_dir,
        resume=not args.force,
        progress_bar=True,
    )

    payload = {
        "created_at": timestamp(),
        "config_path": args.config,
        "model_config": model_cfg.__dict__,
        "training_config": train_cfg.__dict__,
        "history": history,
        "tokenization_manifest": str(prep["manifest_path"]),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    init_experiment_registry(db_path)
    run_id = register_run(
        db_path=db_path,
        run_name="transformer_teacher_train",
        script_or_notebook="run_transformer_teacher_train.py",
        config_path=args.config,
        metadata={
            "timestamp_start": payload["created_at"],
            "timestamp_end": timestamp(),
            "corpus": "conan_doyle",
            "model_family": "tiny_gpt",
            "selection_rule": "teacher",
            "seed": int(train_cfg.seed),
            "output_dir": str(teacher_dir.relative_to(root)),
            "metrics_summary": json.dumps({"best_checkpoint": history.get("best_checkpoint_path")}),
            "status": "completed",
        },
    )
    log_artifact(db_path, run_id, "teacher_summary", str(summary_path.relative_to(root)))
    if history.get("best_checkpoint_path"):
        log_artifact(db_path, run_id, "teacher_best_checkpoint", str(Path(history["best_checkpoint_path"]).relative_to(root)))
    if history.get("val_loss_history"):
        log_run_metric(db_path, run_id, "final_val_loss", float(history["val_loss_history"][-1]["val_loss"]), split="val")

    export_registry_csv(db_path, resolve_path(root, "GitHub/data/outputs/csv/experiment_registry.csv"))

    print(f"Teacher training complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
