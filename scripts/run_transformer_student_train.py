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
from drift_selection.environments import load_environment_tokens
from drift_selection.training import TrainingConfig, train_language_model
from drift_selection.transformer import TransformerConfig, build_tiny_gpt, save_model_config
from drift_selection.transformer_pipeline import ensure_tokenizer_and_encoded_splits, load_main_config, resolve_path
from drift_selection.utils import ensure_dir, timestamp


def _env_token_path(env_dir: Path, env_name: str) -> Path:
    if env_name == "original":
        return env_dir / "original" / "original_environment_token_ids.npy"
    if env_name == "neutral":
        return env_dir / "neutral" / "neutral_token_ids.npy"
    if env_name == "selected":
        return env_dir / "selected" / "selected_token_ids.npy"
    raise ValueError(env_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train students on original/neutral/selected environments")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_teacher_student.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_main_config(root, args.config)
    prep = ensure_tokenizer_and_encoded_splits(root, cfg, force=False)
    tokenizer = prep["tokenizer"]
    split_ids = prep["encoded_ids"]

    paths_cfg = cfg["paths"]
    out_root = ensure_dir(resolve_path(root, paths_cfg["root_outputs_dir"]))
    students_root = ensure_dir(out_root / "students")
    env_root = out_root / "environments"

    model_cfg = TransformerConfig(vocab_size=int(tokenizer.vocab_size), **cfg["model"])
    student_train_cfg = TrainingConfig(**cfg["student_training"])

    db_path = resolve_path(root, paths_cfg["db_path"])
    init_experiment_registry(db_path)

    for env_name in ["original", "neutral", "selected"]:
        token_path = _env_token_path(env_root, env_name)
        if not token_path.exists():
            raise FileNotFoundError(f"Environment tokens missing: {token_path}")

        env_ids = load_environment_tokens(token_path)
        student_dir = ensure_dir(students_root / env_name)
        summary_path = student_dir / "training_summary.json"
        save_model_config(model_cfg, student_dir / "model_config.json")

        if summary_path.exists() and not args.force:
            print(f"Skipping {env_name}: summary exists")
            continue

        model = build_tiny_gpt(model_cfg)
        history = train_language_model(
            model=model,
            train_ids=env_ids,
            val_ids=split_ids["val"],
            train_cfg=student_train_cfg,
            out_dir=student_dir,
            resume=not args.force,
            progress_bar=True,
        )

        payload = {
            "created_at": timestamp(),
            "mode": args.mode,
            "environment": env_name,
            "environment_token_path": str(token_path),
            "token_count": len(env_ids),
            "model_config": model_cfg.__dict__,
            "training_config": student_train_cfg.__dict__,
            "history": history,
        }
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        run_id = register_run(
            db_path=db_path,
            run_name=f"transformer_student_train_{env_name}_{args.mode}",
            script_or_notebook="run_transformer_student_train.py",
            config_path=args.config,
            metadata={
                "timestamp_start": payload["created_at"],
                "timestamp_end": timestamp(),
                "corpus": "conan_doyle",
                "model_family": "tiny_gpt_student",
                "selection_rule": env_name,
                "seed": int(student_train_cfg.seed),
                "output_dir": str(student_dir.relative_to(root)),
                "metrics_summary": json.dumps({"best_checkpoint": history.get("best_checkpoint_path")}),
                "status": "completed",
            },
        )
        log_artifact(db_path, run_id, "student_summary", str(summary_path.relative_to(root)))
        if history.get("best_checkpoint_path"):
            log_artifact(db_path, run_id, "student_best_checkpoint", str(Path(history["best_checkpoint_path"]).relative_to(root)))
        if history.get("val_loss_history"):
            log_run_metric(db_path, run_id, "final_val_loss", float(history["val_loss_history"][-1]["val_loss"]), split="val")

    export_registry_csv(db_path, resolve_path(root, "GitHub/data/outputs/csv/experiment_registry.csv"))
    print("Student training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
