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

from drift_selection.database import export_registry_csv, init_experiment_registry, log_artifact, register_run
from drift_selection.environments import (
    EnvironmentGenerationConfig,
    build_environment_from_prompts,
    generate_original_environment_subset,
    save_environment_tokens,
)
from drift_selection.selected_decoding import SelectedDecodingConfig
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate original/neutral/selected environments")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_teacher_student.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_main_config(root, args.config)
    prep = ensure_tokenizer_and_encoded_splits(root, cfg, force=False)
    tokenizer = prep["tokenizer"]
    ids = prep["encoded_ids"]

    paths_cfg = cfg["paths"]
    out_root = ensure_dir(resolve_path(root, paths_cfg["root_outputs_dir"]))
    env_root = ensure_dir(out_root / "environments")
    prompt_root = ensure_dir(out_root / "manifests")

    env_cfg = cfg["environments"]
    total_tokens = int(env_cfg["pilot_total_tokens"] if args.mode == "pilot" else env_cfg["full_total_tokens"])
    prompt_len = int(env_cfg["prompt_len"])
    gen_n_prompts = int(env_cfg["generation_prompt_count"])
    eval_n_prompts = int(cfg["evaluation"]["pilot_prompt_count"] if args.mode == "pilot" else cfg["evaluation"]["full_prompt_count"])

    generation_prompt_path = prompt_root / f"generation_prompts_{args.mode}.json"
    evaluation_prompt_path = prompt_root / f"evaluation_prompts_{args.mode}.json"

    if args.force or not generation_prompt_path.exists():
        generation_prompt_path = write_prompt_bank(
            out_dir=prompt_root,
            tokenizer=tokenizer,
            token_ids=ids["train"],
            prompt_len=prompt_len,
            n_prompts=gen_n_prompts,
            seed=int(cfg["seeds"]["prompts"]),
            name=f"generation_prompts_{args.mode}",
        )
    if args.force or not evaluation_prompt_path.exists():
        evaluation_prompt_path = write_prompt_bank(
            out_dir=prompt_root,
            tokenizer=tokenizer,
            token_ids=ids["test"],
            prompt_len=prompt_len,
            n_prompts=eval_n_prompts,
            seed=int(cfg["seeds"]["prompts"]) + 1,
            name=f"evaluation_prompts_{args.mode}",
        )

    generation_prompts = load_prompt_bank(generation_prompt_path)

    teacher_dir = out_root / "teacher"
    teacher_summary = teacher_dir / "training_summary.json"
    if not teacher_summary.exists():
        raise FileNotFoundError("Teacher summary not found. Run run_transformer_teacher_train.py first.")
    teacher_info = json.loads(teacher_summary.read_text(encoding="utf-8"))
    teacher_ckpt = teacher_info["history"]["best_checkpoint_path"] or teacher_info["history"]["final_checkpoint_path"]
    model_cfg = load_model_config(teacher_dir / "model_config.json")

    import torch

    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    teacher = load_trained_model(Path(teacher_ckpt), model_cfg, device)

    # Original environment
    original_dir = ensure_dir(env_root / "original")
    original_tokens = generate_original_environment_subset(ids["train"], total_tokens=total_tokens, seed=int(cfg["seeds"]["environments"]))
    original_manifest = save_environment_tokens(original_dir, "original_environment", original_tokens, tokenizer)

    # Neutral environment
    neutral_dir = ensure_dir(env_root / "neutral")
    neutral_manifest_path = neutral_dir / "neutral_environment_manifest.json"
    if args.force or not neutral_manifest_path.exists():
        neutral_cfg = EnvironmentGenerationConfig(
            mode="neutral",
            total_tokens=total_tokens,
            prompt_len=prompt_len,
            temperature=float(env_cfg["neutral_temperature"]),
            top_k=env_cfg.get("neutral_top_k"),
            top_p=env_cfg.get("neutral_top_p"),
            seed=int(cfg["seeds"]["environments"]) + 1,
            max_new_tokens_per_prompt=int(env_cfg["max_new_tokens_per_prompt"]),
            resume=True,
            checkpoint_interval_prompts=int(env_cfg["checkpoint_interval_prompts"]),
        )
        neutral_manifest = build_environment_from_prompts(
            model=teacher,
            prompt_bank=generation_prompts,
            cfg=neutral_cfg,
            out_dir=neutral_dir,
            device=device,
        )
    else:
        neutral_manifest = json.loads(neutral_manifest_path.read_text(encoding="utf-8"))

    # Selected environment
    selected_dir = ensure_dir(env_root / "selected")
    selected_manifest_path = selected_dir / "selected_environment_manifest.json"
    selected_cfg_dict = load_selected_cfg(root, cfg["configs"]["selected_decoding_config_path"])
    selected_cfg = SelectedDecodingConfig(**selected_cfg_dict)

    if args.force or not selected_manifest_path.exists():
        sel_env_cfg = EnvironmentGenerationConfig(
            mode="selected",
            total_tokens=total_tokens,
            prompt_len=prompt_len,
            seed=int(cfg["seeds"]["environments"]) + 2,
            selected_cfg=selected_cfg,
            max_new_tokens_per_prompt=int(env_cfg["max_new_tokens_per_prompt"]),
            resume=True,
            checkpoint_interval_prompts=int(env_cfg["checkpoint_interval_prompts"]),
        )
        selected_manifest = build_environment_from_prompts(
            model=teacher,
            prompt_bank=generation_prompts,
            cfg=sel_env_cfg,
            out_dir=selected_dir,
            device=device,
        )
    else:
        selected_manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))

    combined = {
        "created_at": timestamp(),
        "mode": args.mode,
        "total_tokens": total_tokens,
        "generation_prompt_bank": str(generation_prompt_path),
        "evaluation_prompt_bank": str(evaluation_prompt_path),
        "original": original_manifest,
        "neutral": neutral_manifest,
        "selected": selected_manifest,
        "teacher_checkpoint": teacher_ckpt,
    }
    combined_path = env_root / f"environment_generation_manifest_{args.mode}.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")

    db_path = resolve_path(root, paths_cfg["db_path"])
    init_experiment_registry(db_path)
    run_id = register_run(
        db_path=db_path,
        run_name=f"transformer_generate_environments_{args.mode}",
        script_or_notebook="run_transformer_generate_environments.py",
        config_path=args.config,
        metadata={
            "timestamp_start": combined["created_at"],
            "timestamp_end": timestamp(),
            "corpus": "conan_doyle",
            "model_family": "tiny_gpt",
            "selection_rule": "neutral_vs_selected",
            "seed": int(cfg["seeds"]["environments"]),
            "output_dir": str(env_root.relative_to(root)),
            "metrics_summary": json.dumps({"total_tokens": total_tokens}),
            "status": "completed",
        },
    )
    log_artifact(db_path, run_id, "environments_manifest", str(combined_path.relative_to(root)))
    export_registry_csv(db_path, resolve_path(root, "GitHub/data/outputs/csv/experiment_registry.csv"))

    print(f"Environment generation complete: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
