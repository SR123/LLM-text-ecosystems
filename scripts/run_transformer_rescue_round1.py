#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.corpus import load_conan_doyle_split  # noqa: E402
from drift_selection.environments import (  # noqa: E402
    EnvironmentGenerationConfig,
    build_environment_from_prompts,
    generate_original_environment_subset,
    load_environment_tokens,
    save_environment_tokens,
)
from drift_selection.metrics import distinct_n  # noqa: E402
from drift_selection.selected_decoding import (  # noqa: E402
    SelectedDecodingConfig,
    repetition_penalty_value,
    selected_next_token,
)
from drift_selection.tokenization import (  # noqa: E402
    decode_ids,
    encode_split_dict,
    encode_text,
    load_sentencepiece_tokenizer,
    save_token_ids,
    train_sentencepiece_tokenizer,
)
from drift_selection.training import (  # noqa: E402
    TrainingConfig,
    get_best_device,
    load_trained_model,
    sample_lm_batch,
    train_language_model,
)
from drift_selection.transformer import (  # noqa: E402
    TransformerConfig,
    build_tiny_gpt,
    save_model_config,
)
from drift_selection.transformer_pipeline import (  # noqa: E402
    load_main_config,
    load_selected_cfg,
    resolve_path,
)
from drift_selection.utils import ensure_dir, timestamp  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_sentencepiece_training_text(text: str, max_line_chars: int = 2000) -> str:
    import re

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    rough_sentences = re.split(r"(?<=[.!?;:])\s+", normalized)
    lines: list[str] = []
    for sent in rough_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_line_chars:
            lines.append(sent)
            continue

        words = sent.split()
        chunk: list[str] = []
        chunk_len = 0
        for w in words:
            w_len = len(w) + (1 if chunk else 0)
            if chunk and chunk_len + w_len > max_line_chars:
                lines.append(" ".join(chunk))
                chunk = [w]
                chunk_len = len(w)
            else:
                chunk.append(w)
                chunk_len += w_len
        if chunk:
            lines.append(" ".join(chunk))

    return "\n".join(lines) + "\n"


def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _sequence_match_ratio(a: str, b: str) -> float:
    # lightweight approximation without importing difflib for large strings
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = 0
    lim = min(len(a), len(b))
    for i in range(lim):
        if a[i] == b[i]:
            common += 1
    return common / max(len(a), len(b))


def _model_next_logits(model, prefix_ids: list[int], device: torch.device) -> torch.Tensor:
    ctx = int(getattr(getattr(model, "config", object()), "context_len", 128))
    max_seq = int(getattr(getattr(model, "config", object()), "max_seq_len", ctx))
    win = max(1, min(ctx, max_seq))
    x = torch.tensor([prefix_ids[-win:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[:, -1, :]
    return logits[0]


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    out = logits.clone()
    if top_k is not None and top_k > 0:
        vals, _ = torch.topk(out, min(int(top_k), out.numel()))
        out = torch.where(out < vals[-1], torch.full_like(out, float("-inf")), out)
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > top_p
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        restored = torch.full_like(out, float("-inf"))
        restored.scatter_(0, sorted_idx, sorted_logits)
        out = restored
    return out


def _has_repeated_span(seq: list[int], span_len: int = 8) -> bool:
    if len(seq) < span_len:
        return False
    seen = set()
    for i in range(len(seq) - span_len + 1):
        g = tuple(seq[i : i + span_len])
        if g in seen:
            return True
        seen.add(g)
    return False


def _unique_fraction(seq: list[int]) -> float:
    if not seq:
        return 0.0
    return len(set(seq)) / float(len(seq))


def _token_entropy(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    vals, counts = np.unique(np.asarray(token_ids, dtype=np.int64), return_counts=True)
    _ = vals
    probs = counts.astype(float) / float(np.sum(counts))
    return float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())


def _generate_autoregressive(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    seed: int | None = None,
) -> list[int]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = _model_next_logits(model, out, device)
        logits = logits / max(temperature, 1e-6)
        logits = _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
        if greedy:
            next_id = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
                next_id = int(torch.argmax(_model_next_logits(model, out, device)).item())
            else:
                next_id = int(torch.multinomial(probs, num_samples=1).item())
        out.append(next_id)
    return out


def _archive_failed_run(root: Path, cfg: dict, force: bool = False) -> dict[str, Any]:
    rescue_cfg = cfg.get("rescue", {})
    src_dir = resolve_path(root, rescue_cfg.get("failed_run_source_dir", "GitHub/data/outputs/transformer_conan_doyle"))
    archive_dir = resolve_path(root, rescue_cfg.get("failed_archive_dir", "GitHub/data/outputs/transformer_conan_doyle/failed_run_01"))
    diagnostics_dir = resolve_path(root, rescue_cfg.get("failed_diagnostics_dir", "GitHub/data/outputs/diagnostics"))

    if not src_dir.exists():
        return {
            "status": "skipped",
            "reason": f"source_missing:{src_dir}",
            "archive_dir": str(archive_dir),
        }

    if archive_dir.exists() and not force:
        return {
            "status": "exists",
            "archive_dir": str(archive_dir),
        }

    if archive_dir.exists() and force:
        shutil.rmtree(archive_dir)

    archive_dir.parent.mkdir(parents=True, exist_ok=True)

    src_resolved = src_dir.resolve()
    archive_resolved = archive_dir.resolve()
    archive_inside_source = archive_resolved.is_relative_to(src_resolved)

    if archive_inside_source:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for child in src_dir.iterdir():
            if child.resolve() == archive_resolved:
                continue
            if child.name.startswith("failed_run_"):
                continue
            dst = archive_dir / child.name
            if child.is_dir():
                shutil.copytree(child, dst)
            else:
                shutil.copy2(child, dst)
    else:
        shutil.copytree(src_dir, archive_dir)

    copied_diag = None
    if diagnostics_dir.exists():
        copied_diag = archive_dir / "diagnostics_bundle"
        if copied_diag.exists():
            shutil.rmtree(copied_diag)
        shutil.copytree(diagnostics_dir, copied_diag)

    manifest = {
        "archived_at": _utc_now(),
        "status": "failed_generator_selection",
        "source_dir": str(src_dir),
        "archive_dir": str(archive_dir),
        "diagnostics_snapshot": str(copied_diag) if copied_diag else None,
    }
    (archive_dir / "failed_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _tokenizer_roundtrip_stats(tokenizer, test_text: str, snippets: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    unk_id = int(tokenizer.unk_id())
    edits: list[float] = []
    mismatch: list[float] = []
    unk_counts: list[int] = []
    tokens_per_word: list[float] = []

    for _ in range(snippets):
        if len(test_text) < 500:
            snip = test_text
        else:
            start = rng.randint(0, len(test_text) - 401)
            snip = test_text[start : start + 400]
        ids = [int(x) for x in tokenizer.sp.encode(snip, out_type=int)]
        dec = decode_ids(tokenizer, ids)
        ed = _edit_distance(snip, dec)
        ratio = _sequence_match_ratio(snip, dec)
        edits.append(float(ed))
        mismatch.append(float(1.0 - ratio))
        unk_counts.append(int(sum(1 for tid in ids if tid == unk_id)))
        tokens_per_word.append(float(len(ids) / max(1, len(snip.split()))))

    return {
        "roundtrip_median_edit_distance": float(np.median(edits)) if edits else float("nan"),
        "roundtrip_median_mismatch": float(np.median(mismatch)) if mismatch else float("nan"),
        "roundtrip_mean_unk_count": float(np.mean(unk_counts)) if unk_counts else float("nan"),
        "roundtrip_mean_tokens_per_word": float(np.mean(tokens_per_word)) if tokens_per_word else float("nan"),
    }


def _quick_generator_val_loss(
    tokenizer,
    split_texts: dict[str, str],
    cfg: dict,
    out_dir: Path,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    tokenized = encode_split_dict(tokenizer, split_texts)
    model_cfg = TransformerConfig(vocab_size=int(tokenizer.vocab_size), **cfg["model"])
    model = build_tiny_gpt(model_cfg)

    qcfg = cfg.get("rescue", {}).get("tokenizer_eval_training", {})
    train_cfg = TrainingConfig(
        batch_size=int(qcfg.get("batch_size", 16)),
        context_len=int(qcfg.get("context_len", cfg["training"]["context_len"])),
        learning_rate=float(qcfg.get("learning_rate", 4e-4)),
        weight_decay=float(qcfg.get("weight_decay", 0.0)),
        max_steps=int(qcfg.get("max_steps", 140)),
        eval_interval=int(qcfg.get("eval_interval", 35)),
        eval_batches=int(qcfg.get("eval_batches", 8)),
        warmup_steps=int(qcfg.get("warmup_steps", 20)),
        gradient_accumulation_steps=1,
        clip_grad_norm=1.0,
        seed=int(seed),
        device_preference="mps_or_cpu",
        early_stopping_patience=0,
        save_interval=70,
    )

    hist = train_language_model(
        model=model,
        train_ids=tokenized["train"],
        val_ids=tokenized["val"],
        train_cfg=train_cfg,
        out_dir=out_dir,
        resume=False,
        progress_bar=False,
    )
    val_hist = hist.get("val_loss_history", [])
    best_val = float("inf")
    if val_hist:
        best_val = float(min(float(x["val_loss"]) for x in val_hist))
    return best_val, {
        "history": hist,
        "token_counts": {k: len(v) for k, v in tokenized.items()},
    }


def _train_generator(root: Path, cfg: dict, split_ids: dict[str, list[int]], tokenizer, force: bool) -> tuple[dict, TransformerConfig, Path]:
    out_root = ensure_dir(resolve_path(root, cfg["paths"]["root_outputs_dir"]))
    generator_dir = ensure_dir(out_root / "generator")
    summary_path = generator_dir / "training_summary.json"
    model_cfg = TransformerConfig(vocab_size=int(tokenizer.vocab_size), **cfg["model"])
    save_model_config(model_cfg, generator_dir / "model_config.json")

    if summary_path.exists() and not force:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        best_ckpt = payload.get("history", {}).get("best_checkpoint_path")
        return payload, model_cfg, Path(best_ckpt) if best_ckpt else Path(payload["history"]["final_checkpoint_path"])

    model = build_tiny_gpt(model_cfg)
    train_cfg = TrainingConfig(**cfg["training"])
    hist = train_language_model(
        model=model,
        train_ids=split_ids["train"],
        val_ids=split_ids["val"],
        train_cfg=train_cfg,
        out_dir=generator_dir,
        resume=not force,
        progress_bar=True,
    )
    payload = {
        "created_at": timestamp(),
        "model_config": asdict(model_cfg),
        "training_config": asdict(train_cfg),
        "history": hist,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    best_ckpt = hist.get("best_checkpoint_path") or hist.get("final_checkpoint_path")
    return payload, model_cfg, Path(str(best_ckpt))


def _tiny_overfit_gate(model_cfg: TransformerConfig, split_ids: dict[str, list[int]], out_dir: Path, seed: int) -> dict[str, Any]:
    tiny_train = split_ids["train"][:30000]
    tiny_val = split_ids["train"][:6000]
    cfg = TrainingConfig(
        batch_size=32,
        context_len=int(model_cfg.context_len),
        learning_rate=5e-4,
        weight_decay=0.0,
        max_steps=300,
        eval_interval=50,
        eval_batches=10,
        warmup_steps=20,
        gradient_accumulation_steps=1,
        clip_grad_norm=1.0,
        seed=int(seed),
        device_preference="mps_or_cpu",
        early_stopping_patience=0,
        save_interval=100,
    )
    model = build_tiny_gpt(model_cfg)
    hist = train_language_model(
        model=model,
        train_ids=tiny_train,
        val_ids=tiny_val,
        train_cfg=cfg,
        out_dir=out_dir,
        resume=False,
        progress_bar=False,
    )

    device = get_best_device(prefer_mps=True)
    best_ckpt = Path(hist["best_checkpoint_path"])
    fit_model = load_trained_model(best_ckpt, model_cfg, device)
    acc_vals: list[float] = []
    for _ in range(30):
        xb, yb = sample_lm_batch(tiny_train, batch_size=16, context_len=model_cfg.context_len, device=device)
        with torch.no_grad():
            logits = fit_model(xb)
            pred = torch.argmax(logits, dim=-1)
            acc_vals.append(float((pred == yb).float().mean().item()))

    return {
        "tiny_train_tokens": len(tiny_train),
        "tiny_val_tokens": len(tiny_val),
        "tiny_next_token_accuracy": float(np.mean(acc_vals)) if acc_vals else 0.0,
        "history": hist,
    }


def _build_prompt_bank(ids: list[int], prompt_len: int, n_prompts: int, seed: int) -> list[list[int]]:
    if len(ids) <= prompt_len:
        return [list(ids[:prompt_len])] * n_prompts
    rng = random.Random(seed)
    prompts = []
    max_start = len(ids) - prompt_len
    for _ in range(n_prompts):
        s = rng.randint(0, max_start)
        prompts.append([int(x) for x in ids[s : s + prompt_len]])
    return prompts


def _decode_mode_outputs(model, tokenizer, prompts: list[list[int]], mode: str, selected_cfg: SelectedDecodingConfig | None, device: torch.device, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        if mode == "greedy":
            full = _generate_autoregressive(model, p, 80, device, greedy=True, seed=seed + i)
        elif mode == "top_p":
            full = _generate_autoregressive(model, p, 80, device, top_p=0.95, temperature=1.0, seed=seed + i)
        elif mode == "top_k":
            full = _generate_autoregressive(model, p, 80, device, top_k=40, temperature=1.0, seed=seed + i)
        elif mode == "selected" and selected_cfg is not None:
            out = list(p)
            for _ in range(80):
                step = selected_next_token(model, out, selected_cfg, device)
                out.append(int(step["chosen_next_token"]))
            full = out
        else:
            raise ValueError(mode)

        cont = full[len(p):]
        rows.append(
            {
                "mode": mode,
                "prompt_index": i,
                "repetition_4": float(1.0 - distinct_n([cont], n=4)),
                "distinct_2": float(distinct_n([cont], n=2)),
                "repeated_span_rate": float(_has_repeated_span(cont, span_len=8)),
                "unique_token_fraction": float(_unique_fraction(cont)),
                "output_text": decode_ids(tokenizer, cont),
            }
        )
    return rows


def _environment_health(tokenizer, env_name: str, token_ids: list[int]) -> dict[str, Any]:
    entropy = _token_entropy(token_ids)
    d1 = len(set(token_ids)) / max(1, len(token_ids))
    d2 = float(distinct_n([token_ids], n=2))
    rep4 = float(1.0 - distinct_n([token_ids], n=4))

    sample_lines: list[str] = []
    if len(token_ids) > 160:
        rng = random.Random(2026 + len(token_ids))
        for _ in range(20):
            s = rng.randint(0, len(token_ids) - 121)
            sample_lines.append(decode_ids(tokenizer, token_ids[s : s + 120]))

    return {
        "environment": env_name,
        "total_token_count": int(len(token_ids)),
        "distinct_1": float(d1),
        "distinct_2": float(d2),
        "repetition_4": float(rep4),
        "token_entropy_nats": float(entropy),
        "sample_lines": sample_lines,
    }


def _write_placeholder_csv(path: Path, header: list[str], reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([reason] + [""] * (len(header) - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescue pipeline for Conan Doyle generator/agent experiment")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_rescue_round1.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--run-agents", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_main_config(root, args.config)
    _set_seed(int(cfg.get("seeds", {}).get("master", 12345)))

    out_root = ensure_dir(resolve_path(root, cfg["paths"]["root_outputs_dir"]))
    diag_root = ensure_dir(out_root / "diagnostics")
    text_root = ensure_dir(diag_root / "text_samples")

    run_manifest: dict[str, Any] = {
        "created_at": _utc_now(),
        "mode": args.mode,
        "config": args.config,
        "steps": {},
    }

    if not args.skip_archive:
        archive_info = _archive_failed_run(root, cfg, force=args.force)
        run_manifest["steps"]["archive_failed_run"] = archive_info

    split_dir = resolve_path(root, cfg["corpus"]["split_dir"])
    split_texts = load_conan_doyle_split(split_dir)

    # Tokenizer comparison
    tokenizer_candidates = cfg.get("rescue", {}).get("tokenizer_candidates", [])
    if not tokenizer_candidates:
        tokenizer_candidates = [
            {"name": "bpe_1000", "model_type": "bpe", "vocab_size": 1000},
            {"name": "bpe_2000", "model_type": "bpe", "vocab_size": 2000},
            {"name": "unigram_1000", "model_type": "unigram", "vocab_size": 1000},
        ]

    tok_compare_rows: list[dict[str, Any]] = []
    tok_candidates_dir = ensure_dir(diag_root / "tokenizer_candidates")
    sp_text = _prepare_sentencepiece_training_text(split_texts["train"])
    train_text_path = tok_candidates_dir / "train_text_for_sp.txt"
    train_text_path.write_text(sp_text, encoding="utf-8")

    best_name = None
    best_score = float("inf")
    best_model_path: Path | None = None
    best_tokenizer = None

    for i, tcfg in enumerate(tokenizer_candidates):
        name = str(tcfg["name"])
        tdir = ensure_dir(tok_candidates_dir / name)
        model_path = train_sentencepiece_tokenizer(
            input_text_path=train_text_path,
            output_dir=tdir,
            vocab_size=int(tcfg["vocab_size"]),
            model_type=str(tcfg["model_type"]),
            character_coverage=float(tcfg.get("character_coverage", 1.0)),
        )
        tok = load_sentencepiece_tokenizer(model_path)
        rt = _tokenizer_roundtrip_stats(tok, split_texts["test"], snippets=100, seed=7000 + i)

        quick_dir = ensure_dir(tdir / "quick_generator")
        quick_val, quick_meta = _quick_generator_val_loss(tok, split_texts, cfg, quick_dir, seed=8000 + i)

        score = float(quick_val) + 4.0 * float(rt["roundtrip_median_mismatch"])
        row = {
            "tokenizer_name": name,
            "model_type": str(tcfg["model_type"]),
            "vocab_size": int(tcfg["vocab_size"]),
            **rt,
            "quick_generator_best_val_loss": float(quick_val),
            "selection_score": float(score),
            "model_path": str(model_path),
            "quick_meta_path": str((quick_dir / "checkpoints").resolve()),
        }
        tok_compare_rows.append(row)

        if score < best_score:
            best_score = score
            best_name = name
            best_model_path = model_path
            best_tokenizer = tok

    tok_compare_df = pd.DataFrame(tok_compare_rows).sort_values("selection_score")
    tok_compare_df.to_csv(diag_root / "tokenizer_comparison.csv", index=False)

    if best_model_path is None or best_tokenizer is None:
        raise RuntimeError("Failed to train tokenizer candidates")

    tokenizer_out_dir = ensure_dir(out_root / "tokenizer")
    shutil.copy2(best_model_path, tokenizer_out_dir / "conan_doyle_sp.model")
    vocab_src = best_model_path.with_suffix(".vocab")
    if vocab_src.exists():
        shutil.copy2(vocab_src, tokenizer_out_dir / "conan_doyle_sp.vocab")

    selected_tokenizer = load_sentencepiece_tokenizer(tokenizer_out_dir / "conan_doyle_sp.model")
    split_ids = encode_split_dict(selected_tokenizer, split_texts)

    processed_dir = ensure_dir(resolve_path(root, cfg["paths"]["processed_dir"]))
    for split, ids in split_ids.items():
        save_token_ids(processed_dir / f"conan_doyle_{split}_ids.npy", ids)

    run_manifest["steps"]["tokenizer"] = {
        "selected_tokenizer": best_name,
        "selected_model_path": str(best_model_path),
        "tokenizer_comparison": str((diag_root / "tokenizer_comparison.csv").relative_to(root)),
    }

    # Train generator
    gen_payload, model_cfg, best_ckpt = _train_generator(root, cfg, split_ids, selected_tokenizer, force=args.force)
    run_manifest["steps"]["generator_train"] = {
        "summary_path": str((out_root / "generator" / "training_summary.json").relative_to(root)),
        "best_checkpoint": str(best_ckpt),
    }

    # Tiny overfit gate
    overfit_dir = ensure_dir(diag_root / "generator_tiny_overfit")
    overfit = _tiny_overfit_gate(model_cfg, split_ids, overfit_dir, seed=int(cfg["seeds"]["master"]) + 33)
    (diag_root / "generator_overfit_metrics.csv").write_text(
        pd.DataFrame(
            [
                {
                    "tiny_train_tokens": overfit["tiny_train_tokens"],
                    "tiny_val_tokens": overfit["tiny_val_tokens"],
                    "tiny_next_token_accuracy": overfit["tiny_next_token_accuracy"],
                    "best_val_loss": min(float(x["val_loss"]) for x in overfit["history"].get("val_loss_history", []) or [{"val_loss": float("nan")}]),
                }
            ]
        ).to_csv(index=False),
        encoding="utf-8",
    )

    overfit_threshold = float(cfg.get("rescue", {}).get("gates", {}).get("tiny_overfit_min_accuracy", 0.55))
    overfit_pass = float(overfit["tiny_next_token_accuracy"]) >= overfit_threshold
    report_lines = [
        "# Generator Overfit Report",
        "",
        f"- tiny_next_token_accuracy: {overfit['tiny_next_token_accuracy']:.4f}",
        f"- required_min_accuracy: {overfit_threshold:.4f}",
        f"- gate_pass: {overfit_pass}",
    ]
    (diag_root / "generator_overfit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    device = get_best_device(prefer_mps=True)
    generator = load_trained_model(Path(best_ckpt), model_cfg, device)

    # Prompt quality gate (real held-out prompts)
    prompt_len = int(cfg["environments"]["prompt_len"])
    prompt_count = int(cfg.get("rescue", {}).get("prompt_quality_count", 32))
    prompts = _build_prompt_bank(split_ids["test"], prompt_len=prompt_len, n_prompts=prompt_count, seed=int(cfg["seeds"]["prompts"]))

    mode_rows = []
    mode_rows.extend(_decode_mode_outputs(generator, selected_tokenizer, prompts, "greedy", None, device, seed=9200))
    mode_rows.extend(_decode_mode_outputs(generator, selected_tokenizer, prompts, "top_p", None, device, seed=9300))
    mode_rows.extend(_decode_mode_outputs(generator, selected_tokenizer, prompts, "top_k", None, device, seed=9400))

    quality_df = pd.DataFrame(mode_rows)
    quality_df.to_csv(diag_root / "generator_prompt_quality_sheet.csv", index=False)

    samples_lines = []
    for i, p in enumerate(prompts[:20]):
        samples_lines.append("=" * 88)
        samples_lines.append(f"PROMPT_{i}")
        samples_lines.append(decode_ids(selected_tokenizer, p))
        for mode in ["greedy", "top_p", "top_k"]:
            txt = quality_df[(quality_df["mode"] == mode) & (quality_df["prompt_index"] == i)]["output_text"].tolist()
            if txt:
                samples_lines.append(f"[{mode}] {txt[0]}")
    (text_root / "generator_prompt_quality_samples.txt").write_text("\n".join(samples_lines) + "\n", encoding="utf-8")

    top_p_stats = quality_df[quality_df["mode"] == "top_p"]
    max_repeated_span = float(cfg.get("rescue", {}).get("gates", {}).get("top_p_max_repeated_span_rate", 0.45))
    min_unique = float(cfg.get("rescue", {}).get("gates", {}).get("top_p_min_unique_fraction", 0.20))
    prompt_quality_pass = (
        float(top_p_stats["repeated_span_rate"].mean()) <= max_repeated_span
        and float(top_p_stats["unique_token_fraction"].mean()) >= min_unique
    )

    if not overfit_pass or not prompt_quality_pass:
        run_manifest["status"] = "stopped_generator_quality_gate"
        run_manifest["steps"]["gates"] = {
            "tiny_overfit_pass": overfit_pass,
            "prompt_quality_pass": prompt_quality_pass,
        }
        (diag_root / "round2_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

        _write_placeholder_csv(
            diag_root / "selected_vs_neutral_decoding_summary.csv",
            ["status", "mode", "distinct_2", "repetition_4", "repeated_span_rate", "unique_token_fraction"],
            "skipped_due_to_generator_quality_gate",
        )
        _write_placeholder_csv(
            diag_root / "environment_health_round2.csv",
            ["status", "environment", "token_entropy_nats", "distinct_2", "repetition_4"],
            "skipped_due_to_generator_quality_gate",
        )
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "skipped_due_to_generator_quality_gate",
        )
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_ngram_control_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "skipped_due_to_generator_quality_gate",
        )
        (diag_root / "selected_branch_audit_round2.jsonl").write_text("", encoding="utf-8")
        print(f"Stopped at generator quality gate. Details: {diag_root / 'round2_manifest.json'}")
        return 0

    # Selected decoding audit (sampled-reranked primary)
    sel_cfg_dict = load_selected_cfg(root, cfg["configs"]["selected_decoding_config_path"])
    selected_cfg = SelectedDecodingConfig(**sel_cfg_dict)

    sel_rows = _decode_mode_outputs(generator, selected_tokenizer, prompts, "selected", selected_cfg, device, seed=9500)
    sel_df = pd.DataFrame(sel_rows)
    agg = []
    for mode, frame in [("neutral_top_p", top_p_stats), ("selected", sel_df)]:
        agg.append(
            {
                "mode": mode,
                "distinct_2": float(frame["distinct_2"].mean()),
                "repetition_4": float(frame["repetition_4"].mean()),
                "repeated_span_rate": float(frame["repeated_span_rate"].mean()),
                "unique_token_fraction": float(frame["unique_token_fraction"].mean()),
            }
        )
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(diag_root / "selected_vs_neutral_decoding_summary.csv", index=False)

    with (diag_root / "selected_branch_audit_round2.jsonl").open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts[:20]):
            info = selected_next_token(generator, prompt, selected_cfg, device)
            chosen = int(info["chosen_next_token"])
            for rank, row in enumerate(info["candidate_table"][:32]):
                bid = [int(x) for x in row.get("best_branch_ids", [])]
                payload = {
                    "prompt_index": i,
                    "prompt_text": decode_ids(selected_tokenizer, prompt),
                    "candidate_rank": int(rank),
                    "first_token_id": int(row["first_token_id"]),
                    "first_token_piece": selected_tokenizer.sp.id_to_piece(int(row["first_token_id"])),
                    "first_logprob": float(row.get("first_logprob", float("nan"))),
                    "branch_score": float(row.get("branch_score", float("nan"))),
                    "logprob_sum": float(row.get("logprob_sum", float("nan"))),
                    "best_branch_ids": bid,
                    "best_branch_text": decode_ids(selected_tokenizer, bid),
                    "repetition_penalty_contribution": float(
                        repetition_penalty_value(bid, n=selected_cfg.repetition_ngram, coeff=selected_cfg.repetition_penalty)
                    ),
                    "selected_token": chosen,
                    "is_selected": int(row["first_token_id"]) == chosen,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    sel_loop_delta = float(cfg.get("rescue", {}).get("gates", {}).get("selected_max_repeated_span_delta", 0.10))
    sel_unique_delta = float(cfg.get("rescue", {}).get("gates", {}).get("selected_min_unique_fraction_delta", -0.12))
    neutral_row = agg_df[agg_df["mode"] == "neutral_top_p"].iloc[0]
    selected_row = agg_df[agg_df["mode"] == "selected"].iloc[0]
    selected_gate_pass = (
        float(selected_row["repeated_span_rate"]) <= float(neutral_row["repeated_span_rate"]) + sel_loop_delta
        and float(selected_row["unique_token_fraction"]) >= float(neutral_row["unique_token_fraction"]) + sel_unique_delta
    )

    if not selected_gate_pass:
        run_manifest["status"] = "stopped_selected_branch_gate"
        run_manifest["steps"]["gates"] = {
            "tiny_overfit_pass": overfit_pass,
            "prompt_quality_pass": prompt_quality_pass,
            "selected_branch_pass": selected_gate_pass,
        }
        (diag_root / "round2_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
        _write_placeholder_csv(
            diag_root / "environment_health_round2.csv",
            ["status", "environment", "token_entropy_nats", "distinct_2", "repetition_4"],
            "skipped_due_to_selected_branch_gate",
        )
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "skipped_due_to_selected_branch_gate",
        )
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_ngram_control_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "skipped_due_to_selected_branch_gate",
        )
        print(f"Stopped at selected branch gate. Details: {diag_root / 'round2_manifest.json'}")
        return 0

    # Build environments after gates pass
    env_cfg = cfg["environments"]
    total_tokens = int(env_cfg["pilot_total_tokens"] if args.mode == "pilot" else env_cfg["full_total_tokens"])
    generation_prompt_count = int(env_cfg["generation_prompt_count"])
    generation_prompts = _build_prompt_bank(
        split_ids["train"],
        prompt_len=int(env_cfg["prompt_len"]),
        n_prompts=generation_prompt_count,
        seed=int(cfg["seeds"]["prompts"]),
    )

    env_root = ensure_dir(out_root / "environments")
    original_dir = ensure_dir(env_root / "original")
    neutral_dir = ensure_dir(env_root / "neutral")
    selected_dir = ensure_dir(env_root / "selected")

    orig_ids = generate_original_environment_subset(split_ids["train"], total_tokens=total_tokens, seed=int(cfg["seeds"]["environments"]))
    save_environment_tokens(original_dir, "original_environment", orig_ids, selected_tokenizer)

    neutral_manifest = build_environment_from_prompts(
        model=generator,
        prompt_bank=[list(p) for p in generation_prompts],
        cfg=EnvironmentGenerationConfig(
            mode="neutral",
            total_tokens=total_tokens,
            prompt_len=int(env_cfg["prompt_len"]),
            temperature=float(env_cfg["neutral_temperature"]),
            top_k=env_cfg.get("neutral_top_k"),
            top_p=env_cfg.get("neutral_top_p"),
            seed=int(cfg["seeds"]["environments"]) + 1,
            max_new_tokens_per_prompt=int(env_cfg["max_new_tokens_per_prompt"]),
            resume=True,
            checkpoint_interval_prompts=int(env_cfg["checkpoint_interval_prompts"]),
        ),
        out_dir=neutral_dir,
        device=device,
    )
    selected_manifest = build_environment_from_prompts(
        model=generator,
        prompt_bank=[list(p) for p in generation_prompts],
        cfg=EnvironmentGenerationConfig(
            mode="selected",
            total_tokens=total_tokens,
            prompt_len=int(env_cfg["prompt_len"]),
            seed=int(cfg["seeds"]["environments"]) + 2,
            selected_cfg=selected_cfg,
            max_new_tokens_per_prompt=int(env_cfg["max_new_tokens_per_prompt"]),
            resume=True,
            checkpoint_interval_prompts=int(env_cfg["checkpoint_interval_prompts"]),
        ),
        out_dir=selected_dir,
        device=device,
    )

    neutral_ids = load_environment_tokens(Path(neutral_manifest["token_ids_path"]))
    selected_ids = load_environment_tokens(Path(selected_manifest["token_ids_path"]))

    save_environment_tokens(neutral_dir, "neutral_environment", neutral_ids, selected_tokenizer)
    save_environment_tokens(selected_dir, "selected_environment", selected_ids, selected_tokenizer)

    env_rows = [
        _environment_health(selected_tokenizer, "original", orig_ids),
        _environment_health(selected_tokenizer, "neutral", neutral_ids),
        _environment_health(selected_tokenizer, "selected", selected_ids),
    ]
    env_df = pd.DataFrame([{k: v for k, v in row.items() if k != "sample_lines"} for row in env_rows])
    env_df.to_csv(diag_root / "environment_health_round2.csv", index=False)

    for row in env_rows:
        (text_root / f"environment_samples_{row['environment']}.txt").write_text("\n\n".join(row["sample_lines"]) + "\n", encoding="utf-8")

    env_index = {row["environment"]: row for row in env_rows}
    max_entropy_drop = float(cfg.get("rescue", {}).get("gates", {}).get("selected_max_entropy_drop", 1.0))
    max_rep_delta = float(cfg.get("rescue", {}).get("gates", {}).get("selected_max_repetition_delta", 0.08))
    env_gate_pass = (
        float(env_index["selected"]["token_entropy_nats"]) >= float(env_index["neutral"]["token_entropy_nats"]) - max_entropy_drop
        and float(env_index["selected"]["repetition_4"]) <= float(env_index["neutral"]["repetition_4"]) + max_rep_delta
    )

    run_manifest["steps"]["gates"] = {
        "tiny_overfit_pass": overfit_pass,
        "prompt_quality_pass": prompt_quality_pass,
        "selected_branch_pass": selected_gate_pass,
        "environment_health_pass": env_gate_pass,
    }

    if not env_gate_pass:
        run_manifest["status"] = "stopped_environment_health_gate"
        (diag_root / "round2_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "skipped_due_to_environment_health_gate",
        )
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_ngram_control_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "skipped_due_to_environment_health_gate",
        )
        print(f"Stopped at environment health gate. Details: {diag_root / 'round2_manifest.json'}")
        return 0

    # Optional agent stage
    if not args.run_agents:
        run_manifest["status"] = "ready_for_agent_training"
        run_manifest["next_step"] = "Run with --run-agents after reviewing diagnostics"
        (diag_root / "round2_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "not_run_use_run_agents_flag",
        )
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_ngram_control_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "not_run_use_run_agents_flag",
        )
        print(f"Rescue round reached agent-ready state. Manifest: {diag_root / 'round2_manifest.json'}")
        return 0

    # Agent training + evaluation (reuse existing scripts with rescue config)
    py = sys.executable
    script_dir = root / "GitHub" / "scripts"
    subprocess.run([py, str(script_dir / "run_transformer_student_train.py"), "--root", str(root), "--config", args.config, "--mode", args.mode, *(["--force"] if args.force else [])], check=True)
    subprocess.run([py, str(script_dir / "run_transformer_evaluation.py"), "--root", str(root), "--config", args.config, "--mode", args.mode], check=True)

    eval_csv = out_root / "evaluation" / "csv" / f"teacher_student_metrics_{args.mode}.csv"
    if eval_csv.exists():
        shutil.copy2(eval_csv, diag_root / "agent_policy_metrics_round2.csv")
    else:
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "evaluation_csv_missing",
        )

    # If an n-gram control metric file exists from diagnostics, carry it as round2 control artifact.
    prior_ngram = resolve_path(root, "GitHub/data/outputs/diagnostics/teacher_student_policy_metrics_ngram_control.csv")
    if prior_ngram.exists():
        shutil.copy2(prior_ngram, diag_root / "agent_policy_metrics_ngram_control_round2.csv")
    else:
        _write_placeholder_csv(
            diag_root / "agent_policy_metrics_ngram_control_round2.csv",
            ["status", "agent", "mean_selected_probability", "top1_match_rate", "mean_branch_score"],
            "ngram_control_not_run_in_round2",
        )

    run_manifest["status"] = "completed_with_agents"
    (diag_root / "round2_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rescue round complete: {diag_root / 'round2_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
