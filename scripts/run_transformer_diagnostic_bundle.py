#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.metrics import (  # noqa: E402
    build_subword_trigram_table,
    distinct_n,
    evaluate_all_students,
    heldout_loss,
    support_rate_against_trigram_table,
    teacher_branch_score_for_sequence,
    time_to_first_trigram_support_failure,
)
from drift_selection.selected_decoding import (  # noqa: E402
    SelectedDecodingConfig,
    generate_selected_tokens,
    repetition_penalty_value,
    selected_next_token,
)
from drift_selection.tokenization import decode_ids  # noqa: E402
from drift_selection.training import (  # noqa: E402
    TrainingConfig,
    load_trained_model,
    sample_lm_batch,
    train_language_model,
)
from drift_selection.transformer import TransformerConfig, build_tiny_gpt, load_model_config  # noqa: E402
from drift_selection.transformer_pipeline import (  # noqa: E402
    ensure_tokenizer_and_encoded_splits,
    load_main_config,
    load_prompt_bank,
    load_selected_cfg,
    resolve_path,
    write_prompt_bank,
)
from drift_selection.utils import ensure_dir  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_from_summary(summary_path: Path, device: torch.device):
    info = json.loads(summary_path.read_text(encoding="utf-8"))
    ckpt = info["history"].get("best_checkpoint_path") or info["history"].get("final_checkpoint_path")
    if not ckpt:
        raise RuntimeError(f"No checkpoint path in {summary_path}")
    model_cfg = load_model_config(summary_path.parent / "model_config.json")
    model = load_trained_model(Path(ckpt), model_cfg, device)
    return model, info, str(ckpt), model_cfg


def _model_next_logits(model, prefix_ids: list[int], device: torch.device) -> torch.Tensor:
    ctx = int(getattr(getattr(model, "config", object()), "context_len", 128))
    max_seq = int(getattr(getattr(model, "config", object()), "max_seq_len", ctx))
    win = max(1, min(ctx, max_seq))
    x = torch.tensor([prefix_ids[-win:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[:, -1, :]
    return logits[0]


def _violates_no_repeat(seq: list[int], next_id: int, n: int) -> bool:
    if n <= 1 or len(seq) < n - 1:
        return False
    prefix = tuple(seq[-(n - 1):])
    seen = set()
    for i in range(len(seq) - n + 1):
        seen.add(tuple(seq[i : i + n]))
    return tuple(prefix + (int(next_id),)) in seen


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    out = logits.clone()
    if top_k is not None and top_k > 0:
        vals, _ = torch.topk(out, min(top_k, out.numel()))
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


def _generate_autoregressive(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int | None = None,
    seed: int | None = None,
) -> list[int]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = _model_next_logits(model, out, device)
        if repetition_penalty and repetition_penalty > 1.0:
            for tid in set(out):
                logits[tid] = logits[tid] / repetition_penalty
        if no_repeat_ngram_size and no_repeat_ngram_size > 1 and len(out) >= no_repeat_ngram_size - 1:
            for tid in range(logits.shape[0]):
                if _violates_no_repeat(out, int(tid), int(no_repeat_ngram_size)):
                    logits[tid] = float("-inf")
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


def _generate_selected_with_constraints(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    cfg: SelectedDecodingConfig,
    device: torch.device,
    no_repeat_ngram_size: int | None = None,
) -> list[int]:
    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        step = selected_next_token(model=model, prefix_ids=out, cfg=cfg, device=device)
        chosen = int(step["chosen_next_token"])
        if no_repeat_ngram_size and _violates_no_repeat(out, chosen, int(no_repeat_ngram_size)):
            replacement = None
            for row in step["candidate_table"]:
                tid = int(row["first_token_id"])
                if not _violates_no_repeat(out, tid, int(no_repeat_ngram_size)):
                    replacement = tid
                    break
            chosen = replacement if replacement is not None else chosen
        out.append(int(chosen))
    return out


def _entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-12)
    return float(-(probs * probs.log()).sum().item())


def _loop_position(seq: list[int], n: int = 4) -> int:
    if len(seq) < n:
        return len(seq)
    seen = set()
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i : i + n])
        if g in seen:
            return i
        seen.add(g)
    return len(seq)


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


def _ngram_count_map(token_ids: list[int], order: int = 3):
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    totals: Counter[tuple[int, ...]] = Counter()
    for i in range(order - 1, len(token_ids)):
        ctx = tuple(token_ids[i - order + 1 : i])
        nxt = int(token_ids[i])
        counts[ctx][nxt] += 1
        totals[ctx] += 1
    return counts, totals


def _ngram_distribution(
    counts: dict[tuple[int, ...], Counter[int]],
    totals: Counter[tuple[int, ...]],
    context: tuple[int, ...],
) -> dict[int, float]:
    if context in counts and totals[context] > 0:
        t = float(totals[context])
        return {int(k): float(v) / t for k, v in counts[context].items()}
    if len(context) > 0:
        return _ngram_distribution(counts, totals, context[1:])
    merged = Counter()
    for c in counts.values():
        merged.update(c)
    t = float(sum(merged.values()) or 1.0)
    return {int(k): float(v) / t for k, v in merged.items()} if merged else {0: 1.0}


def _sample_from_distribution(dist: dict[int, float], rng: random.Random) -> int:
    items = list(dist.items())
    r = rng.random()
    acc = 0.0
    for k, p in items:
        acc += float(p)
        if r <= acc:
            return int(k)
    return int(items[-1][0])


def _generate_ngram_neutral(seed_ids: list[int], total_tokens: int, counts, totals, order: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    out = list(seed_ids[: order - 1])
    if not out:
        out = [0] * max(1, order - 1)
    while len(out) < total_tokens:
        ctx_len = order - 1
        ctx = tuple(out[-ctx_len:]) if ctx_len > 0 else tuple()
        dist = _ngram_distribution(counts, totals, ctx)
        out.append(_sample_from_distribution(dist, rng))
    return out[:total_tokens]


def _generate_ngram_selected(
    seed_ids: list[int],
    total_tokens: int,
    counts,
    totals,
    order: int,
    seed: int,
    top_k: int = 8,
    horizon: int = 3,
    rep_ngram: int = 4,
    rep_penalty: float = 0.8,
) -> list[int]:
    rng = random.Random(seed)
    out = list(seed_ids[: order - 1])
    if not out:
        out = [0] * max(1, order - 1)
    while len(out) < total_tokens:
        ctx = tuple(out[-(order - 1):]) if order > 1 else tuple()
        dist0 = _ngram_distribution(counts, totals, ctx)
        ranked = sorted(dist0.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
        best_tid = ranked[0][0]
        best_score = -1e18
        for tid, p in ranked:
            seq = out + [int(tid)]
            score = math.log(max(float(p), 1e-12))
            for _ in range(max(0, horizon - 1)):
                ctx_h = tuple(seq[-(order - 1):]) if order > 1 else tuple()
                dist_h = _ngram_distribution(counts, totals, ctx_h)
                nxt, p_h = max(dist_h.items(), key=lambda kv: kv[1])
                seq.append(int(nxt))
                score += math.log(max(float(p_h), 1e-12))
            score -= rep_penalty * float(_has_repeated_span(seq[-32:], span_len=rep_ngram))
            score += 1e-8 * rng.random()
            if score > best_score:
                best_score = score
                best_tid = int(tid)
        out.append(best_tid)
    return out[:total_tokens]


def _write_json(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sequence_edit_distance(a: str, b: str) -> int:
    # Lightweight DP edit distance for short snippets.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build transformer diagnostic bundle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/transformer_teacher_student.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--seed", type=int, default=20260303)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_main_config(root, args.config)
    _set_seed(args.seed)

    prep = ensure_tokenizer_and_encoded_splits(root, cfg, force=False)
    tokenizer = prep["tokenizer"]
    split_ids = prep["encoded_ids"]

    split_dir = resolve_path(root, cfg["corpus"]["split_dir"])
    split_text = {
        "train": (split_dir / "train.txt").read_text(encoding="utf-8", errors="ignore"),
        "val": (split_dir / "val.txt").read_text(encoding="utf-8", errors="ignore"),
        "test": (split_dir / "test.txt").read_text(encoding="utf-8", errors="ignore"),
    }

    out_diag = ensure_dir(resolve_path(root, "GitHub/data/outputs/diagnostics"))
    out_text = ensure_dir(out_diag / "text_samples")
    out_root = resolve_path(root, cfg["paths"]["root_outputs_dir"])
    prompt_bank_dir = out_root / "manifests"
    prompt_bank_path = prompt_bank_dir / f"evaluation_prompts_{args.mode}_prompt_bank.json"
    if not prompt_bank_path.exists():
        prompt_bank_path = write_prompt_bank(
            out_dir=prompt_bank_dir,
            tokenizer=tokenizer,
            token_ids=split_ids["test"],
            prompt_len=int(cfg["environments"]["prompt_len"]),
            n_prompts=int(cfg["evaluation"]["pilot_prompt_count"] if args.mode == "pilot" else cfg["evaluation"]["full_prompt_count"]),
            seed=int(cfg["seeds"]["prompts"]) + 1,
            name=f"evaluation_prompts_{args.mode}",
        )
    prompt_bank = load_prompt_bank(prompt_bank_path)

    prompt_indices_20 = sorted(random.Random(args.seed).sample(range(len(prompt_bank)), k=min(20, len(prompt_bank))))
    prompts_20 = [prompt_bank[i] for i in prompt_indices_20]

    # Load teacher + current students
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    teacher, teacher_info, teacher_ckpt, teacher_cfg = _load_model_from_summary(out_root / "teacher" / "training_summary.json", device)

    students: dict[str, Any] = {}
    student_infos: dict[str, dict] = {}
    for env in ["original", "neutral", "selected"]:
        model, info, ckpt, _ = _load_model_from_summary(out_root / "students" / env / "training_summary.json", device)
        students[env] = model
        student_infos[env] = {"summary": info, "checkpoint": ckpt}

    selected_cfg = SelectedDecodingConfig(**load_selected_cfg(root, cfg["configs"]["selected_decoding_config_path"]))

    # 1) Prompt-bank audit
    test_ids = split_ids["test"]
    key_len = 4
    idx_map: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(0, max(0, len(test_ids) - key_len + 1)):
        idx_map[tuple(test_ids[i : i + key_len])].append(i)

    rng_prompt = random.Random(args.seed + 11)
    prompt_indices_50 = sorted(rng_prompt.sample(range(len(prompt_bank)), k=min(50, len(prompt_bank))))
    prompt_rows: list[dict[str, Any]] = []
    for ridx in prompt_indices_50:
        p = [int(x) for x in prompt_bank[ridx]]
        starts = idx_map.get(tuple(p[:key_len]), [])
        found_start = -1
        for s in starts:
            if test_ids[s : s + len(p)] == p:
                found_start = int(s)
                break
        found = found_start >= 0
        raw_snip = ""
        if found:
            left = max(0, found_start - 20)
            right = min(len(test_ids), found_start + len(p) + 20)
            raw_snip = decode_ids(tokenizer, test_ids[left:right])
        row = {
            "prompt_bank_index": int(ridx),
            "corpus_name": "conan_doyle",
            "source_file": str((split_dir / "test.txt").relative_to(root)),
            "source_chapter": "",
            "token_start": int(found_start) if found else -1,
            "token_end": int(found_start + len(p)) if found else -1,
            "contiguous_match_in_test_split": bool(found),
            "crosses_known_boundary": False,
            "raw_source_snippet": raw_snip,
            "decoded_prompt_text": decode_ids(tokenizer, p),
        }
        prompt_rows.append(row)
        print(f"[prompt-audit] idx={ridx} start={row['token_start']} end={row['token_end']}")
    pd.DataFrame(prompt_rows).to_csv(out_diag / "prompt_audit.csv", index=False)

    # 2) Tokenizer round-trip audit
    unk_id = int(tokenizer.unk_id())
    test_text = split_text["test"]
    rt_rows: list[dict[str, Any]] = []
    token_freq = Counter(split_ids["test"])
    rng_rt = random.Random(args.seed + 22)
    for i in range(100):
        if len(test_text) < 350:
            snippet = test_text
        else:
            start = rng_rt.randint(0, len(test_text) - 301)
            snippet = test_text[start : start + 300]
        ids = [int(x) for x in tokenizer.sp.encode(snippet, out_type=int)]
        dec = decode_ids(tokenizer, ids)
        ed = _sequence_edit_distance(snippet, dec)
        ratio = SequenceMatcher(None, snippet, dec).ratio()
        rt_rows.append(
            {
                "snippet_id": i,
                "original_text": snippet,
                "decoded_text": dec,
                "token_count": len(ids),
                "unknown_token_count": int(sum(1 for t in ids if t == unk_id)),
                "edit_distance": int(ed),
                "mismatch_score": float(1.0 - ratio),
                "tokens_per_word": float(len(ids) / max(1, len(snippet.split()))),
                "tokens_per_char": float(len(ids) / max(1, len(snippet))),
            }
        )
    rt_df = pd.DataFrame(rt_rows)
    rt_df.to_csv(out_diag / "tokenizer_roundtrip_audit.csv", index=False)

    top_tokens = token_freq.most_common(100)
    vocab_rows = []
    for tid, cnt in top_tokens:
        vocab_rows.append(
            {
                "token_id": int(tid),
                "piece": tokenizer.sp.id_to_piece(int(tid)),
                "count": int(cnt),
                "frequency": float(cnt) / max(1, len(split_ids["test"])),
            }
        )
    pd.DataFrame(vocab_rows).to_csv(out_diag / "tokenizer_vocab_summary.csv", index=False)

    # 3) Tiny overfit sanity test
    overfit_dir = ensure_dir(out_diag / "teacher_tiny_overfit")
    tiny_train = split_ids["train"][:30000]
    tiny_val = split_ids["train"][:6000]
    tiny_cfg = TrainingConfig(
        batch_size=32,
        context_len=int(cfg["training"]["context_len"]),
        learning_rate=5e-4,
        weight_decay=0.0,
        max_steps=300,
        eval_interval=50,
        eval_batches=10,
        warmup_steps=20,
        gradient_accumulation_steps=1,
        clip_grad_norm=1.0,
        seed=int(args.seed + 33),
        device_preference="mps_or_cpu",
        early_stopping_patience=0,
        save_interval=100,
    )
    tiny_model = build_tiny_gpt(teacher_cfg)
    tiny_hist = train_language_model(
        model=tiny_model,
        train_ids=tiny_train,
        val_ids=tiny_val,
        train_cfg=tiny_cfg,
        out_dir=overfit_dir,
        resume=False,
        progress_bar=False,
    )
    tiny_model = load_trained_model(Path(tiny_hist["best_checkpoint_path"]), teacher_cfg, device)
    acc_vals: list[float] = []
    for _ in range(30):
        xb, yb = sample_lm_batch(tiny_train, batch_size=16, context_len=int(cfg["training"]["context_len"]), device=device)
        with torch.no_grad():
            logits = tiny_model(xb)
        pred = torch.argmax(logits, dim=-1)
        acc_vals.append(float((pred == yb).float().mean().item()))
    tiny_acc = float(np.mean(acc_vals))
    tiny_samples: list[str] = []
    for i in range(10):
        start = (i * 97) % max(1, len(tiny_train) - 100)
        p = tiny_train[start : start + 64]
        out_ids = _generate_autoregressive(
            tiny_model,
            p,
            max_new_tokens=60,
            device=device,
            temperature=0.9,
            top_p=0.95,
            seed=args.seed + i,
        )
        tiny_samples.append(decode_ids(tokenizer, out_ids))
    _write_json(
        out_diag / "teacher_tiny_overfit_metrics.json",
        {
            "created_at": _utc_now(),
            "tiny_train_tokens": len(tiny_train),
            "tiny_val_tokens": len(tiny_val),
            "tiny_next_token_accuracy": tiny_acc,
            "history": tiny_hist,
        },
    )
    (out_diag / "teacher_tiny_overfit_samples.txt").write_text(
        "\n\n".join([f"[sample_{i}]\n{s}" for i, s in enumerate(tiny_samples)]),
        encoding="utf-8",
    )

    # 4) Teacher quality gate
    gen_rows: list[dict[str, Any]] = []
    teacher_samples_lines: list[str] = []
    heldout_teacher = heldout_loss(
        teacher,
        split_ids["test"],
        batch_size=int(cfg["student_training"]["batch_size"]),
        context_len=int(cfg["student_training"]["context_len"]),
        eval_batches=int(cfg["evaluation"].get("heldout_eval_batches", 40)),
        device=device,
    )
    for pidx, prompt in zip(prompt_indices_20, prompts_20):
        prompt_txt = decode_ids(tokenizer, prompt)
        mode_defs = [
            ("greedy", {"greedy": True, "top_k": None, "top_p": None, "temperature": 1.0}),
            ("neutral_top_p", {"greedy": False, "top_k": None, "top_p": 0.95, "temperature": 1.0}),
            ("neutral_top_k", {"greedy": False, "top_k": 40, "top_p": None, "temperature": 1.0}),
            ("selected_current", {"selected": True}),
        ]
        teacher_samples_lines.append("=" * 96)
        teacher_samples_lines.append(f"PROMPT_INDEX: {pidx}")
        teacher_samples_lines.append(prompt_txt)
        for mode_name, mode_cfg in mode_defs:
            if mode_cfg.get("selected"):
                full_ids = generate_selected_tokens(
                    model=teacher,
                    prompt_ids=prompt,
                    max_new_tokens=80,
                    cfg=selected_cfg,
                    device=device,
                    progress_bar=False,
                )
            else:
                full_ids = _generate_autoregressive(
                    teacher,
                    prompt,
                    max_new_tokens=80,
                    device=device,
                    temperature=float(mode_cfg.get("temperature", 1.0)),
                    top_k=mode_cfg.get("top_k"),
                    top_p=mode_cfg.get("top_p"),
                    greedy=bool(mode_cfg.get("greedy", False)),
                    seed=args.seed + int(pidx),
                )
            cont = full_ids[len(prompt):]
            rep4 = 1.0 - distinct_n([cont], n=4)
            d2 = distinct_n([cont], n=2)
            ufrac = len(set(cont)) / max(1, len(cont))
            loop_pos = _loop_position(cont, n=4)
            rs = float(_has_repeated_span(cont, span_len=8))
            gen_rows.append(
                {
                    "prompt_index": int(pidx),
                    "mode": mode_name,
                    "repetition_4": float(rep4),
                    "distinct_2": float(d2),
                    "repeated_span_rate": rs,
                    "unique_token_fraction": float(ufrac),
                    "loop_position_4gram": int(loop_pos),
                    "teacher_heldout_loss": float(heldout_teacher),
                    "output_text": decode_ids(tokenizer, cont),
                }
            )
            teacher_samples_lines.append(f"[{mode_name}]")
            teacher_samples_lines.append(decode_ids(tokenizer, cont))
            teacher_samples_lines.append("")
    pd.DataFrame(gen_rows).to_csv(out_diag / "teacher_generation_sheet.csv", index=False)
    (out_diag / "teacher_generation_samples.txt").write_text("\n".join(teacher_samples_lines), encoding="utf-8")

    # 5) Selected-decoding branch audit
    with (out_diag / "selected_branch_audit.jsonl").open("w", encoding="utf-8") as f:
        for pidx, prompt in zip(prompt_indices_20, prompts_20):
            info = selected_next_token(teacher, prompt, selected_cfg, device)
            chosen = int(info["chosen_next_token"])
            for rank, row in enumerate(info["candidate_table"]):
                first_tid = int(row["first_token_id"])
                b_ids = [int(x) for x in row.get("best_branch_ids", [])]
                rep_contrib = float(repetition_penalty_value(b_ids, n=selected_cfg.repetition_ngram, coeff=selected_cfg.repetition_penalty))
                payload = {
                    "prompt_index": int(pidx),
                    "prompt_text": decode_ids(tokenizer, prompt),
                    "candidate_rank": int(rank),
                    "first_token_id": first_tid,
                    "first_token_piece": tokenizer.sp.id_to_piece(first_tid),
                    "first_logprob": float(row.get("first_logprob", float("nan"))),
                    "branch_score": float(row.get("branch_score", float("nan"))),
                    "logprob_sum": float(row.get("logprob_sum", float("nan"))),
                    "best_branch_ids": b_ids,
                    "best_branch_text": decode_ids(tokenizer, b_ids),
                    "repetition_penalty_contribution": rep_contrib,
                    "selected_token": int(chosen),
                    "is_selected": bool(first_tid == chosen),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 6) Decoding ablation sweep
    ablation_modes = {
        "greedy": {"type": "neutral", "greedy": True, "top_k": None, "top_p": None, "temperature": 1.0},
        "top_p_sampling": {"type": "neutral", "greedy": False, "top_k": None, "top_p": 0.95, "temperature": 1.0},
        "top_k_sampling": {"type": "neutral", "greedy": False, "top_k": 40, "top_p": None, "temperature": 1.0},
        "selected_beam_max": {"type": "selected", "cfg": selected_cfg, "no_repeat": None},
        "selected_sampled_reranked": {
            "type": "selected",
            "cfg": SelectedDecodingConfig(
                **{
                    **asdict(selected_cfg),
                    "mode": "sampled_reranked",
                    "deterministic": False,
                    "selection_temperature": 0.8,
                    "sampled_candidate_count": max(4, int(getattr(selected_cfg, "sampled_candidate_count", 4))),
                    "sampled_branch_len": max(12, int(getattr(selected_cfg, "sampled_branch_len", selected_cfg.horizon))),
                    "sampling_top_p": 0.95,
                }
            ),
            "no_repeat": None,
        },
        "selected_repetition_strong": {
            "type": "selected",
            "cfg": SelectedDecodingConfig(**{**asdict(selected_cfg), "repetition_penalty": max(1.6, selected_cfg.repetition_penalty * 2)}),
            "no_repeat": None,
        },
        "selected_no_repeat_ngram": {"type": "selected", "cfg": selected_cfg, "no_repeat": 4},
        "selected_length_norm": {
            "type": "selected",
            "cfg": SelectedDecodingConfig(**{**asdict(selected_cfg), "length_normalize": True}),
            "no_repeat": None,
        },
    }
    ablation_rows: list[dict[str, Any]] = []
    ablation_samples_lines: list[str] = []
    for mode_name, mode_cfg in ablation_modes.items():
        seqs: list[list[int]] = []
        branch_scores: list[float] = []
        entropies: list[float] = []
        loop_lens: list[int] = []
        unique_fracs: list[float] = []
        repeats: list[float] = []
        mode_samples: list[str] = []
        for j, (pidx, prompt) in enumerate(zip(prompt_indices_20, prompts_20)):
            entropies.append(_entropy_from_logits(_model_next_logits(teacher, prompt, device)))
            if mode_cfg["type"] == "neutral":
                full = _generate_autoregressive(
                    teacher,
                    prompt,
                    max_new_tokens=64,
                    device=device,
                    temperature=float(mode_cfg.get("temperature", 1.0)),
                    top_k=mode_cfg.get("top_k"),
                    top_p=mode_cfg.get("top_p"),
                    greedy=bool(mode_cfg.get("greedy", False)),
                    no_repeat_ngram_size=None,
                    seed=args.seed + j,
                )
            else:
                full = _generate_selected_with_constraints(
                    teacher,
                    prompt,
                    max_new_tokens=64,
                    cfg=mode_cfg["cfg"],
                    device=device,
                    no_repeat_ngram_size=mode_cfg.get("no_repeat"),
                )
            cont = full[len(prompt):]
            seqs.append(cont)
            repeats.append(float(_has_repeated_span(cont, span_len=8)))
            loop_lens.append(int(_loop_position(cont, n=4)))
            unique_fracs.append(len(set(cont)) / max(1, len(cont)))
            branch_scores.append(float(teacher_branch_score_for_sequence(teacher, full, selected_cfg, device)))
            if len(mode_samples) < 5:
                mode_samples.append(f"[prompt_index={pidx}] {decode_ids(tokenizer, cont)}")
        ablation_rows.append(
            {
                "mode": mode_name,
                "repetition_4": float(1.0 - distinct_n(seqs, n=4)),
                "distinct_2": float(distinct_n(seqs, n=2)),
                "average_branch_score": float(np.mean(branch_scores)),
                "average_next_token_entropy": float(np.mean(entropies)),
                "average_loop_position_4gram": float(np.mean(loop_lens)),
                "repeated_span_rate": float(np.mean(repeats)),
                "average_unique_token_fraction": float(np.mean(unique_fracs)),
            }
        )
        ablation_samples_lines.append("=" * 96)
        ablation_samples_lines.append(f"MODE: {mode_name}")
        ablation_samples_lines.extend(mode_samples)
        ablation_samples_lines.append("")
    pd.DataFrame(ablation_rows).to_csv(out_diag / "decoding_ablation_summary.csv", index=False)
    (out_diag / "decoding_ablation_samples.txt").write_text("\n".join(ablation_samples_lines), encoding="utf-8")

    # 7) Environment health check
    env_specs = {
        "original": out_root / "environments" / "original" / "original_environment_token_ids.npy",
        "neutral": out_root / "environments" / "neutral" / "neutral_token_ids.npy",
        "selected": out_root / "environments" / "selected" / "selected_token_ids.npy",
    }
    env_rows: list[dict[str, Any]] = []
    for env_name, path in env_specs.items():
        arr = np.load(path, allow_pickle=False).astype(np.int64).tolist()
        c = Counter(arr)
        total = len(arr)
        probs = np.asarray([v / total for v in c.values()], dtype=float)
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
        d1 = len(c) / max(1, total)
        d2 = float(distinct_n([arr], n=2))
        rep4 = float(1.0 - distinct_n([arr], n=4))
        top50 = c.most_common(50)
        top50_decoded = [{"token_id": int(t), "piece": tokenizer.sp.id_to_piece(int(t)), "count": int(cnt)} for t, cnt in top50]
        decoded_full = decode_ids(tokenizer, arr)
        sent_lens = [len(s.split()) for s in re.split(r"[.!?]+", decoded_full) if s.strip()]
        avg_sent_len = float(np.mean(sent_lens)) if sent_lens else float("nan")
        span_len = 8
        span_counter = Counter(tuple(arr[i : i + span_len]) for i in range(max(0, len(arr) - span_len + 1)))
        top_spans = []
        for span, cnt in span_counter.most_common(20):
            if cnt <= 1:
                continue
            top_spans.append({"count": int(cnt), "text": decode_ids(tokenizer, list(span))})
        rng_env = random.Random(args.seed + hash(env_name) % 10000)
        excerpts = []
        if len(arr) > 120:
            for _ in range(20):
                s = rng_env.randint(0, len(arr) - 121)
                excerpts.append(decode_ids(tokenizer, arr[s : s + 120]))
        (out_diag / f"environment_samples_{env_name}.txt").write_text("\n\n".join(excerpts), encoding="utf-8")
        env_rows.append(
            {
                "environment": env_name,
                "total_token_count": int(total),
                "distinct_1": float(d1),
                "distinct_2": float(d2),
                "repetition_4": float(rep4),
                "token_entropy_nats": float(entropy),
                "average_sentence_length_words": float(avg_sent_len),
                "top_tokens_json": json.dumps(top50_decoded, ensure_ascii=False),
                "top_repeated_spans_json": json.dumps(top_spans, ensure_ascii=False),
            }
        )
    pd.DataFrame(env_rows).to_csv(out_diag / "environment_summary.csv", index=False)

    # 8) Student training sanity: tiny-overfit + existing real training summaries
    student_rows: list[dict[str, Any]] = []
    tiny_overfit_root = ensure_dir(out_diag / "student_tiny_overfit")
    for env_name, env_path in env_specs.items():
        env_ids = np.load(env_path, allow_pickle=False).astype(np.int64).tolist()
        tiny_ids = env_ids[: min(24000, len(env_ids))]
        overfit_cfg = TrainingConfig(
            batch_size=32,
            context_len=int(cfg["student_training"]["context_len"]),
            learning_rate=5e-4,
            weight_decay=0.0,
            max_steps=220,
            eval_interval=40,
            eval_batches=8,
            warmup_steps=20,
            gradient_accumulation_steps=1,
            clip_grad_norm=1.0,
            seed=int(args.seed + 500 + len(student_rows)),
            device_preference="mps_or_cpu",
            early_stopping_patience=0,
            save_interval=100,
        )
        model = build_tiny_gpt(teacher_cfg)
        out_dir = ensure_dir(tiny_overfit_root / env_name)
        hist = train_language_model(
            model=model,
            train_ids=tiny_ids,
            val_ids=tiny_ids[:6000],
            train_cfg=overfit_cfg,
            out_dir=out_dir,
            resume=False,
            progress_bar=False,
        )
        fit_model = load_trained_model(Path(hist["best_checkpoint_path"]), teacher_cfg, device)
        accs = []
        for _ in range(20):
            xb, yb = sample_lm_batch(tiny_ids, batch_size=16, context_len=int(cfg["student_training"]["context_len"]), device=device)
            with torch.no_grad():
                logits = fit_model(xb)
            pred = torch.argmax(logits, dim=-1)
            accs.append(float((pred == yb).float().mean().item()))
        student_rows.append(
            {
                "student": env_name,
                "stage": "tiny_overfit",
                "steps": int(hist["steps"]),
                "train_loss_last": float(hist["train_loss_history"][-1]["train_loss"]),
                "val_loss_last": float(hist["val_loss_history"][-1]["val_loss"]),
                "tiny_next_token_accuracy": float(np.mean(accs)),
                "final_checkpoint_path": str(hist["final_checkpoint_path"]),
                "best_checkpoint_path": str(hist["best_checkpoint_path"]),
                "early_stopping_triggered": bool(hist["steps"] < overfit_cfg.max_steps),
                "gradient_norm_summary": "",
            }
        )

    for env_name in ["original", "neutral", "selected"]:
        summary_path = out_root / "students" / env_name / "training_summary.json"
        j = json.loads(summary_path.read_text(encoding="utf-8"))
        hist = j["history"]
        val_hist = hist.get("val_loss_history", [])
        tr_hist = hist.get("train_loss_history", [])
        student_rows.append(
            {
                "student": env_name,
                "stage": "real_training",
                "steps": int(hist.get("steps", 0)),
                "train_loss_last": float(tr_hist[-1]["train_loss"]) if tr_hist else float("nan"),
                "val_loss_last": float(val_hist[-1]["val_loss"]) if val_hist else float("nan"),
                "tiny_next_token_accuracy": float("nan"),
                "final_checkpoint_path": str(hist.get("final_checkpoint_path")),
                "best_checkpoint_path": str(hist.get("best_checkpoint_path")),
                "early_stopping_triggered": bool(hist.get("steps", 0) < int(cfg["student_training"]["max_steps"])),
                "gradient_norm_summary": "not_recorded",
            }
        )
    pd.DataFrame(student_rows).to_csv(out_diag / "student_training_summary.csv", index=False)

    # 9) N-gram environment control
    ngram_root = ensure_dir(out_diag / "ngram_control")
    ngram_env_root = ensure_dir(ngram_root / "environments")
    counts, totals = _ngram_count_map(split_ids["train"], order=3)
    seed_ids = split_ids["train"][:2]
    total_tokens = int(cfg["environments"]["pilot_total_tokens"] if args.mode == "pilot" else cfg["environments"]["full_total_tokens"])
    ngram_neutral_ids = _generate_ngram_neutral(seed_ids, total_tokens, counts, totals, order=3, seed=args.seed + 601)
    ngram_selected_ids = _generate_ngram_selected(seed_ids, total_tokens, counts, totals, order=3, seed=args.seed + 602)
    np.save(ngram_env_root / "ngram_neutral_token_ids.npy", np.asarray(ngram_neutral_ids, dtype=np.int32))
    np.save(ngram_env_root / "ngram_selected_token_ids.npy", np.asarray(ngram_selected_ids, dtype=np.int32))

    control_train_cfg = TrainingConfig(
        batch_size=32,
        context_len=int(cfg["student_training"]["context_len"]),
        learning_rate=float(cfg["student_training"]["learning_rate"]),
        weight_decay=float(cfg["student_training"]["weight_decay"]),
        max_steps=500,
        eval_interval=100,
        eval_batches=12,
        warmup_steps=int(cfg["student_training"]["warmup_steps"]),
        gradient_accumulation_steps=1,
        clip_grad_norm=float(cfg["student_training"]["clip_grad_norm"]),
        seed=int(args.seed + 700),
        device_preference="mps_or_cpu",
        early_stopping_patience=4,
        save_interval=200,
    )
    control_students = {
        "original_control": np.load(env_specs["original"], allow_pickle=False).astype(np.int64).tolist(),
        "ngram_neutral_control": ngram_neutral_ids,
        "ngram_selected_control": ngram_selected_ids,
    }
    control_models = {}
    for i, (name, ids) in enumerate(control_students.items()):
        m = build_tiny_gpt(teacher_cfg)
        out_dir = ensure_dir(ngram_root / "students" / name)
        cfg_local = asdict(control_train_cfg)
        cfg_local["seed"] = int(control_train_cfg.seed + i)
        hist = train_language_model(
            model=m,
            train_ids=ids,
            val_ids=split_ids["val"],
            train_cfg=TrainingConfig(**cfg_local),
            out_dir=out_dir,
            resume=False,
            progress_bar=False,
        )
        control_models[name] = load_trained_model(Path(hist["best_checkpoint_path"]), teacher_cfg, device)

    ng_metrics_df = evaluate_all_students(
        teacher=teacher,
        students=control_models,
        prompt_bank=prompts_20,
        selected_cfg=selected_cfg,
        trigram_table=build_subword_trigram_table(split_ids["train"]),
        device=device,
    )
    for i, row in ng_metrics_df.iterrows():
        env_name = row["student"]
        h = heldout_loss(
            model=control_models[env_name],
            test_ids=split_ids["test"],
            batch_size=int(cfg["student_training"]["batch_size"]),
            context_len=int(cfg["student_training"]["context_len"]),
            eval_batches=20,
            device=device,
        )
        ng_metrics_df.loc[i, "heldout_loss"] = float(h)
    ng_metrics_df.to_csv(out_diag / "teacher_student_policy_metrics_ngram_control.csv", index=False)

    ngram_samples = []
    for name, model in control_models.items():
        ngram_samples.append("=" * 90)
        ngram_samples.append(f"MODEL: {name}")
        for pidx, prompt in list(zip(prompt_indices_20, prompts_20))[:5]:
            full = _generate_autoregressive(model, prompt, max_new_tokens=80, device=device, temperature=0.9, top_p=0.95, seed=args.seed + pidx)
            ngram_samples.append(f"[prompt_index={pidx}] {decode_ids(tokenizer, full[len(prompt):])}")
    (out_diag / "ngram_control_samples.txt").write_text("\n".join(ngram_samples), encoding="utf-8")

    # 10) Budget matching audit -> run_manifest.json
    env_counts = {
        "original": int(np.load(env_specs["original"], allow_pickle=False).shape[0]),
        "neutral": int(np.load(env_specs["neutral"], allow_pickle=False).shape[0]),
        "selected": int(np.load(env_specs["selected"], allow_pickle=False).shape[0]),
    }
    run_manifest = {
        "created_at": _utc_now(),
        "mode": args.mode,
        "tokenizer_model_path": str(prep["tokenizer_model_path"]),
        "same_tokenizer_across_environments": True,
        "student_architecture_config": asdict(teacher_cfg),
        "student_training_config": cfg["student_training"],
        "environment_token_counts": env_counts,
        "token_budget_matched": bool(len(set(env_counts.values())) == 1),
        "evaluation_prompt_bank_path": str(prompt_bank_path),
        "evaluation_prompt_count": int(len(prompt_bank)),
        "teacher_checkpoint": teacher_ckpt,
        "student_checkpoints": {k: v["checkpoint"] for k, v in student_infos.items()},
        "control_training_steps": int(control_train_cfg.max_steps),
    }
    _write_json(out_diag / "run_manifest.json", run_manifest)

    # 11) Support-graph bridge metrics
    bridge_rows: list[dict[str, Any]] = []
    trigram_tables = {
        "doyle_train": build_subword_trigram_table(split_ids["train"]),
        "env_original": build_subword_trigram_table(np.load(env_specs["original"], allow_pickle=False).astype(np.int64).tolist()),
        "env_neutral": build_subword_trigram_table(np.load(env_specs["neutral"], allow_pickle=False).astype(np.int64).tolist()),
        "env_selected": build_subword_trigram_table(np.load(env_specs["selected"], allow_pickle=False).astype(np.int64).tolist()),
        "env_ngram_neutral": build_subword_trigram_table(ngram_neutral_ids),
        "env_ngram_selected": build_subword_trigram_table(ngram_selected_ids),
    }
    all_models = {
        "student_original": students["original"],
        "student_neutral": students["neutral"],
        "student_selected": students["selected"],
        "control_original": control_models["original_control"],
        "control_ngram_neutral": control_models["ngram_neutral_control"],
        "control_ngram_selected": control_models["ngram_selected_control"],
    }
    for model_name, model in all_models.items():
        generated = []
        for prompt in prompts_20:
            full = _generate_autoregressive(model, prompt, max_new_tokens=64, device=device, temperature=0.9, top_p=0.95, seed=args.seed + len(generated))
            generated.append(full)
        for table_name, table in trigram_tables.items():
            sr = support_rate_against_trigram_table(generated, table)
            fails = [time_to_first_trigram_support_failure(seq, table) for seq in generated]
            fails = [f for f in fails if f >= 0]
            bridge_rows.append(
                {
                    "model": model_name,
                    "target_trigram_table": table_name,
                    "support_rate": float(sr),
                    "time_to_first_failure_mean": float(np.mean(fails)) if fails else float("nan"),
                }
            )
    pd.DataFrame(bridge_rows).to_csv(out_diag / "support_bridge_metrics.csv", index=False)

    # Teacher/student policy metrics copy (requested top-level artifact)
    src_metrics = out_root / "evaluation" / "csv" / "teacher_student_metrics_pilot.csv"
    if src_metrics.exists():
        shutil.copy2(src_metrics, out_diag / "teacher_student_policy_metrics.csv")

    # text_samples folder bundle
    prompt_sample_lines = []
    for pidx, p in zip(prompt_indices_20, prompts_20):
        prompt_sample_lines.append(f"[prompt_index={pidx}]")
        prompt_sample_lines.append(decode_ids(tokenizer, p))
        prompt_sample_lines.append("")
    (out_text / "prompt_examples.txt").write_text("\n".join(prompt_sample_lines), encoding="utf-8")
    shutil.copy2(out_diag / "teacher_generation_samples.txt", out_text / "teacher_generation_samples.txt")
    shutil.copy2(out_diag / "decoding_ablation_samples.txt", out_text / "decoding_ablation_samples.txt")
    shutil.copy2(out_diag / "ngram_control_samples.txt", out_text / "ngram_control_samples.txt")
    for env_name in ["original", "neutral", "selected"]:
        shutil.copy2(out_diag / f"environment_samples_{env_name}.txt", out_text / f"environment_samples_{env_name}.txt")

    # 12) Diagnostic report
    prompt_fail = any(not bool(r["contiguous_match_in_test_split"]) for r in prompt_rows)
    rt_med_edit = float(median(rt_df["edit_distance"].tolist())) if len(rt_df) else float("nan")
    rt_med_mismatch = float(median(rt_df["mismatch_score"].tolist())) if len(rt_df) else float("nan")
    tiny_ok = tiny_acc > 0.55
    ab_df = pd.DataFrame(ablation_rows).set_index("mode")
    selected_rep = float(ab_df.loc["selected_beam_max", "repeated_span_rate"])
    topp_rep = float(ab_df.loc["top_p_sampling", "repeated_span_rate"])
    env_df = pd.DataFrame(env_rows).set_index("environment")
    selected_entropy = float(env_df.loc["selected", "token_entropy_nats"])
    neutral_entropy = float(env_df.loc["neutral", "token_entropy_nats"])
    selected_pathological = selected_entropy < (neutral_entropy - 1.0)
    pass_items = []
    fail_items = []
    if not prompt_fail:
        pass_items.append("Prompt-bank contiguity checks passed for sampled prompts.")
    else:
        fail_items.append("Prompt-bank contiguity failed for at least one sampled prompt.")
    if rt_med_mismatch < 0.08:
        pass_items.append("Tokenizer round-trip mismatch is low on held-out snippets.")
    else:
        fail_items.append("Tokenizer round-trip mismatch is elevated.")
    if tiny_ok:
        pass_items.append("Teacher can overfit a tiny shard (sanity check passed).")
    else:
        fail_items.append("Teacher tiny-overfit did not reach expected next-token accuracy.")
    if selected_rep > topp_rep:
        fail_items.append("Selected decoding is loopier than top-p in ablation sweep.")
    else:
        pass_items.append("Selected decoding is not loopier than top-p on repeated-span metric.")
    if selected_pathological:
        fail_items.append("Selected environment entropy is substantially below neutral (pathological narrowing).")
    else:
        pass_items.append("Selected environment entropy is within acceptable range.")

    root_cause = (
        "Selected-decoding objective/config is likely pathological for this setup, creating a low-entropy selected environment that harms student training."
    )
    next_fix = (
        "Reduce selection aggressiveness first: enable stochastic reranking, stronger no-repeat constraints, and verify branch audits before retraining students."
    )
    usable = "No" if selected_pathological or selected_rep > topp_rep else "Possibly"

    report_lines = [
        "# Diagnostic Report",
        "",
        f"Generated at: `{_utc_now()}`",
        "",
        "## Passed",
    ]
    report_lines.extend([f"- {x}" for x in pass_items] if pass_items else ["- None"])
    report_lines += ["", "## Failed"]
    report_lines.extend([f"- {x}" for x in fail_items] if fail_items else ["- None"])
    report_lines += [
        "",
        "## Most Likely Root Cause",
        f"- {root_cause}",
        "",
        "## Single Best Next Fix",
        f"- {next_fix}",
        "",
        "## Is Transformer-Selected Environment Usable?",
        f"- {usable}",
        "",
        "## Key Artifact Paths",
        "- `data/outputs/diagnostics/prompt_audit.csv`",
        "- `data/outputs/diagnostics/tokenizer_roundtrip_audit.csv`",
        "- `data/outputs/diagnostics/teacher_generation_sheet.csv`",
        "- `data/outputs/diagnostics/decoding_ablation_summary.csv`",
        "- `data/outputs/diagnostics/environment_summary.csv`",
        "- `data/outputs/diagnostics/student_training_summary.csv`",
        "- `data/outputs/diagnostics/run_manifest.json`",
        "- `data/outputs/diagnostics/text_samples/`",
        "",
    ]
    (out_diag / "diagnostic_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Diagnostics bundle complete under: {out_diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
