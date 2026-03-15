from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .selected_decoding import SelectedDecodingConfig, branch_objective, teacher_selected_policy_token
from .training import sample_lm_batch


def support_rate(nonzero_items: int, total_items: int) -> float:
    if total_items <= 0:
        return 0.0
    return nonzero_items / float(total_items)


def _normalize_sequences(sequences: list[list[int]] | list[str] | list[list[str]]) -> list[list[Any]]:
    if not sequences:
        return []
    first = sequences[0]
    if isinstance(first, str):
        return [str(s).split() for s in sequences]  # type: ignore[arg-type]
    return [list(s) for s in sequences]  # type: ignore[arg-type]


def distinct_n(
    sequences: list[list[int]] | list[str],
    n: int = 2,
) -> float:
    seqs = _normalize_sequences(sequences)
    if n <= 0 or not seqs:
        return 0.0
    grams: list[tuple[Any, ...]] = []
    for seq in seqs:
        if len(seq) < n:
            continue
        grams.extend(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repetition_n(
    sequences: list[list[int]] | list[str],
    n: int = 4,
) -> float:
    return 1.0 - distinct_n(sequences, n=n)


def shannon_entropy(probabilities: dict[str, float]) -> float:
    ent = 0.0
    for p in probabilities.values():
        if p > 0:
            ent -= p * math.log(p)
    return ent


def kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), eps)
        qk = max(q.get(k, 0.0), eps)
        out += pk * math.log(pk / qk)
    return out


def token_histogram(tokens: list[str]) -> dict[str, int]:
    return dict(Counter(tokens))


def _next_token_logprobs(model, prompt_ids: list[int], device) -> torch.Tensor:
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        logprobs = F.log_softmax(logits, dim=-1)
    return logprobs[0]


def selected_teacher_logprob_for_student(
    teacher,
    student,
    prompt_ids: list[int],
    selected_cfg,
    device,
) -> float:
    token = teacher_selected_policy_token(teacher, prompt_ids, selected_cfg, device)
    student_lp = _next_token_logprobs(student, prompt_ids, device)
    return float(student_lp[token].item())


def selected_teacher_top1_match(
    teacher,
    student,
    prompt_ids: list[int],
    selected_cfg,
    device,
) -> int:
    token = teacher_selected_policy_token(teacher, prompt_ids, selected_cfg, device)
    student_lp = _next_token_logprobs(student, prompt_ids, device)
    top1 = int(torch.argmax(student_lp).item())
    return int(top1 == token)


def evaluate_policy_agreement(
    teacher,
    student,
    prompt_bank: list[list[int]],
    selected_cfg,
    device,
) -> dict:
    if not prompt_bank:
        return {
            "mean_selected_logprob": float("nan"),
            "mean_selected_probability": float("nan"),
            "top1_match_rate": float("nan"),
            "num_prompts": 0,
        }

    logps: list[float] = []
    probs: list[float] = []
    matches: list[int] = []

    for prompt in prompt_bank:
        lp = selected_teacher_logprob_for_student(teacher, student, prompt, selected_cfg, device)
        logps.append(lp)
        probs.append(math.exp(lp))
        matches.append(selected_teacher_top1_match(teacher, student, prompt, selected_cfg, device))

    return {
        "mean_selected_logprob": float(np.mean(logps)),
        "mean_selected_probability": float(np.mean(probs)),
        "top1_match_rate": float(np.mean(matches)),
        "num_prompts": int(len(prompt_bank)),
    }


def teacher_branch_score_for_sequence(
    teacher,
    full_sequence_ids: list[int],
    selected_cfg,
    device,
) -> float:
    if len(full_sequence_ids) < 2:
        return float("nan")
    logprob_sum = 0.0
    for i in range(1, len(full_sequence_ids)):
        prefix = full_sequence_ids[:i]
        target = full_sequence_ids[i]
        lp = _next_token_logprobs(teacher, prefix, device)
        logprob_sum += float(lp[target].item())
    return float(branch_objective(logprob_sum, full_sequence_ids, selected_cfg))


def evaluate_branch_scores(
    teacher,
    model,
    prompt_bank: list[list[int]],
    selected_cfg,
    generation_kwargs: dict,
    device,
) -> dict:
    if not prompt_bank:
        return {
            "mean_branch_score": float("nan"),
            "std_branch_score": float("nan"),
            "se_branch_score": float("nan"),
            "num_prompts": 0,
        }

    scores: list[float] = []
    max_new_tokens = int(generation_kwargs.get("max_new_tokens", selected_cfg.horizon))
    temperature = float(generation_kwargs.get("temperature", 1.0))
    top_k = generation_kwargs.get("top_k")
    top_p = generation_kwargs.get("top_p")

    for prompt in prompt_bank:
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        seq = [int(t) for t in out[0].tolist()]
        scores.append(teacher_branch_score_for_sequence(teacher, seq, selected_cfg, device))

    arr = np.asarray(scores, dtype=float)
    return {
        "mean_branch_score": float(np.mean(arr)),
        "std_branch_score": float(np.std(arr)),
        "se_branch_score": float(np.std(arr) / max(1, np.sqrt(len(arr)))),
        "num_prompts": int(len(scores)),
    }


@torch.no_grad()
def heldout_loss(
    model,
    test_ids,
    batch_size,
    context_len,
    eval_batches,
    device,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        xb, yb = sample_lm_batch(test_ids, batch_size=batch_size, context_len=context_len, device=device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def build_subword_trigram_table(
    token_ids: list[int],
) -> set[tuple[int, int, int]]:
    if len(token_ids) < 3:
        return set()
    return {tuple(token_ids[i : i + 3]) for i in range(len(token_ids) - 2)}


def support_rate_against_trigram_table(
    sequences: list[list[int]],
    trigram_table: set[tuple[int, int, int]],
) -> float:
    if not sequences:
        return float("nan")
    total = 0
    supported = 0
    for seq in sequences:
        if len(seq) < 3:
            continue
        for i in range(len(seq) - 2):
            total += 1
            if tuple(seq[i : i + 3]) in trigram_table:
                supported += 1
    if total == 0:
        return 0.0
    return supported / total


def time_to_first_trigram_support_failure(
    sequence: list[int],
    trigram_table: set[tuple[int, int, int]],
) -> int:
    if len(sequence) < 3:
        return -1
    for i in range(len(sequence) - 2):
        if tuple(sequence[i : i + 3]) not in trigram_table:
            return i
    return len(sequence) - 2


def evaluate_all_students(
    teacher,
    students: dict[str, object],
    prompt_bank: list[list[int]],
    selected_cfg,
    trigram_table=None,
    device=None,
) -> pd.DataFrame:
    if device is None:
        device = next(teacher.parameters()).device

    rows: list[dict[str, Any]] = []
    for name, student in students.items():
        pol = evaluate_policy_agreement(
            teacher=teacher,
            student=student,
            prompt_bank=prompt_bank,
            selected_cfg=selected_cfg,
            device=device,
        )
        branch = evaluate_branch_scores(
            teacher=teacher,
            model=student,
            prompt_bank=prompt_bank,
            selected_cfg=selected_cfg,
            generation_kwargs={"max_new_tokens": selected_cfg.horizon, "temperature": 1.0},
            device=device,
        )

        generated: list[list[int]] = []
        for prompt in prompt_bank:
            x = torch.tensor([prompt], dtype=torch.long, device=device)
            with torch.no_grad():
                out = student.generate(x, max_new_tokens=selected_cfg.horizon, temperature=1.0, top_k=None, top_p=None)
            generated.append([int(t) for t in out[0].tolist()])

        row = {
            "student": name,
            **pol,
            **branch,
            "distinct_2": float(distinct_n(generated, n=2)),
            "repetition_4": float(repetition_n(generated, n=4)),
        }

        if trigram_table is not None:
            row["support_rate_trigram"] = float(support_rate_against_trigram_table(generated, trigram_table))
            failures = [time_to_first_trigram_support_failure(seq, trigram_table) for seq in generated]
            failures = [f for f in failures if f >= 0]
            row["time_to_failure_mean"] = float(np.mean(failures)) if failures else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)
