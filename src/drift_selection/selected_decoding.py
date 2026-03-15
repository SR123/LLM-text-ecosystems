from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@dataclass
class SelectedDecodingConfig:
    mode: str = "beam_max"  # beam_max | sampled_reranked
    horizon: int = 4
    first_step_top_k: int = 8
    beam_width: int = 4
    branch_temperature: float = 1.0
    selection_temperature: float = 1.0
    sampling_top_k: int | None = None
    sampling_top_p: float | None = 0.95
    sampled_candidate_count: int = 4
    sampled_branch_len: int = 16
    repetition_ngram: int = 4
    repetition_penalty: float = 0.8
    repeated_span_length: int = 8
    repeated_span_penalty: float = 0.0
    unique_token_floor: float = 0.0
    unique_token_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    max_identical_token_run: int = 0
    length_normalize: bool = True
    deterministic: bool = True
    device_preference: str = "mps_or_cpu"


def top_k_candidates_from_logits(
    logits: torch.Tensor,
    top_k: int,
) -> list[tuple[int, float]]:
    logprobs = F.log_softmax(logits, dim=-1)
    vals, idx = torch.topk(logprobs, k=min(top_k, logprobs.size(-1)))
    return [(int(i.item()), float(v.item())) for i, v in zip(idx, vals)]


def ngram_repeat_count(
    token_ids: list[int],
    n: int,
) -> int:
    if n <= 1 or len(token_ids) < n:
        return 0
    seen = set()
    repeats = 0
    for i in range(len(token_ids) - n + 1):
        g = tuple(token_ids[i : i + n])
        if g in seen:
            repeats += 1
        else:
            seen.add(g)
    return repeats


def repetition_penalty_value(
    token_ids: list[int],
    n: int = 4,
    coeff: float = 0.8,
) -> float:
    return coeff * float(ngram_repeat_count(token_ids, n=n))


def _repeated_span_count(token_ids: list[int], span_len: int = 8) -> int:
    if span_len <= 1 or len(token_ids) < span_len:
        return 0
    seen = set()
    repeats = 0
    for i in range(len(token_ids) - span_len + 1):
        span = tuple(token_ids[i : i + span_len])
        if span in seen:
            repeats += 1
        else:
            seen.add(span)
    return repeats


def _unique_token_fraction(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / float(len(token_ids))


def _max_identical_token_run(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    best = 1
    cur = 1
    prev = token_ids[0]
    for tid in token_ids[1:]:
        if tid == prev:
            cur += 1
            if cur > best:
                best = cur
        else:
            prev = tid
            cur = 1
    return best


def _violates_no_repeat_ngram(token_ids: list[int], next_id: int, n: int) -> bool:
    if n <= 1 or len(token_ids) < n - 1:
        return False
    prefix = tuple(token_ids[-(n - 1):])
    target = prefix + (int(next_id),)
    for i in range(len(token_ids) - n + 1):
        if tuple(token_ids[i : i + n]) == target:
            return True
    return False


def branch_objective(
    logprob_sum: float,
    branch_ids: list[int],
    cfg: SelectedDecodingConfig,
) -> float:
    score = float(logprob_sum)
    score -= repetition_penalty_value(branch_ids, n=cfg.repetition_ngram, coeff=cfg.repetition_penalty)
    score -= float(cfg.repeated_span_penalty) * float(
        _repeated_span_count(branch_ids, span_len=max(2, int(cfg.repeated_span_length)))
    )

    if cfg.unique_token_floor > 0 and cfg.unique_token_penalty > 0 and branch_ids:
        uniq = _unique_token_fraction(branch_ids)
        deficit = max(0.0, float(cfg.unique_token_floor) - uniq)
        score -= float(cfg.unique_token_penalty) * deficit * float(len(branch_ids))

    if cfg.length_normalize and branch_ids:
        score = score / len(branch_ids)
    return score


def _last_logits(model, prefix_ids: list[int], device) -> torch.Tensor:
    ctx_len = int(getattr(getattr(model, "config", object()), "context_len", 128))
    max_seq_len = int(getattr(getattr(model, "config", object()), "max_seq_len", ctx_len))
    window = max(1, min(ctx_len, max_seq_len))
    prefix = prefix_ids[-window:]
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[:, -1, :]
    return logits[0]


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    out = logits.clone()
    if top_k is not None and top_k > 0:
        vals, _ = torch.topk(out, min(int(top_k), out.size(-1)))
        out = torch.where(out < vals[-1], torch.full_like(out, float("-inf")), out)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > float(top_p)
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        restored = torch.full_like(out, float("-inf"))
        restored.scatter_(0, sorted_idx, sorted_logits)
        out = restored
    return out


def _sample_token(logits: torch.Tensor, deterministic: bool) -> tuple[int, float]:
    logprobs = F.log_softmax(logits, dim=-1)
    if deterministic:
        tid = int(torch.argmax(logits).item())
        return tid, float(logprobs[tid].item())
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
        tid = int(torch.argmax(logits).item())
        return tid, float(logprobs[tid].item())
    tid = int(torch.multinomial(probs, num_samples=1).item())
    return tid, float(logprobs[tid].item())


def expand_one_beam_step(
    model,
    prefix_ids: list[int],
    beams: list[dict],
    cfg: SelectedDecodingConfig,
    device,
) -> list[dict]:
    expanded: list[dict] = []
    width = max(1, cfg.beam_width)

    for beam in beams:
        seq = beam["ids"]
        logits = _last_logits(model, seq, device=device)
        logits = logits / max(cfg.branch_temperature, 1e-6)
        candidates = top_k_candidates_from_logits(logits, top_k=width)
        for tok_id, lp in candidates:
            new_ids = seq + [tok_id]
            logprob_sum = float(beam["logprob_sum"] + lp)
            score = branch_objective(logprob_sum, new_ids[len(prefix_ids):], cfg)
            expanded.append(
                {
                    "ids": new_ids,
                    "logprob_sum": logprob_sum,
                    "score": score,
                }
            )

    expanded.sort(key=lambda b: b["score"], reverse=True)
    return expanded[:width]


def _sampled_branch_for_first_token(
    model,
    prefix_ids: list[int],
    first_token_id: int,
    first_logprob: float,
    cfg: SelectedDecodingConfig,
    device,
    sample_idx: int,
) -> dict[str, Any]:
    branch_len = max(1, int(cfg.sampled_branch_len or cfg.horizon))
    seq = list(prefix_ids) + [int(first_token_id)]
    logprob_sum = float(first_logprob)

    for _ in range(max(0, branch_len - 1)):
        logits = _last_logits(model, seq, device=device)
        logits = logits / max(cfg.branch_temperature, 1e-6)
        logits = _apply_top_k_top_p(logits, top_k=cfg.sampling_top_k, top_p=cfg.sampling_top_p)

        # Mask candidates that immediately violate anti-loop constraints.
        if cfg.no_repeat_ngram_size and cfg.no_repeat_ngram_size > 1:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for tid in range(logits.size(-1)):
                if _violates_no_repeat_ngram(seq, int(tid), int(cfg.no_repeat_ngram_size)):
                    mask[tid] = True
            logits = logits.masked_fill(mask, float("-inf"))

        if cfg.max_identical_token_run and cfg.max_identical_token_run > 0 and seq:
            max_run = _max_identical_token_run(seq)
            if max_run >= int(cfg.max_identical_token_run):
                logits[seq[-1]] = float("-inf")

        next_tid, next_lp = _sample_token(logits, deterministic=False)
        seq.append(int(next_tid))
        logprob_sum += float(next_lp)

    branch_ids = seq[len(prefix_ids):]
    score = branch_objective(logprob_sum=logprob_sum, branch_ids=branch_ids, cfg=cfg)
    return {
        "first_token_id": int(first_token_id),
        "best_branch_ids": branch_ids,
        "logprob_sum": float(logprob_sum),
        "branch_score": float(score),
        "first_logprob": float(first_logprob),
        "sample_idx": int(sample_idx),
    }


def _selected_next_token_sampled_reranked(
    model,
    prefix_ids: list[int],
    cfg: SelectedDecodingConfig,
    device,
) -> dict[str, Any]:
    logits = _last_logits(model, prefix_ids, device=device)
    first_candidates = top_k_candidates_from_logits(logits, top_k=max(1, cfg.first_step_top_k))
    rows: list[dict[str, Any]] = []

    n_samples = max(1, int(cfg.sampled_candidate_count))
    for tok_id, first_lp in first_candidates:
        for s in range(n_samples):
            rows.append(
                _sampled_branch_for_first_token(
                    model=model,
                    prefix_ids=prefix_ids,
                    first_token_id=tok_id,
                    first_logprob=first_lp,
                    cfg=cfg,
                    device=device,
                    sample_idx=s,
                )
            )

    rows.sort(key=lambda r: r["branch_score"], reverse=True)
    if cfg.deterministic:
        chosen = rows[0]
    else:
        scores = torch.tensor([r["branch_score"] for r in rows], dtype=torch.float32)
        scores = scores / max(cfg.selection_temperature, 1e-6)
        probs = torch.softmax(scores, dim=0)
        idx = int(torch.multinomial(probs, num_samples=1).item())
        chosen = rows[idx]

    return {
        "chosen_next_token": int(chosen["first_token_id"]),
        "candidate_table": rows,
        "best_branch_score": float(chosen["branch_score"]),
        "best_branch_ids": chosen["best_branch_ids"],
    }


def score_first_token_by_short_horizon_search(
    model,
    prefix_ids: list[int],
    first_token_id: int,
    cfg: SelectedDecodingConfig,
    device,
) -> dict:
    logits0 = _last_logits(model, prefix_ids, device=device)
    logprobs0 = F.log_softmax(logits0, dim=-1)
    first_lp = float(logprobs0[first_token_id].item())

    start = {
        "ids": list(prefix_ids) + [int(first_token_id)],
        "logprob_sum": first_lp,
        "score": first_lp,
    }
    beams = [start]

    # We already committed the first token; expand horizon-1 steps.
    for _ in range(max(0, cfg.horizon - 1)):
        beams = expand_one_beam_step(model=model, prefix_ids=prefix_ids, beams=beams, cfg=cfg, device=device)

    best = max(beams, key=lambda b: b["score"])
    branch_ids = best["ids"][len(prefix_ids):]
    return {
        "first_token_id": int(first_token_id),
        "best_branch_ids": branch_ids,
        "logprob_sum": float(best["logprob_sum"]),
        "branch_score": float(best["score"]),
    }


def selected_next_token(
    model,
    prefix_ids: list[int],
    cfg: SelectedDecodingConfig,
    device,
) -> dict:
    if cfg.mode == "sampled_reranked":
        return _selected_next_token_sampled_reranked(model=model, prefix_ids=prefix_ids, cfg=cfg, device=device)

    logits = _last_logits(model, prefix_ids, device=device)
    first_candidates = top_k_candidates_from_logits(logits, top_k=max(1, cfg.first_step_top_k))

    candidate_rows: list[dict[str, Any]] = []
    for tok_id, lp in first_candidates:
        scored = score_first_token_by_short_horizon_search(
            model=model,
            prefix_ids=prefix_ids,
            first_token_id=tok_id,
            cfg=cfg,
            device=device,
        )
        scored["first_logprob"] = float(lp)
        candidate_rows.append(scored)

    candidate_rows.sort(key=lambda r: r["branch_score"], reverse=True)

    if cfg.deterministic:
        chosen = candidate_rows[0]
    else:
        scores = torch.tensor([r["branch_score"] for r in candidate_rows], dtype=torch.float32)
        scores = scores / max(cfg.selection_temperature, 1e-6)
        probs = torch.softmax(scores, dim=0)
        idx = int(torch.multinomial(probs, num_samples=1).item())
        chosen = candidate_rows[idx]

    return {
        "chosen_next_token": int(chosen["first_token_id"]),
        "candidate_table": candidate_rows,
        "best_branch_score": float(chosen["branch_score"]),
        "best_branch_ids": chosen["best_branch_ids"],
    }


def generate_selected_tokens(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    cfg: SelectedDecodingConfig,
    device,
    progress_bar: bool = False,
) -> list[int]:
    out = list(prompt_ids)
    itr = range(max_new_tokens)
    if progress_bar:
        itr = tqdm(itr, desc="selected-decode", unit="tok")
    for _ in itr:
        step = selected_next_token(model=model, prefix_ids=out, cfg=cfg, device=device)
        out.append(int(step["chosen_next_token"]))
    return out


def compare_greedy_vs_selected_next_token(
    model,
    prompt_ids: list[int],
    cfg: SelectedDecodingConfig,
    device,
) -> dict:
    logits = _last_logits(model, prompt_ids, device=device)
    greedy = int(torch.argmax(logits).item())
    selected = selected_next_token(model=model, prefix_ids=prompt_ids, cfg=cfg, device=device)
    return {
        "greedy_next_token": greedy,
        "selected_next_token": int(selected["chosen_next_token"]),
        "candidate_table": selected["candidate_table"],
        "best_branch_ids": selected["best_branch_ids"],
        "best_branch_score": selected["best_branch_score"],
    }


def teacher_selected_policy_token(
    model,
    prompt_ids: list[int],
    cfg: SelectedDecodingConfig,
    device,
) -> int:
    out = selected_next_token(model=model, prefix_ids=prompt_ids, cfg=cfg, device=device)
    return int(out["chosen_next_token"])


def selected_decoding_step(base_probs: dict[str, float], viability: dict[str, float], horizon_weight: float = 1.0) -> dict[str, float]:
    # Backward-compatible lightweight utility used by earlier n-gram demos.
    weighted = {
        tok: p * (max(viability.get(tok, 0.0), 0.0) ** max(horizon_weight, 0.0))
        for tok, p in base_probs.items()
    }
    s = sum(max(0.0, v) for v in weighted.values())
    if s <= 0:
        n = len(weighted) or 1
        return {k: 1.0 / n for k in weighted}
    return {k: max(0.0, v) / s for k, v in weighted.items()}


def rerank_candidates(candidates: list[str], base_scores: list[float], future_scores: list[float], lam: float = 1.0) -> list[tuple[str, float]]:
    if not (len(candidates) == len(base_scores) == len(future_scores)):
        raise ValueError("candidates, base_scores, future_scores must have same length")
    scored = []
    for c, b, f in zip(candidates, base_scores, future_scores):
        scored.append((c, b + lam * f))
    return sorted(scored, key=lambda x: x[1], reverse=True)
