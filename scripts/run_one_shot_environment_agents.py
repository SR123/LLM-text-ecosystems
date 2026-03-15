#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.metrics import distinct_n  # noqa: E402
from drift_selection.ngram import NgramModel  # noqa: E402
from drift_selection.utils import utc_now_iso  # noqa: E402


def _build_dist_fn(model: NgramModel):
    order = model.order
    cache: dict[tuple[str, ...], dict[str, float]] = {}

    merged: dict[str, int] = {}
    for ctr in model.counts.values():
        for tok, c in ctr.items():
            merged[tok] = merged.get(tok, 0) + int(c)
    total_merged = float(sum(merged.values()) or 1.0)
    global_dist = {tok: float(c) / total_merged for tok, c in merged.items()} if merged else {".": 1.0}

    def dist(context: tuple[str, ...]) -> dict[str, float]:
        context = tuple(context[-(order - 1):]) if order > 1 else tuple()
        if context in cache:
            return cache[context]
        c = context
        while True:
            if c in model.counts and model.totals[c] > 0:
                total = float(model.totals[c])
                out = {tok: float(cnt) / total for tok, cnt in model.counts[c].items()}
                cache[context] = out
                return out
            if len(c) == 0:
                cache[context] = global_dist
                return global_dist
            c = c[1:]

    return dist


def _load_split_tokens(root: Path, split: str) -> list[str]:
    path = root / "GitHub" / "corpora" / "splits" / "conan_doyle" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.split()


def _top_k(dist: dict[str, float], k: int) -> list[tuple[str, float]]:
    return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[: max(1, k)]


def _sample_from_dist(dist: dict[str, float], rng: random.Random) -> str:
    items = list(dist.items())
    r = rng.random()
    acc = 0.0
    for tok, p in items:
        acc += float(p)
        if r <= acc:
            return tok
    return items[-1][0]


def _repeat_count(seq: list[str], n: int = 4) -> int:
    if len(seq) < n:
        return 0
    seen = set()
    rep = 0
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i : i + n])
        if g in seen:
            rep += 1
        else:
            seen.add(g)
    return rep


def _selected_next_token(
    dist_fn,
    order: int,
    prefix: list[str],
    top_k: int = 8,
    horizon: int = 4,
    repetition_penalty: float = 0.8,
) -> str:
    ctx_len = order - 1
    context = tuple(prefix[-ctx_len:]) if ctx_len > 0 else tuple()
    dist0 = dist_fn(context)
    cands = _top_k(dist0, top_k)

    best_tok = cands[0][0]
    best_score = -1e18

    for tok, p0 in cands:
        seq = list(prefix) + [tok]
        logprob_sum = math.log(max(float(p0), 1e-12))
        for _ in range(max(0, horizon - 1)):
            c = tuple(seq[-ctx_len:]) if ctx_len > 0 else tuple()
            dist = dist_fn(c)
            nxt, p = max(dist.items(), key=lambda kv: kv[1])
            seq.append(nxt)
            logprob_sum += math.log(max(float(p), 1e-12))

        score = logprob_sum - repetition_penalty * float(_repeat_count(seq[-32:], n=4))
        if score > best_score:
            best_score = score
            best_tok = tok

    return best_tok


def _generate_environment(
    mode: str,
    generator: NgramModel,
    dist_fn,
    prompt_bank: list[list[str]],
    total_tokens: int,
    max_new_per_prompt: int,
    rng: random.Random,
) -> list[str]:
    if mode == "original":
        raise ValueError("original environment should be sampled directly from corpus")

    env: list[str] = []
    pidx = 0
    while len(env) < total_tokens:
        prompt = list(prompt_bank[pidx % len(prompt_bank)])
        out = list(prompt)
        for _ in range(max_new_per_prompt):
            ctx_len = generator.order - 1
            context = tuple(out[-ctx_len:]) if ctx_len > 0 else tuple()
            if mode == "neutral":
                out.append(_sample_from_dist(dist_fn(context), rng))
            elif mode == "greedy":
                dist = dist_fn(context)
                out.append(max(dist.items(), key=lambda kv: kv[1])[0])
            elif mode == "selected":
                out.append(_selected_next_token(dist_fn, generator.order, out, top_k=8, horizon=4, repetition_penalty=0.8))
            else:
                raise ValueError(mode)
        env.extend(out[len(prompt):])
        pidx += 1
        if pidx % len(prompt_bank) == 0:
            rng.shuffle(prompt_bank)
    return env[:total_tokens]


def _build_prompt_bank(tokens: list[str], prompt_len: int, n_prompts: int, seed: int) -> list[list[str]]:
    if len(tokens) <= prompt_len:
        return [tokens[:prompt_len]] * n_prompts
    rng = random.Random(seed)
    max_start = len(tokens) - prompt_len
    out: list[list[str]] = []
    for _ in range(n_prompts):
        s = rng.randint(0, max_start)
        out.append(tokens[s : s + prompt_len])
    return out


def _train_one_shot_agent(tokens: list[str], order: int, budget_transitions: int) -> NgramModel:
    # one-shot = single fit pass on fixed finite budget, no iterative updates
    usable = tokens[: max(order, budget_transitions + order)]
    agent = NgramModel(order=order)
    agent.fit(usable)
    return agent


def _eval_agent(
    generator: NgramModel,
    agent: NgramModel,
    eval_prompts: list[list[str]],
    continuation_len: int,
    rng: random.Random,
) -> dict[str, float]:
    gen_dist = _build_dist_fn(generator)
    agent_dist_fn = _build_dist_fn(agent)
    probs: list[float] = []
    logps: list[float] = []
    matches: list[int] = []
    continuations: list[list[str]] = []

    for prompt in eval_prompts:
        selected_tok = _selected_next_token(gen_dist, generator.order, prompt, top_k=8, horizon=4, repetition_penalty=0.8)
        ctx_len_agent = agent.order - 1
        context_agent = tuple(prompt[-ctx_len_agent:]) if ctx_len_agent > 0 else tuple()
        p = float(agent_dist_fn(context_agent).get(selected_tok, 0.0))
        probs.append(p)
        logps.append(math.log(max(p, 1e-12)))

        ctx_len = agent.order - 1
        context = tuple(prompt[-ctx_len:]) if ctx_len > 0 else tuple()
        agent_dist = agent_dist_fn(context)
        top_tok = max(agent_dist.items(), key=lambda kv: kv[1])[0]
        matches.append(int(top_tok == selected_tok))

        out = list(prompt)
        for _ in range(continuation_len):
            c = tuple(out[-ctx_len:]) if ctx_len > 0 else tuple()
            out.append(_sample_from_dist(agent_dist_fn(c), rng))
        continuations.append(out[len(prompt):])

    return {
        "mean_selected_probability": float(mean(probs)) if probs else float("nan"),
        "mean_selected_logprob": float(mean(logps)) if logps else float("nan"),
        "top1_match_rate": float(mean(matches)) if matches else float("nan"),
        "distinct_2": float(distinct_n(continuations, n=2)) if continuations else float("nan"),
        "repetition_4": float(1.0 - distinct_n(continuations, n=4)) if continuations else float("nan"),
    }


def _token_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    probs = [v / len(tokens) for v in counts.values()]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot agents across generated environments")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=20260303)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--env-tokens", type=int, default=100000)
    parser.add_argument("--one-shot-budget", type=int, default=20000)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--generation-prompts", type=int, default=256)
    parser.add_argument("--eval-prompts", type=int, default=256)
    parser.add_argument("--continuation-len", type=int, default=48)
    parser.add_argument("--max-new-per-prompt", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rng = random.Random(args.seed)

    train_tokens = _load_split_tokens(root, "train")
    test_tokens = _load_split_tokens(root, "test")

    generator = NgramModel(order=args.order)
    generator.fit(train_tokens)
    generator_dist = _build_dist_fn(generator)

    gen_prompts = _build_prompt_bank(train_tokens, prompt_len=args.prompt_len, n_prompts=args.generation_prompts, seed=args.seed)
    eval_prompts = _build_prompt_bank(test_tokens, prompt_len=args.prompt_len, n_prompts=args.eval_prompts, seed=args.seed + 1)

    # environments
    original_start = rng.randint(0, max(0, len(train_tokens) - args.env_tokens - 1)) if len(train_tokens) > args.env_tokens else 0
    env_original = train_tokens[original_start : original_start + args.env_tokens]
    env_neutral = _generate_environment("neutral", generator, generator_dist, [list(p) for p in gen_prompts], args.env_tokens, args.max_new_per_prompt, rng)
    env_greedy = _generate_environment("greedy", generator, generator_dist, [list(p) for p in gen_prompts], args.env_tokens, args.max_new_per_prompt, rng)
    env_selected = _generate_environment("selected", generator, generator_dist, [list(p) for p in gen_prompts], args.env_tokens, args.max_new_per_prompt, rng)

    envs = {
        "original": env_original,
        "neutral": env_neutral,
        "greedy": env_greedy,
        "selected": env_selected,
    }

    out_root = root / "GitHub" / "data" / "outputs" / "one_shot_env_agents"
    out_dir = out_root / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # environment summaries
    env_rows = []
    for name, toks in envs.items():
        env_rows.append(
            {
                "environment": name,
                "token_count": len(toks),
                "distinct_2": float(distinct_n([toks], n=2)),
                "repetition_4": float(1.0 - distinct_n([toks], n=4)),
                "token_entropy": _token_entropy(toks),
            }
        )
        (out_dir / f"environment_{name}.txt").write_text(" ".join(toks), encoding="utf-8")
    pd.DataFrame(env_rows).to_csv(out_dir / "environment_summary.csv", index=False)

    # one-shot agents + evaluation
    agent_rows = []
    sample_lines = []
    for name, toks in envs.items():
        agent = _train_one_shot_agent(toks, order=args.order, budget_transitions=args.one_shot_budget)
        metrics = _eval_agent(generator, agent, eval_prompts, continuation_len=args.continuation_len, rng=rng)
        row = {"agent_environment": name, **metrics}
        agent_rows.append(row)
        agent_dist_for_samples = _build_dist_fn(agent)

        sample_lines.append("=" * 80)
        sample_lines.append(f"AGENT: {name}")
        for i, prompt in enumerate(eval_prompts[:5]):
            out = list(prompt)
            for _ in range(40):
                ctx = tuple(out[-(agent.order - 1):]) if agent.order > 1 else tuple()
                out.append(_sample_from_dist(agent_dist_for_samples(ctx), rng))
            sample_lines.append(f"[prompt_{i}] {' '.join(prompt)}")
            sample_lines.append(f"[gen_{i}] {' '.join(out[len(prompt):])}")
            sample_lines.append("")

    metrics_df = pd.DataFrame(agent_rows).sort_values("mean_selected_probability", ascending=False)
    metrics_df.to_csv(out_dir / "one_shot_agent_metrics.csv", index=False)
    (out_dir / "one_shot_agent_samples.txt").write_text("\n".join(sample_lines), encoding="utf-8")

    manifest = {
        "created_at": utc_now_iso(),
        "config": {
            "seed": args.seed,
            "order": args.order,
            "env_tokens": args.env_tokens,
            "one_shot_budget": args.one_shot_budget,
            "prompt_len": args.prompt_len,
            "generation_prompts": args.generation_prompts,
            "eval_prompts": args.eval_prompts,
            "continuation_len": args.continuation_len,
            "max_new_per_prompt": args.max_new_per_prompt,
        },
        "outputs": {
            "environment_summary": str((out_dir / "environment_summary.csv").relative_to(root)),
            "agent_metrics": str((out_dir / "one_shot_agent_metrics.csv").relative_to(root)),
            "agent_samples": str((out_dir / "one_shot_agent_samples.txt").relative_to(root)),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote one-shot environment experiment to: {out_dir}")
    print(f"Top agent by selected-token probability: {metrics_df.iloc[0]['agent_environment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
