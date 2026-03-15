from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Iterable


@dataclass
class NgramModel:
    order: int = 3

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be >= 1")
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.totals: Counter[tuple[str, ...]] = Counter()

    def fit(self, tokens: Iterable[str]) -> None:
        seq = list(tokens)
        if len(seq) < self.order:
            return
        for i in range(self.order - 1, len(seq)):
            ctx = tuple(seq[i - self.order + 1:i])
            nxt = seq[i]
            self.counts[ctx][nxt] += 1
            self.totals[ctx] += 1

    def context_distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        if context in self.counts and self.totals[context] > 0:
            total = float(self.totals[context])
            return {tok: c / total for tok, c in self.counts[context].items()}
        # back off by dropping oldest token
        if len(context) > 0:
            return self.context_distribution(context[1:])
        # global fallback
        merged = Counter()
        for c in self.counts.values():
            merged.update(c)
        total = float(sum(merged.values()) or 1)
        return {tok: c / total for tok, c in merged.items()} if merged else {".": 1.0}

    def sample_next(self, context: tuple[str, ...], rng: random.Random | None = None) -> str:
        rng = rng or random.Random()
        dist = self.context_distribution(context)
        items = list(dist.items())
        r = rng.random()
        acc = 0.0
        for tok, p in items:
            acc += p
            if r <= acc:
                return tok
        return items[-1][0]

    def generate(self, seed_tokens: list[str], length: int, rng: random.Random | None = None) -> list[str]:
        rng = rng or random.Random()
        out = list(seed_tokens)
        while len(out) < length:
            ctx_len = self.order - 1
            ctx = tuple(out[-ctx_len:]) if ctx_len > 0 else tuple()
            out.append(self.sample_next(ctx, rng=rng))
        return out

    def support_size(self, context: tuple[str, ...]) -> int:
        return len(self.context_distribution(context))


def support_dropout_rate(source: NgramModel, sampled: NgramModel) -> float:
    source_contexts = set(source.counts.keys())
    if not source_contexts:
        return 0.0
    missing = sum(1 for c in source_contexts if c not in sampled.counts)
    return missing / float(len(source_contexts))
