from __future__ import annotations

import random


def normalize(probs: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in probs.values())
    if total <= 0:
        n = len(probs) or 1
        return {k: 1.0 / n for k in probs} if probs else {}
    return {k: max(v, 0.0) / total for k, v in probs.items()}


def viability_selection(probs: dict[str, float], viability: dict[str, float]) -> dict[str, float]:
    weighted = {k: probs.get(k, 0.0) * max(viability.get(k, 0.0), 0.0) for k in probs}
    return normalize(weighted)


def anti_viability_selection(probs: dict[str, float], viability: dict[str, float]) -> dict[str, float]:
    weighted = {k: probs.get(k, 0.0) * max(1.0 - viability.get(k, 0.0), 0.0) for k in probs}
    return normalize(weighted)


def random_filter_selection(probs: dict[str, float], keep_prob: float, rng: random.Random | None = None) -> dict[str, float]:
    rng = rng or random.Random()
    kept = {k: v if rng.random() < keep_prob else 0.0 for k, v in probs.items()}
    if sum(kept.values()) == 0 and probs:
        # keep one token to avoid empty support
        k = rng.choice(list(probs.keys()))
        kept[k] = probs[k]
    return normalize(kept)


def frequency_biased_selection(probs: dict[str, float], freq: dict[str, float], beta: float = 1.0) -> dict[str, float]:
    weighted = {k: probs.get(k, 0.0) * (freq.get(k, 0.0) ** beta) for k in probs}
    return normalize(weighted)
