from __future__ import annotations

from typing import Sequence
import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    m = np.array(matrix, dtype=float)
    row_sums = m.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return m / row_sums


def survival_probabilities(P: np.ndarray, alive: Sequence[bool], horizon: int) -> np.ndarray:
    P = normalize_rows(P)
    alive_mask = np.array(alive, dtype=bool)
    n = P.shape[0]
    h = np.zeros((horizon + 1, n), dtype=float)
    h[0, alive_mask] = 1.0
    for t in range(1, horizon + 1):
        h[t] = P @ h[t - 1]
        h[t, ~alive_mask] = 0.0
    return h


def doob_h_transform(P: np.ndarray, h: np.ndarray, alive: Sequence[bool]) -> np.ndarray:
    P = normalize_rows(P)
    alive_mask = np.array(alive, dtype=bool)
    out = np.zeros_like(P)
    for i in range(P.shape[0]):
        if not alive_mask[i] or h[i] <= 0:
            continue
        for j in range(P.shape[1]):
            if h[j] > 0:
                out[i, j] = P[i, j] * h[j] / h[i]
    return normalize_rows(out)
