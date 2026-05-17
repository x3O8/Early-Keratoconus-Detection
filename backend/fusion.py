from __future__ import annotations

import numpy as np


def align_probability_vectors(p: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size == n:
        out = p
    elif p.size < n:
        out = np.full(n, 1e-8, dtype=np.float64)
        out[: p.size] = p
    else:
        out = p[:n]
    out = np.clip(out, 1e-8, None)
    return out / out.sum()


def weighted_fusion(p1: np.ndarray, p2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    w1 = float(max(w1, 0.0))
    w2 = float(max(w2, 0.0))
    s = w1 + w2
    if s <= 0:
        w1, w2 = 0.5, 0.5
        s = 1.0
    w1, w2 = w1 / s, w2 / s
    n = max(p1.size, p2.size)
    a1 = align_probability_vectors(p1, n)
    a2 = align_probability_vectors(p2, n)
    fused = w1 * a1 + w2 * a2
    return fused / fused.sum()


def entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def model_agreement(p1: np.ndarray, p2: np.ndarray) -> float:
    n = max(p1.size, p2.size)
    a1 = align_probability_vectors(p1, n)
    a2 = align_probability_vectors(p2, n)
    return float(1.0 - 0.5 * np.abs(a1 - a2).sum())
