"""Metrics for repeated, noisy agent runs."""
from __future__ import annotations

import math

import numpy as np


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: well-behaved for small n and rates near 0 or 1."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k samples passes), unbiased from n samples with c passes (Chen et al., 2021)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(all k samples pass): the *reliability* view. An agent users can trust needs this high, not pass@k."""
    if k > n:
        raise ValueError("k must be <= n")
    return math.comb(c, k) / math.comb(n, k)


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def paired_bootstrap(base: np.ndarray, cand: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (cand - base) per case with a 95% bootstrap CI over cases."""
    d = np.asarray(cand, float) - np.asarray(base, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
