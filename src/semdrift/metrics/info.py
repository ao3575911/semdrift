"""Information-theoretic and lexical-diversity measures. Pure NumPy; all logs base 2."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


def entropy_from_counts(counts: npt.NDArray[np.integer] | FloatArray) -> float:
    """Shannon entropy H = -sum p log2 p (Shannon 1948)."""
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def unigram_entropy(codes: IntArray) -> float:
    return entropy_from_counts(np.bincount(codes)) if codes.size else 0.0


def _pairs(docs: Sequence[IntArray]) -> tuple[IntArray, IntArray]:
    xs = [d[:-1] for d in docs if d.size > 1]
    ys = [d[1:] for d in docs if d.size > 1]
    if not xs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(xs), np.concatenate(ys)


def bigram_entropies(docs: Sequence[IntArray]) -> tuple[float, float]:
    """(joint H(X,Y), conditional H(Y|X) = H(X,Y) - H(X)) over adjacent pairs within documents."""
    x, y = _pairs(docs)
    if x.size == 0:
        return 0.0, 0.0
    vocab = int(max(x.max(), y.max())) + 1
    joint = entropy_from_counts(np.unique(x * vocab + y, return_counts=True)[1])
    hx = entropy_from_counts(np.bincount(x))
    return joint, joint - hx


def ttr(codes: IntArray) -> float:
    return float(np.unique(codes).size / codes.size) if codes.size else 0.0


def _prev_next(codes: IntArray) -> tuple[IntArray, IntArray]:
    n = codes.size
    order = np.lexsort((np.arange(n), codes))
    same = np.r_[False, codes[order][1:] == codes[order][:-1]]
    prev = np.full(n, -1, np.int64)
    nxt = np.full(n, n, np.int64)
    idx = np.nonzero(same)[0]
    prev[order[idx]] = order[idx - 1]
    nxt[order[idx - 1]] = order[idx]
    return prev, nxt


def mattr(codes: IntArray, window: int = 500) -> float:
    """Moving-average TTR (Covington & McFadden 2010). Falls back to TTR when N <= window.

    Uses an O(N) recurrence on previous/next occurrence indices:
    d(i+1) = d(i) - 1 + [prev(i+W) <= i] + [next(i) < i+W].
    """
    n = codes.size
    if n == 0:
        return 0.0
    if window <= 0:
        raise ValueError("window must be positive")
    if n <= window:
        return ttr(codes)
    prev, nxt = _prev_next(codes)
    d0 = int(np.unique(codes[:window]).size)
    i = np.arange(n - window)
    steps = -1 + (prev[i + window] <= i).astype(np.int64) + (nxt[i] < i + window).astype(np.int64)
    d = np.concatenate(([d0], d0 + np.cumsum(steps)))
    return float(d.mean() / window)


def zipf_exponent(codes: IntArray) -> float:
    """MLE of s in f(r) ∝ r^-s over ranks 1..V (cf. Clauset, Shalizi & Newman 2009)."""
    freqs = np.sort(np.bincount(codes)[np.bincount(codes) > 0])[::-1].astype(np.float64)
    if freqs.size < 2:
        return float("nan")
    log_r = np.log(np.arange(1, freqs.size + 1, dtype=np.float64))
    n = freqs.sum()
    a = float((freqs * log_r).sum())

    def nll(s: float) -> float:
        log_h = float(np.logaddexp.reduce(-s * log_r))
        return s * a + n * log_h

    res = minimize_scalar(nll, bounds=(0.01, 6.0), method="bounded", options={"xatol": 1e-9})
    return float(res.x)


def heaps_fit(codes: IntArray, points: int = 50) -> tuple[float, float]:
    """Fit V(N) = K N^beta (Heaps 1978) by least squares in log space. Returns (K, beta)."""
    n = codes.size
    if n < 10:
        return float("nan"), float("nan")
    first = np.zeros(n, bool)
    _, idx = np.unique(codes, return_index=True)
    first[idx] = True
    growth = np.cumsum(first)
    ns = np.unique(np.geomspace(10, n, points).astype(np.int64))
    beta, log_k = np.polyfit(np.log(ns), np.log(growth[ns - 1]), 1)
    return float(np.exp(log_k)), float(beta)


def _dists(
    p_counts: FloatArray, q_counts: FloatArray, alpha: float
) -> tuple[FloatArray, FloatArray]:
    p = p_counts + alpha
    q = q_counts + alpha
    return p / p.sum(), q / q.sum()


def kl_divergence(p_counts: FloatArray, q_counts: FloatArray, alpha: float = 0.5) -> float:
    """D(P||Q) in bits, add-alpha smoothed on the shared vocabulary (Kullback & Leibler 1951)."""
    p, q = _dists(p_counts, q_counts, alpha)
    return float((p * np.log2(p / q)).sum())


def js_divergence(p_counts: FloatArray, q_counts: FloatArray) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1] (Lin 1991). No smoothing needed."""
    p = p_counts / p_counts.sum()
    q = q_counts / q_counts.sum()
    m = 0.5 * (p + q)

    def part(a: FloatArray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / m[mask])).sum())

    return max(0.0, min(1.0, 0.5 * part(p) + 0.5 * part(q)))


@dataclass(frozen=True)
class Estimate:
    value: float
    ci_low: float
    ci_high: float
    resamples: int


def bootstrap(
    docs: Sequence[IntArray],
    stat: Callable[[Sequence[IntArray]], float],
    resamples: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> Estimate:
    """Percentile CI from resampling whole documents with replacement (Efron & Tibshirani 1993)."""
    value = stat(docs)
    if resamples <= 0 or len(docs) < 2:
        return Estimate(value, float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    vals = np.empty(resamples)
    for b in range(resamples):
        pick = rng.integers(0, len(docs), len(docs))
        vals[b] = stat([docs[i] for i in pick])
    lo, hi = np.nanquantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return Estimate(value, float(lo), float(hi), resamples)


def group_jackknife(
    docs: Sequence[IntArray],
    stat: Callable[[Sequence[IntArray]], float],
    groups: int = 20,
    seed: int = 0,
) -> Estimate:
    """Delete-a-group jackknife 95% normal interval (Quenouille 1956; Efron & Stein 1981).

    Preferred over the bootstrap for type-based measures (TTR, H1, Zipf, Heaps): resampling
    with replacement duplicates documents and biases them downward.
    """
    value = stat(docs)
    g = min(groups, len(docs))
    if g < 2:
        return Estimate(value, float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    label = rng.permutation(len(docs)) % g
    thetas = np.array(
        [stat([d for d, lab in zip(docs, label, strict=True) if lab != k]) for k in range(g)]
    )
    se = float(np.sqrt((g - 1) / g * ((thetas - thetas.mean()) ** 2).sum()))
    return Estimate(value, value - 1.959964 * se, value + 1.959964 * se, g)
