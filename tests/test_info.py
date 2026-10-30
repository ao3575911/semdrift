from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from semdrift.metrics import info

codes_st = st.lists(st.integers(0, 12), min_size=1, max_size=200).map(
    lambda xs: np.array(xs, np.int64)
)


def naive_mattr(codes: np.ndarray, w: int) -> float:
    if codes.size <= w:
        return len(set(codes.tolist())) / codes.size
    vals = [len(set(codes[i : i + w].tolist())) / w for i in range(codes.size - w + 1)]
    return sum(vals) / len(vals)


@given(codes_st, st.integers(1, 40))
def test_mattr_matches_naive(codes: np.ndarray, w: int) -> None:
    assert math.isclose(info.mattr(codes, w), naive_mattr(codes, w), rel_tol=1e-12)


@given(codes_st)
def test_mattr_equals_ttr_when_window_covers(codes: np.ndarray) -> None:
    assert info.mattr(codes, codes.size) == info.ttr(codes)


def test_entropy_golden() -> None:
    assert info.entropy_from_counts(np.array([1, 1, 1, 1])) == 2.0
    assert math.isclose(info.entropy_from_counts(np.array([3, 1])), 0.8112781244591328)
    joint, cond = info.bigram_entropies([np.array([0, 1, 0, 1, 0], np.int64)])
    # pairs: (0,1),(1,0),(0,1),(1,0) -> H=1; H(X)=1 over x=[0,1,0,1] -> H(Y|X)=0
    assert (joint, cond) == (1.0, 0.0)


count_pair = (
    st.integers(2, 30)
    .flatmap(
        lambda n: st.tuples(
            st.lists(st.integers(0, 50), min_size=n, max_size=n),
            st.lists(st.integers(0, 50), min_size=n, max_size=n),
        )
    )
    .filter(lambda t: sum(t[0]) > 0 and sum(t[1]) > 0)
)


@given(count_pair)
def test_js_symmetric_bounded(t: tuple[list[int], list[int]]) -> None:
    p, q = np.array(t[0], float), np.array(t[1], float)
    a, b = info.js_divergence(p, q), info.js_divergence(q, p)
    assert math.isclose(a, b, abs_tol=1e-12)
    assert 0.0 <= a <= 1.0
    assert info.kl_divergence(p, q) >= -1e-12


def test_js_disjoint_is_one() -> None:
    assert info.js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0


def test_zipf_recovers_exponent() -> None:
    rng = np.random.default_rng(1)
    ranks = np.arange(1, 2001)
    p = ranks**-1.1
    p /= p.sum()
    codes = rng.choice(2000, size=200_000, p=p).astype(np.int64)
    assert abs(info.zipf_exponent(codes) - 1.1) < 0.05


def test_heaps_on_all_new_types() -> None:
    k, beta = info.heaps_fit(np.arange(1000, dtype=np.int64))
    assert math.isclose(beta, 1.0, abs_tol=1e-9)
    assert math.isclose(k, 1.0, rel_tol=1e-6)


@settings(max_examples=25)
@given(st.lists(codes_st, min_size=2, max_size=8), st.integers(0, 10_000))
def test_bootstrap_ci_brackets_and_is_deterministic(docs: list[np.ndarray], seed: int) -> None:
    def stat(ds: object) -> float:
        return info.unigram_entropy(np.concatenate(list(ds)))  # type: ignore[call-overload]

    a = info.bootstrap(docs, stat, 50, seed)
    b = info.bootstrap(docs, stat, 50, seed)
    assert a == b
    assert a.ci_low <= a.ci_high


def test_jackknife_brackets_value() -> None:
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 5000, 200).astype(np.int64) for _ in range(30)]

    def stat(ds: object) -> float:
        return info.ttr(np.concatenate(list(ds)))  # type: ignore[call-overload]

    e = info.group_jackknife(docs, stat, 10, 0)
    assert e.ci_low < e.value < e.ci_high
    assert e.resamples == 10
