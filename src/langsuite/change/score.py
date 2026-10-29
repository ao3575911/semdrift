"""Calibrated, controlled semantic change between two slices.

Method:
1. Each slice has R seed replicates of SGNS vectors (`vec train`).
2. Replicate r of A is aligned to replicate r of B by orthogonal Procrustes on the shared
   vocabulary (Schönemann 1966; Hamilton, Leskovec & Jurafsky 2016).
3. Raw change is the cosine distance after alignment, averaged over replicates.
4. Control: the documents of A and B are pooled and split at random into two pseudo-slices of
   the same sizes; the same pipeline is run on them (shuffled-slice null; Dubossarsky,
   Weinshall & Grossman 2017). This measures how much "change" sampling, topic mix and seed
   noise produce on their own.
5. Each word is tested against its *own* null: net change = raw - mean of that word's null
   runs. The spread uses the pooled within-word null variance of words in the same
   log-frequency bin, with a predictive t test (scale s*sqrt(1 + 1/n)). q is
   Benjamini-Hochberg over all tested words. Distances are first divided by their run's
   median word distance, so run-level shifts shared by every word (alignment quality, slice
   size) don't count as change: the test is for word-specific change.
   (Pooling raw null distances across words was rejected: words differ in baseline
   noise, which made the test anti-conservative.)
6. A second, alignment-free measure: Jaccard overlap of top-10 neighbours in A and B.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import t as student_t

from langsuite.corpus.workspace import Workspace, atomic_write_bytes, hash_bytes
from langsuite.errors import InputError, UsageError
from langsuite.geom.space import (
    F32,
    Space,
    TrainParams,
    knn_rows,
    load_replicates,
    store_dir,
    train_space,
)

F64 = npt.NDArray[np.float64]


def procrustes(src: F32, dst: F32) -> F32:
    """R = argmin ||src R - dst||_F with R orthogonal: R = U V^T, U S V^T = svd(src^T dst)."""
    u, _, vt = np.linalg.svd(src.T.astype(np.float64) @ dst.astype(np.float64))
    r: F32 = (u @ vt).astype(np.float32)
    return r


def _shared(a: Space, b: Space, min_count: int) -> list[str]:
    return [
        w
        for w in a.vocab
        if w in b.index and a.counts[a.index[w]] >= min_count and b.counts[b.index[w]] >= min_count
    ]


def aligned_distances(a: Space, b: Space, words: Sequence[str]) -> F64:
    ia = [a.index[w] for w in words]
    ib = [b.index[w] for w in words]
    ma, mb = a.vectors[ia], b.vectors[ib]
    r = procrustes(ma, mb)
    ra = ma @ r
    ra /= np.linalg.norm(ra, axis=1, keepdims=True)
    d: F64 = 1.0 - (ra * mb).sum(axis=1).astype(np.float64)
    return d


def nn_overlap(a: Space, b: Space, word: str, k: int = 10) -> float:
    na = {a.vocab[i] for i in knn_rows(a.vec(word), a.vectors, k, [a.index[word]])[0]}
    nb = {b.vocab[i] for i in knn_rows(b.vec(word), b.vectors, k, [b.index[word]])[0]}
    return len(na & nb) / max(1, len(na | nb))


def bh(p: F64) -> F64:
    """Benjamini-Hochberg q-values (Benjamini & Hochberg 1995)."""
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


@dataclass
class WordChange:
    word: str
    freq_a: int
    freq_b: int
    change: float  # relative distance minus this word's null mean (units: median word distance)
    raw: float  # absolute aligned cosine distance, mean over replicates
    null_median: float  # this word's null mean, relative units
    seed_sd: float
    z: float
    p: float
    p_empirical: float
    q: float
    nn_overlap: float
    stable: bool


def _doc_sentences(ws: Workspace, doc_ids: Sequence[str]) -> list[list[str]]:
    df = (
        ws.tokens()
        .filter(pl.col("doc_id").is_in(list(doc_ids)) & pl.col("norm").str.contains(r"\w"))
        .group_by("doc_id", "sent", maintain_order=True)
        .agg(pl.col("norm").sort_by("i"))
        .sort("doc_id", "sent")
        .collect()
    )
    return [list(s) for s in df["norm"].to_list()]


def _manifest(ws: Workspace, slice_name: str) -> dict[str, object]:
    path = store_dir(ws, slice_name) / "manifest.json"
    if not path.exists():
        raise InputError(
            f"no embeddings for slice '{slice_name}'",
            f"train them with: langsuite vec train --slice {slice_name}",
        )
    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return data


def null_spaces(
    ws: Workspace, a: str, b: str, runs: int, params: TrainParams, seeds: Sequence[int]
) -> list[tuple[list[Space], list[Space]]]:
    """Shuffled-slice null: pool A and B docs, split at random (sizes kept), train replicates."""
    ids_a = ws.slice_doc_ids(ws.slices()[a])
    ids_b = ws.slice_doc_ids(ws.slices()[b])
    if len(ids_a) < 2 or len(ids_b) < 2:
        raise InputError(
            "each slice needs at least 2 documents for the shuffled null",
            "split long texts into several documents",
        )
    pool = sorted(set(ids_a) | set(ids_b))
    key = hash_bytes(
        ws.corpus_hash().encode(), json.dumps([a, b, asdict(params), list(seeds)]).encode()
    )
    cdir = ws.state / "null" / key[:20]
    out: list[tuple[list[Space], list[Space]]] = []
    for run in range(runs):
        rng = np.random.default_rng(10_000 + run)
        perm = list(rng.permutation(pool))
        pa, pb = perm[: len(ids_a)], perm[len(ids_a) : len(ids_a) + len(ids_b)]
        pair: tuple[list[Space], list[Space]] = ([], [])
        for side, ids in ((0, pa), (1, pb)):
            sents: list[list[str]] | None = None
            for s in seeds:
                f = cdir / f"run{run}_side{side}_seed{s}.npz"
                if f.exists():
                    pair[side].append(Space.load(f))
                    continue
                if sents is None:
                    sents = _doc_sentences(ws, [str(x) for x in ids])
                sp = train_space(sents, params, s)
                sp.save(f)
                pair[side].append(sp)
        out.append(pair)
    atomic_write_bytes(cdir / "meta.json", json.dumps({"a": a, "b": b, "runs": runs}).encode())
    return out


def score(
    ws: Workspace, a: str, b: str, null_runs: int = 3, min_count: int = 10, progress: bool = False
) -> tuple[list[WordChange], dict[str, object]]:
    if a == b:
        raise UsageError("pick two different slices")
    if null_runs < 1:
        raise UsageError(
            "--null-runs must be at least 1 (the null is what makes scores meaningful)"
        )
    for s in (a, b):
        if s not in ws.slices():
            raise UsageError(
                f"unknown slice '{s}'", f"defined: {', '.join(sorted(ws.slices())) or '(none)'}"
            )
    ma, mb = _manifest(ws, a), _manifest(ws, b)
    if ma["params"] != mb["params"] or ma["seeds"] != mb["seeds"]:
        raise UsageError(
            "the two slices were trained with different settings or seeds",
            "retrain both with the same vec train options",
        )
    sa, sb = load_replicates(ws, a), load_replicates(ws, b)
    words = _shared(sa[0], sb[0], min_count)
    for x, y in zip(sa[1:], sb[1:], strict=True):
        keep = set(_shared(x, y, min_count))
        words = [w for w in words if w in keep]
    if len(words) < 20:
        raise InputError(
            f"only {len(words)} words reach --min-count {min_count} in both slices",
            "lower --min-count or use larger slices (alignment needs a shared vocabulary)",
        )
    raw = np.array([aligned_distances(x, y, words) for x, y in zip(sa, sb, strict=True)])
    params_raw = ma["params"]
    assert isinstance(params_raw, dict)
    params = TrainParams(**params_raw)
    seeds_raw = ma["seeds"]
    assert isinstance(seeds_raw, list)
    seeds = [int(s) for s in seeds_raw]
    if null_runs < 2:
        raise UsageError("--null-runs must be at least 2 (the test needs the null's spread)")
    if progress:
        from langsuite.cli.output import info

        info(f"shuffled-slice null: {null_runs} run(s) x {len(seeds)} seed(s) x 2 sides (cached)")
    nulls = null_spaces(ws, a, b, null_runs, params, seeds)
    freq = np.array(
        [min(sa[0].counts[sa[0].index[w]], sb[0].counts[sb[0].index[w]]) for w in words], np.float64
    )
    pos = {w: i for i, w in enumerate(words)}
    null = np.full((len(nulls), len(words)), np.nan)
    for run, (na, nb) in enumerate(nulls):
        common = [w for w in words if all(w in s.index for s in (*na, *nb))]
        if len(common) < 20:
            continue
        cols = [pos[w] for w in common]
        null[run, cols] = np.mean(
            [aligned_distances(x, y, common) for x, y in zip(na, nb, strict=True)], axis=0
        )
    n_w = (~np.isnan(null)).sum(axis=0)
    if (n_w >= 2).sum() < 20:
        raise InputError("the shuffled null had too small a shared vocabulary", "lower --min-count")
    # Remove run-level scale (alignment quality, slice size) shared by all words in a run:
    # express every distance relative to that run's median word distance.
    raw_abs = raw.mean(axis=0)
    raw = raw / np.median(raw, axis=1, keepdims=True)
    null = null / np.nanmedian(null, axis=1, keepdims=True)
    obs = raw.mean(axis=0)
    sd = raw.std(axis=0)
    mean_w = np.nanmean(np.where(n_w >= 1, null, 0.0), axis=0)
    # Pooled within-word residual sd of null run-means, per log-frequency bin.
    n_bins = int(max(1, min(10, len(words) // 30)))
    edges = np.quantile(np.log(freq), np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_w = np.clip(np.searchsorted(edges, np.log(freq), side="right") - 1, 0, n_bins - 1)
    resid = null - mean_w
    sig = np.ones(n_bins)
    dof = np.ones(n_bins)
    for k in range(n_bins):
        m = (bin_w == k) & (n_w >= 2)
        if m.sum() == 0:
            m = n_w >= 2
        ss = np.nansum(resid[:, m] ** 2)
        df_k = float((n_w[m] - 1).sum())
        sig[k] = max(float(np.sqrt(ss / max(df_k, 1.0))), 1e-9)
        dof[k] = max(df_k, 1.0)
    scale = sig[bin_w] * np.sqrt(1.0 + 1.0 / np.maximum(n_w, 1))
    z = np.where(n_w >= 1, (obs - mean_w) / scale, 0.0)
    p = np.where(n_w >= 1, student_t.sf(z, dof[bin_w]), 1.0)
    p_emp = (1 + np.nansum(null >= obs, axis=0)) / (1 + n_w)
    med = mean_w
    q = bh(p)
    stab_a = _stability(sa, words)
    stab_b = _stability(sb, words)
    rows = [
        WordChange(
            w,
            int(sa[0].counts[sa[0].index[w]]),
            int(sb[0].counts[sb[0].index[w]]),
            float(obs[i] - med[i]),
            float(raw_abs[i]),
            float(med[i]),
            float(sd[i]),
            float(z[i]),
            float(p[i]),
            float(p_emp[i]),
            float(q[i]),
            nn_overlap(sa[0], sb[0], w),
            bool(min(stab_a[i], stab_b[i]) >= 0.3) if len(sa) > 1 else False,
        )
        for i, w in enumerate(words)
    ]
    rows.sort(key=lambda r: (r.q, -r.change))
    meta: dict[str, object] = {
        "a": a,
        "b": b,
        "tested_words": len(words),
        "replicates": len(sa),
        "null_runs": null_runs,
        "null_samples": int(n_w.sum()),
        "frequency_bins": n_bins,
        "min_count": min_count,
        "significant_q05": int((q < 0.05).sum()),
    }
    return rows, meta


def _stability(spaces: Sequence[Space], words: Sequence[str], k: int = 10) -> F64:
    if len(spaces) < 2:
        return np.full(len(words), np.nan)
    out = np.zeros(len(words))
    for i, w in enumerate(words):
        sets = [
            {s.vocab[j] for j in knn_rows(s.vec(w), s.vectors, k, [s.index[w]])[0]} for s in spaces
        ]
        pairs = [(x, y) for n, x in enumerate(sets) for y in sets[n + 1 :]]
        out[i] = np.mean([len(x & y) / max(1, len(x | y)) for x, y in pairs])
    return out


def drift(ws: Workspace, word: str, slices: Sequence[str], k: int = 8) -> list[dict[str, object]]:
    """Neighbour timeline: top-k neighbours per slice and distance from the first slice."""
    if len(slices) < 2:
        raise UsageError("give at least two slices, e.g. --slices early,mid,late")
    spaces = [load_replicates(ws, s)[0] for s in slices]
    for s, sp in zip(slices, spaces, strict=True):
        if word not in sp.index:
            raise InputError(
                f"'{word}' is not in the vocabulary of slice '{s}'", "lower --min-count and retrain"
            )
    ref = spaces[0]
    out: list[dict[str, object]] = []
    for s, sp in zip(slices, spaces, strict=True):
        shared = [w for w in ref.vocab if w in sp.index]
        dist = 0.0 if sp is ref else float(aligned_distances(ref, sp, shared)[shared.index(word)])
        idx, _ = knn_rows(sp.vec(word), sp.vectors, k, [sp.index[word]])
        out.append(
            {
                "slice": s,
                "distance_from_first": dist,
                "neighbours": ", ".join(sp.vocab[i] for i in idx),
            }
        )
    return out
