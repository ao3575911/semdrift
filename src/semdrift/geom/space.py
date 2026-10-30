"""Embedding spaces: training (gensim), storage, and geometry queries over seed replicates."""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import version as dist_version
from itertools import combinations
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import xxhash

from semdrift.corpus.workspace import Workspace, atomic_write_bytes
from semdrift.errors import InputError, ResourceError, UsageError, missing_extra

F32 = npt.NDArray[np.float32]
ALGOS = ("w2v", "fasttext")


def _stable_hash(s: str) -> int:
    """Seed-stable replacement for Python's salted hash() in gensim init."""
    return xxhash.xxh64_intdigest(s.encode()) & 0x7FFFFFFF


@dataclass
class Space:
    vocab: list[str]
    vectors: F32  # L2-normalised rows
    counts: npt.NDArray[np.int64]
    seed: int

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def vec(self, word: str) -> F32:
        row: F32 = self.vectors[self.index[word]]
        return row

    def save(self, path: Path) -> None:
        import io

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            vocab=np.array(self.vocab),
            vectors=self.vectors,
            counts=self.counts,
            seed=self.seed,
        )
        atomic_write_bytes(path, buf.getvalue())

    @staticmethod
    def load(path: Path) -> Space:
        with np.load(path, allow_pickle=False) as z:
            return Space(
                [str(w) for w in z["vocab"]],
                z["vectors"].astype(np.float32),
                z["counts"].astype(np.int64),
                int(z["seed"]),
            )


def normalise(m: F32) -> F32:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: F32 = (m / norms).astype(np.float32)
    return out


def sentences(ws: Workspace, slice_name: str | None) -> list[list[str]]:
    df = (
        ws.resolve_slice(slice_name)
        .filter(pl.col("norm").str.contains(r"\w"))
        .group_by("doc_id", "sent", maintain_order=True)
        .agg(pl.col("norm").sort_by("i"))
        .sort("doc_id", "sent")
        .collect()
    )
    return [list(s) for s in df["norm"].to_list()]


@dataclass(frozen=True)
class TrainParams:
    algo: str = "w2v"
    dim: int = 100
    window: int = 5
    min_count: int = 5
    epochs: int = 5
    sg: bool = True

    def check(self) -> None:
        if self.algo not in ALGOS:
            raise UsageError(f"unknown algo '{self.algo}'", f"use one of: {', '.join(ALGOS)}")
        for name in ("dim", "window", "min_count", "epochs"):
            if getattr(self, name) < 1:
                raise UsageError(f"--{name.replace('_', '-')} must be at least 1")


def train_space(sents: Sequence[Sequence[str]], p: TrainParams, seed: int) -> Space:
    try:
        from gensim.models import FastText, Word2Vec
    except ImportError:
        raise missing_extra("gensim", "vec") from None
    cls = Word2Vec if p.algo == "w2v" else FastText
    try:
        model = cls(
            sentences=[list(s) for s in sents],
            vector_size=p.dim,
            window=p.window,
            min_count=p.min_count,
            epochs=p.epochs,
            sg=int(p.sg),
            seed=seed,
            workers=1,  # determinism
            hashfxn=_stable_hash,
        )
    except MemoryError:
        raise ResourceError(
            "out of memory while training", "lower --dim or raise --min-count"
        ) from None
    except RuntimeError as e:
        if "vocabulary" in str(e):
            raise InputError(
                "no word reaches --min-count in this slice",
                "lower --min-count or use a larger slice",
            ) from None
        raise
    kv = model.wv
    vocab: list[str] = [str(w) for w in kv.index_to_key]
    counts = np.array([kv.get_vecattr(w, "count") for w in vocab], np.int64)
    return Space(vocab, normalise(np.asarray(kv.vectors, np.float32)), counts, seed)


def store_dir(ws: Workspace, slice_name: str | None) -> Path:
    return ws.state / "vec" / (slice_name or "_all")


def train_replicates(
    ws: Workspace, slice_name: str | None, p: TrainParams, seeds: Iterable[int]
) -> dict[str, object]:
    p.check()
    sents = sentences(ws, slice_name)
    if not sents:
        raise InputError("no sentences to train on")
    d = store_dir(ws, slice_name)
    seeds = list(seeds)
    spaces = [train_space(sents, p, s) for s in seeds]
    for sp in spaces:
        sp.save(d / f"seed{sp.seed}.npz")
    for old in d.glob("seed*.npz"):
        if int(old.stem[4:]) not in seeds:
            old.unlink()
    manifest: dict[str, object] = {
        "slice": slice_name,
        "params": p.__dict__,
        "seeds": seeds,
        "corpus_hash": ws.corpus_hash(),
        "gensim": dist_version("gensim"),
        "tokens": int(sum(len(s) for s in sents)),
        "vocab": len(spaces[0].vocab),
    }
    atomic_write_bytes(d / "manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


def load_replicates(ws: Workspace, slice_name: str | None) -> list[Space]:
    d = store_dir(ws, slice_name)
    files = sorted(d.glob("seed*.npz"), key=lambda f: int(f.stem[4:]))
    if not files:
        where = f"--slice {slice_name}" if slice_name else ""
        raise InputError(
            f"no embeddings for slice '{slice_name or 'all'}'",
            f"train them with: semdrift vec train {where}".strip(),
        )
    m = d / "manifest.json"
    if m.exists():
        saved = json.loads(m.read_text(encoding="utf-8")).get("corpus_hash")
        if saved and saved != ws.corpus_hash():
            from semdrift.cli.output import warn

            warn("the corpus changed since these embeddings were trained; retrain to be current")
    return [Space.load(f) for f in files]


def require(spaces: Sequence[Space], words: Iterable[str]) -> None:
    vocab = spaces[0].vocab
    for w in words:
        if any(w not in s.index for s in spaces):
            near = difflib.get_close_matches(w, vocab, n=5)
            hint = f"did you mean: {', '.join(near)}?" if near else "lower --min-count and retrain"
            raise InputError(f"'{w}' is not in the embedding vocabulary", hint)


def knn_rows(
    query: F32, matrix: F32, k: int, exclude: Sequence[int] = ()
) -> tuple[npt.NDArray[np.int64], F32]:
    sims = matrix @ query
    if exclude:
        sims[list(exclude)] = -np.inf
    k = min(k, sims.size - len(exclude))
    if k <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    return idx.astype(np.int64), sims[idx]


def mean_topk_sim(space: Space, k: int = 10, chunk: int = 2048) -> F32:
    """r_k(w): mean cosine of w to its k nearest neighbours (for CSLS; Conneau et al. 2018)."""
    m = space.vectors
    out = np.empty(m.shape[0], np.float32)
    kk = min(k, m.shape[0] - 1)
    for start in range(0, m.shape[0], chunk):
        sims = m[start : start + chunk] @ m.T
        rows = np.arange(sims.shape[0])
        sims[rows, rows + start] = -np.inf
        top = np.partition(-sims, kk - 1, axis=1)[:, :kk]
        out[start : start + chunk] = -top.mean(axis=1)
    return out


@dataclass
class Neighbour:
    word: str
    mean_sim: float
    sd_sim: float
    replicates: int


def neighbours(
    spaces: Sequence[Space], word: str, k: int = 20, csls: bool = False
) -> tuple[list[Neighbour], float]:
    """Neighbours aggregated over replicates, plus top-k stability (mean pairwise Jaccard)."""
    require(spaces, [word])
    per: list[dict[str, float]] = []
    for sp in spaces:
        q = sp.vec(word)
        if csls:
            r = mean_topk_sim(sp)
            sims = 2 * (sp.vectors @ q) - r - r[sp.index[word]]
            sims[sp.index[word]] = -np.inf
            idx = np.argsort(-sims)[:k]
            per.append({sp.vocab[i]: float(sims[i]) for i in idx})
        else:
            idx, sims = knn_rows(q, sp.vectors, k, [sp.index[word]])
            per.append({sp.vocab[i]: float(s) for i, s in zip(idx, sims, strict=True)})
    stability = (
        float(
            np.mean(
                [
                    len(a.keys() & b.keys()) / len(a.keys() | b.keys())
                    for a, b in combinations(per, 2)
                ]
            )
        )
        if len(per) > 1
        else float("nan")
    )
    pool = set().union(*(p.keys() for p in per))
    rows: list[Neighbour] = []
    for w in pool:
        vals = [p[w] for p in per if w in p]
        rows.append(Neighbour(w, float(np.mean(vals)), float(np.std(vals)), len(vals)))
    rows.sort(key=lambda n: (-n.replicates, -n.mean_sim, n.word))
    return rows[:k], stability


def analogy(spaces: Sequence[Space], a: str, b: str, c: str, k: int = 10) -> list[Neighbour]:
    """a : b :: c : ?  via 3CosAdd (Mikolov et al. 2013), averaged over replicates."""
    require(spaces, [a, b, c])
    scores: dict[str, list[float]] = {}
    for sp in spaces:
        target = normalise((sp.vec(b) - sp.vec(a) + sp.vec(c))[None, :])[0]
        idx, sims = knn_rows(target, sp.vectors, k, [sp.index[w] for w in (a, b, c)])
        for i, s in zip(idx, sims, strict=True):
            scores.setdefault(sp.vocab[i], []).append(float(s))
    rows = [Neighbour(w, float(np.mean(v)), float(np.std(v)), len(v)) for w, v in scores.items()]
    rows.sort(key=lambda n: (-n.replicates, -n.mean_sim, n.word))
    return rows[:k]


def axis_scores(
    spaces: Sequence[Space], words: Sequence[str], pos: Sequence[str], neg: Sequence[str]
) -> list[dict[str, float | str]]:
    """Cosine of each word with the axis mean(pos) - mean(neg) (SemAxis; An et al. 2018)."""
    if not pos or not neg:
        raise UsageError("an axis needs at least one --pos and one --neg word")
    require(spaces, [*words, *pos, *neg])
    out: list[dict[str, float | str]] = []
    per_word: dict[str, list[float]] = {w: [] for w in words}
    for sp in spaces:
        ax = np.mean([sp.vec(w) for w in pos], axis=0) - np.mean([sp.vec(w) for w in neg], axis=0)
        ax = ax / (np.linalg.norm(ax) or 1.0)
        for w in words:
            per_word[w].append(float(sp.vec(w) @ ax))
    for w, v in per_word.items():
        out.append({"word": w, "score": float(np.mean(v)), "sd": float(np.std(v))})
    out.sort(key=lambda r: -float(r["score"]))
    return out
