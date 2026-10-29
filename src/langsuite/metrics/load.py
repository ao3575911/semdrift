"""Turn a workspace slice into per-document integer code arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from langsuite.corpus.workspace import Workspace
from langsuite.errors import InputError, UsageError
from langsuite.metrics.info import IntArray

UNITS = ("norm", "lemma", "form")


@dataclass
class Coded:
    docs: list[IntArray]
    vocab: list[str]

    @property
    def tokens(self) -> int:
        return int(sum(d.size for d in self.docs))


def load_frame(ws: Workspace, slice_name: str | None, unit: str, keep_punct: bool) -> pl.DataFrame:
    if unit not in UNITS:
        raise UsageError(f"unknown unit '{unit}'", f"use one of: {', '.join(UNITS)}")
    lf = ws.resolve_slice(slice_name).select("doc_id", "i", pl.col(unit).alias("u"))
    if not keep_punct:
        lf = lf.filter(pl.col("u").str.contains(r"\w"))
    df = lf.drop_nulls("u").sort("doc_id", "i").collect()
    if df.height == 0:
        what = f"slice '{slice_name}'" if slice_name else "the corpus"
        hint = (
            "this corpus has no lemmas; re-ingest with a spaCy/Stanza pipeline or use --unit norm"
        )
        raise InputError(f"{what} has no '{unit}' tokens", hint if unit == "lemma" else None)
    return df


def encode(frames: list[pl.DataFrame]) -> list[Coded]:
    """Encode several frames against one shared vocabulary (needed for divergences)."""
    vocab = sorted(set().union(*(set(f["u"].unique().to_list()) for f in frames)))
    index = pl.DataFrame({"u": vocab, "code": np.arange(len(vocab), dtype=np.int64)})
    out: list[Coded] = []
    for f in frames:
        j = f.join(index, on="u", how="left", maintain_order="left")
        docs = [
            g["code"].to_numpy().astype(np.int64)
            for _, g in j.group_by("doc_id", maintain_order=True)
        ]
        out.append(Coded(docs, vocab))
    return out
