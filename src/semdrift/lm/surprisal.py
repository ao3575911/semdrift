"""Word surprisal from a causal LM, with sliding-window context and word alignment.

Each document is the space-joined forms of its sentences. Subword surprisals (bits) are
summed into the corpus words they overlap (Pimentel et al. 2021 style word surprisal).
Long documents use overlapping windows; only tokens with at least `max_len - stride`
tokens of context are scored after the first window (Hugging Face perplexity recipe).
"""

from __future__ import annotations

import math
import re

import numpy as np
import polars as pl

from semdrift.corpus.workspace import Workspace, atomic_write_parquet, hash_bytes
from semdrift.lm.model import LM

LN2 = math.log(2.0)
# Detokenise so the LM sees natural text: "word," not "word ,".
_NO_SPACE_BEFORE = re.compile(
    r"^(?:[.,;:!?%)\]}\u201d\u2019]+|'s|'d|'ll|'re|'ve|'m|n't)$", re.IGNORECASE
)
_NO_SPACE_AFTER = {"(", "[", "{", "$", "\u201c", "\u2018"}


def doc_texts(frame: pl.DataFrame) -> list[tuple[str, str, list[tuple[int, int]], pl.DataFrame]]:
    out = []
    for (doc_id,), g in frame.sort(["doc_id", "sent", "i"]).group_by(
        ["doc_id"], maintain_order=True
    ):
        spans: list[tuple[int, int]] = []
        parts: list[str] = []
        pos = 0
        for form in g["form"].to_list():
            if parts and not _NO_SPACE_BEFORE.match(form) and parts[-1] not in _NO_SPACE_AFTER:
                pos += 1
                parts.append(" ")
            spans.append((pos, pos + len(form)))
            parts.append(form)
            pos += len(form)
        out.append((str(doc_id), "".join(parts), spans, g))
    return out


def token_bits(lm: LM, text: str, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Surprisal (bits) and end offset of every subword token of `text`."""
    import torch

    enc = lm.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids: list[int] = list(enc["input_ids"])
    ends = np.array([e for _, e in enc["offset_mapping"]], dtype=np.int64)
    bits = np.full(len(ids), np.nan)
    if not ids:
        return bits, ends
    full = [lm.bos, *ids]
    win = lm.max_len
    stride = max(1, min(stride, win - 1))
    start = 0
    scored = 0  # number of text tokens already scored
    with torch.inference_mode():
        while scored < len(ids):
            chunk = full[start : start + win]
            logits = lm.model(torch.tensor([chunk])).logits[0].float()
            logp = torch.log_softmax(logits[:-1], dim=-1)
            target = torch.tensor(chunk[1:])
            nll = -logp.gather(1, target[:, None])[:, 0].numpy() / LN2
            # chunk position j predicts text token start + j
            first = scored - start
            bits[start + first : start + len(nll)] = nll[first:]
            scored = start + len(nll)
            start += stride
    return bits, ends


def word_bits(spans: list[tuple[int, int]], bits: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Sum subword surprisal into words: token belongs to the word containing its last char."""
    starts = np.array([s for s, _ in spans])
    w = np.searchsorted(starts, np.maximum(ends - 1, 0), side="right") - 1
    out = np.zeros(len(spans))
    np.add.at(out, np.clip(w, 0, len(spans) - 1), np.nan_to_num(bits))
    return out


def cache_key(ws: Workspace, lm: LM, doc_ids: list[str], stride: int) -> str:
    return hash_bytes(ws.corpus_hash().encode(), lm.name.encode(), str(stride).encode())[:16]


def surprisal_frame(ws: Workspace, frame: pl.DataFrame, lm: LM, stride: int) -> pl.DataFrame:
    """Per-word surprisal, cached per (corpus, model, stride) under .semdrift/lm/."""
    key = cache_key(ws, lm, [], stride)
    path = ws.state / "lm" / f"{key}.parquet"
    cached = pl.read_parquet(path) if path.exists() else None
    want = set(frame["doc_id"].unique().to_list())
    have = set(cached["doc_id"].unique().to_list()) if cached is not None else set()
    todo = frame.filter(pl.col("doc_id").is_in(sorted(want - have)))
    fresh = []
    for _doc_id, text, spans, g in doc_texts(todo):
        bits, ends = token_bits(lm, text, stride)
        fresh.append(
            g.select("doc_id", "sent", "i").with_columns(bits=word_bits(spans, bits, ends))
        )
    parts = ([cached] if cached is not None else []) + fresh
    if fresh:
        atomic_write_parquet(pl.concat(parts), path)
    allw = pl.concat(parts) if parts else pl.DataFrame()
    return frame.join(allw, on=["doc_id", "sent", "i"], how="inner")
