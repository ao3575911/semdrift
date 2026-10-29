"""Ingest files into a workspace through a content-hashed token cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from langsuite.corpus.pipeline import Pipeline
from langsuite.corpus.readers import RawDoc, RawToken, Scalar, detect_format, iter_docs
from langsuite.corpus.workspace import (
    TOKEN_SCHEMA,
    Workspace,
    atomic_write_parquet,
    cache_root,
    docs_frame,
    hash_bytes,
)
from langsuite.errors import InputError


@dataclass
class IngestReport:
    files: int
    docs_added: int
    docs_skipped: int
    tokens: int
    cache_hits: int
    pipeline: str


def _tokens_frame(doc_id: str, toks: list[RawToken]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "doc_id": [doc_id] * len(toks),
            "sent": [t.sent for t in toks],
            "i": list(range(len(toks))),
            "form": [t.form for t in toks],
            "norm": [t.form.casefold() for t in toks],
            "lemma": [t.lemma for t in toks],
            "upos": [t.upos for t in toks],
        },
        schema=TOKEN_SCHEMA,
    )


def ingest(
    ws: Workspace,
    paths: list[Path],
    pipeline: Pipeline,
    fmt: str | None = None,
    text_field: str = "text",
    meta: dict[str, Scalar] | None = None,
) -> IngestReport:
    if not paths:
        raise InputError("no input files", "pass one or more files or a glob")
    all_docs: list[RawDoc] = []
    hashes: list[str] = []
    sources: list[str] = []
    tok_frames: list[pl.DataFrame] = []
    hits = 0
    for path in paths:
        f = fmt or detect_format(path)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise InputError(f"file not found: {path}") from None
        except IsADirectoryError:
            raise InputError(
                f"{path} is a directory", "pass files, or a glob like dir/*.txt"
            ) from None
        except OSError as e:
            raise InputError(f"can't read {path}: {e.strerror}") from None
        key = hash_bytes(data, f.encode(), pipeline.id.encode(), text_field.encode())
        cdir = cache_root() / "ingest" / key
        docs = list(iter_docs(path, f, text_field))
        cached = cdir / "tokens.parquet"
        if cached.exists():
            try:
                toks_all = pl.read_parquet(cached)
                hits += 1
            except Exception:
                cached.unlink(missing_ok=True)  # corrupt cache entry: rebuild
                toks_all = None
        else:
            toks_all = None
        if toks_all is None:
            frames = []
            for d in docs:
                toks = d.tokens if d.tokens is not None else pipeline.run(d.text or "")
                frames.append(_tokens_frame(d.doc_id, toks))
            toks_all = pl.concat(frames) if frames else pl.DataFrame(schema=TOKEN_SCHEMA)
            atomic_write_parquet(toks_all, cached)
        for d in docs:
            d.meta = {**d.meta, **(meta or {})}
            body = d.text.encode() if d.text is not None else repr(d.tokens).encode()
            hashes.append(hash_bytes(body))
            sources.append(path.name)
        all_docs.extend(docs)
        tok_frames.append(toks_all)
    ids = [d.doc_id for d in all_docs]
    if len(set(ids)) != len(ids):
        dup = next(i for i in ids if ids.count(i) > 1)
        raise InputError(f"duplicate document id '{dup}'", "give each document a unique id")
    tokens = pl.concat(tok_frames)
    if tokens.height == 0:
        raise InputError("the input contains no tokens")
    added, skipped = ws.add(docs_frame(all_docs, hashes, sources), tokens)
    return IngestReport(len(paths), added, skipped, tokens.height, hits, pipeline.id)
