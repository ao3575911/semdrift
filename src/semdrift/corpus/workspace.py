"""Workspace: `semdrift.toml` + `.semdrift/` holding the corpus as Parquet, slices and runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import xxhash
from platformdirs import user_cache_dir

from semdrift._meta import BIN, __version__
from semdrift.corpus.readers import RawDoc, Scalar
from semdrift.errors import InputError, UsageError

MARKER = "semdrift.toml"
STATE = ".semdrift"
TOKEN_SCHEMA = {
    "doc_id": pl.String,
    "sent": pl.Int32,
    "i": pl.Int32,
    "form": pl.String,
    "norm": pl.String,
    "lemma": pl.String,
    "upos": pl.String,
}


def cache_root() -> Path:
    return Path(os.environ.get("SEMDRIFT_CACHE_DIR") or user_cache_dir(BIN))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".parquet")
    os.close(fd)
    try:
        df.write_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def hash_bytes(*parts: bytes) -> str:
    h = xxhash.xxh3_128()
    for p in parts:
        h.update(len(p).to_bytes(8, "little"))
        h.update(p)
    return h.hexdigest()


_SLICE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


@dataclass
class Workspace:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / STATE

    @property
    def docs_path(self) -> Path:
        return self.state / "corpus" / "docs.parquet"

    @property
    def tokens_path(self) -> Path:
        return self.state / "corpus" / "tokens.parquet"

    @property
    def slices_path(self) -> Path:
        return self.state / "slices.json"

    @staticmethod
    def init(root: Path) -> Workspace:
        root.mkdir(parents=True, exist_ok=True)
        marker = root / MARKER
        if marker.exists():
            raise UsageError(f"{root} is already a workspace")
        marker.write_text(f'# {BIN} workspace\nversion = "{__version__}"\n', encoding="utf-8")
        (root / STATE / "corpus").mkdir(parents=True, exist_ok=True)
        (root / STATE / "runs").mkdir(parents=True, exist_ok=True)
        return Workspace(root)

    @staticmethod
    def open(start: Path) -> Workspace:
        start = start.resolve()
        for d in (start, *start.parents):
            if (d / MARKER).is_file():
                return Workspace(d)
        raise UsageError(
            f"no workspace found at or above {start}",
            f"create one with: {BIN} init <dir>, or pass --workspace",
        )

    # ---- corpus -------------------------------------------------------
    def has_corpus(self) -> bool:
        return self.docs_path.exists() and self.tokens_path.exists()

    def docs(self) -> pl.DataFrame:
        if not self.has_corpus():
            raise InputError("the workspace has no corpus yet", f"add files with: {BIN} ingest")
        return pl.read_parquet(self.docs_path)

    def tokens(self) -> pl.LazyFrame:
        if not self.has_corpus():
            raise InputError("the workspace has no corpus yet", f"add files with: {BIN} ingest")
        return pl.scan_parquet(self.tokens_path)

    def corpus_hash(self) -> str:
        docs = self.docs().sort("doc_id")
        return hash_bytes(*(h.encode() for h in docs["content_hash"].to_list()))

    def add(self, docs: pl.DataFrame, tokens: pl.DataFrame) -> tuple[int, int]:
        """Append documents; skip ids already present. Returns (added, skipped)."""
        if self.has_corpus():
            old_docs = pl.read_parquet(self.docs_path)
            fresh = docs.filter(~pl.col("doc_id").is_in(old_docs["doc_id"].implode()))
            skipped = docs.height - fresh.height
            tokens = tokens.filter(pl.col("doc_id").is_in(fresh["doc_id"].implode()))
            all_docs = pl.concat([old_docs, fresh], how="diagonal_relaxed")
            all_toks = pl.concat([pl.read_parquet(self.tokens_path), tokens])
        else:
            fresh, skipped, all_docs, all_toks = docs, 0, docs, tokens
        atomic_write_parquet(all_toks, self.tokens_path)
        atomic_write_parquet(all_docs, self.docs_path)
        return fresh.height, skipped

    # ---- slices -------------------------------------------------------
    def slices(self) -> dict[str, str]:
        if not self.slices_path.exists():
            return {}
        data: Any = json.loads(self.slices_path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}

    def define_slice(self, name: str, expr: str) -> int:
        if not _SLICE_NAME.match(name):
            raise UsageError(f"bad slice name '{name}'", "use letters, digits, _ or -")
        n = len(self.slice_doc_ids(expr))
        s = self.slices()
        s[name] = expr
        atomic_write_bytes(self.slices_path, json.dumps(s, indent=2).encode())
        return n

    def slice_doc_ids(self, expr: str) -> list[str]:
        docs = self.docs()
        if ";" in expr:
            raise UsageError("slice expressions can't contain ';'")
        try:
            out = pl.SQLContext(docs=docs).execute(
                f"SELECT doc_id FROM docs WHERE {expr}", eager=True
            )
        except Exception as e:
            cols = ", ".join(c for c in docs.columns if c not in ("content_hash",))
            raise UsageError(
                f"bad slice expression: {str(e).splitlines()[0][:200]}",
                f"write a SQL WHERE clause over: {cols}",
            ) from None
        return out["doc_id"].to_list()

    def resolve_slice(self, name: str | None) -> pl.LazyFrame:
        toks = self.tokens()
        if name is None:
            return toks
        s = self.slices()
        if name not in s:
            known = ", ".join(sorted(s)) or "(none)"
            raise UsageError(f"unknown slice '{name}'", f"defined slices: {known}")
        ids = self.slice_doc_ids(s[name])
        if not ids:
            raise InputError(f"slice '{name}' selects no documents")
        return toks.filter(pl.col("doc_id").is_in(ids))


def docs_frame(raw: list[RawDoc], hashes: list[str], source: list[str]) -> pl.DataFrame:
    rows: list[dict[str, Scalar]] = []
    for d, h, src in zip(raw, hashes, source, strict=True):
        rows.append({"doc_id": d.doc_id, "source": src, "content_hash": h, **d.meta})
    return pl.DataFrame(rows, infer_schema_length=None)
