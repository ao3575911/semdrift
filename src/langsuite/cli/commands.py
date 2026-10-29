"""Command registration. Each milestone adds commands here."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from langsuite._meta import BIN
from langsuite.cli.output import SETTINGS, emit, info
from langsuite.corpus.readers import FORMATS, Scalar, coerce
from langsuite.corpus.workspace import Workspace
from langsuite.errors import UsageError

slice_app = typer.Typer(help="Define and list corpus slices (metadata filters).")
corpus_app = typer.Typer(help="Corpus summary.")


def ws() -> Workspace:
    return Workspace.open(Path(SETTINGS.workspace))


def init(path: Annotated[Path, typer.Argument(help="Directory to create.")] = Path(".")) -> None:
    """Create a workspace."""
    w = Workspace.init(path)
    emit({"workspace": str(w.root.resolve())})
    info(f"next: {BIN} ingest FILES... (run inside {path})")


def _parse_meta(items: list[str]) -> dict[str, Scalar]:
    out: dict[str, Scalar] = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not k:
            raise UsageError(f"bad --meta '{item}'", "use KEY=VALUE, e.g. --meta year=1850")
        out[k] = coerce(v)
    return out


def ingest_cmd(
    paths: Annotated[list[Path], typer.Argument(help="Files to ingest.")],
    fmt: Annotated[
        str | None, typer.Option("--format", "-f", help=f"One of {', '.join(FORMATS)}.")
    ] = None,
    pipeline: Annotated[
        str, typer.Option(help="auto | basic | spacy:MODEL | stanza:LANG.")
    ] = "auto",
    text_field: Annotated[str, typer.Option(help="Text field for jsonl/csv.")] = "text",
    meta: Annotated[
        list[str] | None, typer.Option("--meta", help="KEY=VALUE added to every document.")
    ] = None,
) -> None:
    """Tokenise files and add them to the workspace corpus."""
    from langsuite.corpus.ingest import ingest
    from langsuite.corpus.pipeline import resolve

    if fmt is not None and fmt not in FORMATS:
        raise UsageError(f"unknown format '{fmt}'", f"use one of: {', '.join(FORMATS)}")
    w = ws()
    p = resolve(pipeline)
    info(f"pipeline: {p.id}")
    r = ingest(w, paths, p, fmt, text_field, _parse_meta(meta or []))
    emit(
        {
            "files": r.files,
            "docs_added": r.docs_added,
            "docs_skipped": r.docs_skipped,
            "tokens": r.tokens,
            "cache_hits": r.cache_hits,
            "pipeline": r.pipeline,
        }
    )


@corpus_app.command("info")
def corpus_info() -> None:
    """Documents, tokens, types, sentences and metadata columns."""
    w = ws()
    docs = w.docs()
    stats = (
        w.tokens()
        .select(
            pl.len().alias("tokens"),
            pl.col("norm").n_unique().alias("types"),
            pl.struct("doc_id", "sent").n_unique().alias("sentences"),
        )
        .collect()
        .row(0, named=True)
    )
    meta = [c for c in docs.columns if c not in ("doc_id", "source", "content_hash")]
    emit(
        {
            "documents": docs.height,
            **stats,
            "metadata": ", ".join(meta) or "(none)",
            "slices": ", ".join(sorted(w.slices())) or "(none)",
            "corpus_hash": w.corpus_hash()[:16],
        }
    )


@slice_app.command("define")
def slice_define(
    name: Annotated[str, typer.Argument(help="Slice name.")],
    expr: Annotated[str, typer.Argument(help='SQL WHERE clause, e.g. "year < 1900".')],
) -> None:
    """Define (or redefine) a named slice."""
    n = ws().define_slice(name, expr)
    emit({"slice": name, "expr": expr, "documents": n})


@slice_app.command("ls")
def slice_ls() -> None:
    """List slices and their document counts."""
    w = ws()
    rows = [
        {"slice": k, "expr": v, "documents": len(w.slice_doc_ids(v))}
        for k, v in sorted(w.slices().items())
    ]
    emit(rows)


def register(app: typer.Typer) -> None:
    if getattr(app, "_langsuite_registered", False):
        return
    app._langsuite_registered = True  # type: ignore[attr-defined]
    app.command("init")(init)
    app.command("ingest")(ingest_cmd)
    app.add_typer(corpus_app, name="corpus")
    app.add_typer(slice_app, name="slice")
    from langsuite.cli import info_cmd

    app.add_typer(info_cmd.app, name="info")
    from langsuite.cli import vec_cmd

    app.add_typer(vec_cmd.app, name="vec")
    from langsuite.cli import change_cmd

    app.add_typer(change_cmd.app, name="change")
