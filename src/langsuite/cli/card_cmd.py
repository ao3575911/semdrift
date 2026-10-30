"""`langsuite card WORD`: one evidence card from every available measurement."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from langsuite.cli.output import SETTINGS, emit, info, out_console
from langsuite.corpus.workspace import atomic_write_bytes
from langsuite.errors import UsageError


def card_cmd(
    word: Annotated[str, typer.Argument(help="Word to report on.")],
    slices: Annotated[str, typer.Option("--slices", help="Comma-separated slices, in order.")],
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Add LM surprisal with this pulled model.")
    ] = None,
    null_runs: Annotated[int, typer.Option(help="Shuffled-slice null runs.")] = 3,
    min_count: Annotated[int, typer.Option(help="Minimum frequency for the change test.")] = 10,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Markdown file to write.")] = None,
) -> None:
    """Frequency, examples, change test, drift and surprisal, with methods and provenance."""
    from langsuite.cli.commands import ws as get_ws
    from langsuite.evidence.card import build, markdown

    names = [s.strip() for s in slices.split(",") if s.strip()]
    if not names:
        raise UsageError("--slices needs at least one slice name")
    ws = get_ws()
    card = build(ws, word, names, model, null_runs, min_count)
    md = markdown(card)
    safe = re.sub(r"[^\w-]+", "_", card["word"]) or "word"
    dest = out or ws.state / "cards" / f"{safe}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(dest, md.encode())
    atomic_write_bytes(dest.with_suffix(".json"), json.dumps(card, indent=2, default=str).encode())
    if SETTINGS.json:
        emit(card)
        return
    from rich.markdown import Markdown

    out_console.print(Markdown(md))
    info(f"wrote {dest} and {dest.with_suffix('.json').name}")
