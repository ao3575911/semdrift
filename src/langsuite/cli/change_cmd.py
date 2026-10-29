"""`langsuite change ...`: calibrated semantic change."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

import typer

from langsuite.cli.output import SETTINGS, emit, warn
from langsuite.corpus.workspace import Workspace
from langsuite.errors import UsageError

app = typer.Typer(help="Semantic change with nulls, intervals and FDR control.")

Slices = Annotated[
    str, typer.Option("--slices", help="Comma-separated slice names, e.g. early,late.")
]


def _ws() -> Workspace:
    from langsuite.cli.commands import ws

    return ws()


def _split(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        raise UsageError("--slices needs at least two names, e.g. --slices early,late")
    return parts


@app.command("score")
def score_cmd(
    slices: Slices,
    top: Annotated[int, typer.Option(help="Rows to show.")] = 20,
    null_runs: Annotated[int, typer.Option(help="Shuffled-slice null runs (cached).")] = 3,
    min_count: Annotated[int, typer.Option(help="Minimum frequency in both slices.")] = 10,
    all_words: Annotated[
        bool, typer.Option("--all", help="Include non-significant words.")
    ] = False,
) -> None:
    """Rank words by change net of a frequency-matched shuffled-slice null, with BH q-values."""
    from langsuite.change.score import score

    a, b = _split(slices)[:2]
    rows, meta = score(_ws(), a, b, null_runs, min_count, progress=True)
    shown = rows if all_words else [r for r in rows if r.q < 0.05]
    if not all_words and not shown:
        warn("no word changed more than the shuffled null at q < 0.05; showing the top anyway")
        shown = rows
    shown = shown[:top]
    if SETTINGS.json:
        emit({**meta, "words": [asdict(r) for r in shown]})
        return
    emit(
        [
            {
                "word": r.word,
                "change": r.change,
                "raw": r.raw,
                "±seed": r.seed_sd,
                "q": r.q,
                "nn_overlap": r.nn_overlap,
                "stable": "yes" if r.stable else "no",
                "freq": f"{r.freq_a}/{r.freq_b}",
            }
            for r in shown
        ],
        title=f"{a} -> {b}: {meta['significant_q05']} of {meta['tested_words']} words at q<0.05",
    )
    if meta["replicates"] == 1:
        warn("one replicate per slice: stability can't be judged; train with --seeds 3 or more")


@app.command("drift")
def drift_cmd(
    word: Annotated[str, typer.Argument(help="Word to follow.")],
    slices: Slices,
    k: Annotated[int, typer.Option("-k", help="Neighbours per slice.")] = 8,
) -> None:
    """Neighbour timeline across slices (aligned distance from the first slice)."""
    from langsuite.change.score import drift

    emit(drift(_ws(), word.casefold(), _split(slices), k))
