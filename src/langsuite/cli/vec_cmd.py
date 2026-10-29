"""`langsuite vec ...`: static embeddings with seed replicates."""

from __future__ import annotations

from typing import Annotated

import typer

from langsuite.cli.output import emit, info
from langsuite.corpus.workspace import Workspace
from langsuite.errors import UsageError

app = typer.Typer(help="Static embeddings: train replicates, neighbours, analogies, axes.")

Slice = Annotated[str | None, typer.Option("--slice", "-s", help="Slice name (default: all).")]
K = Annotated[int, typer.Option("-k", help="Number of results.")]


def _ws() -> Workspace:
    from langsuite.cli.commands import ws

    return ws()


@app.command("train")
def train(
    slice_name: Slice = None,
    algo: Annotated[str, typer.Option(help="w2v or fasttext.")] = "w2v",
    seeds: Annotated[int, typer.Option(help="Seed replicates (stability needs >= 2).")] = 3,
    seed: Annotated[int, typer.Option(help="First seed.")] = 0,
    dim: Annotated[int, typer.Option(help="Vector size.")] = 100,
    window: Annotated[int, typer.Option(help="Context window.")] = 5,
    min_count: Annotated[int, typer.Option(help="Minimum frequency.")] = 5,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 5,
) -> None:
    """Train SGNS word2vec or fastText on a slice, once per seed."""
    from langsuite.geom.space import TrainParams, train_replicates

    if seeds < 1:
        raise UsageError("--seeds must be at least 1")
    p = TrainParams(algo, dim, window, min_count, epochs)
    info(f"training {seeds} replicate(s) on slice '{slice_name or 'all'}' ...")
    m = train_replicates(_ws(), slice_name, p, range(seed, seed + seeds))
    emit({k: v for k, v in m.items() if k != "params"} | {"algo": algo, "dim": dim})


@app.command("nn")
def nn(
    word: Annotated[str, typer.Argument(help="Query word (casefolded).")],
    slice_name: Slice = None,
    k: K = 15,
    csls: Annotated[
        bool, typer.Option("--csls", help="CSLS instead of cosine (less hubness).")
    ] = False,
) -> None:
    """Nearest neighbours, averaged over seed replicates, with a stability score."""
    from langsuite.cli.output import SETTINGS
    from langsuite.geom.space import load_replicates, neighbours

    spaces = load_replicates(_ws(), slice_name)
    rows, stab = neighbours(spaces, word.casefold(), k, csls)
    data = [
        {
            "word": n.word,
            "sim": n.mean_sim,
            "sd": n.sd_sim,
            "replicates": f"{n.replicates}/{len(spaces)}",
        }
        for n in rows
    ]
    if SETTINGS.json:
        emit({"word": word, "stability_jaccard": stab, "neighbours": data})
        return
    emit(data, title=f"{word} - top-{k} stability (Jaccard) {stab:.2f}")
    if stab == stab and stab < 0.5:
        from langsuite.cli.output import warn

        warn("neighbours are unstable across seeds (<0.5); treat them with caution")


@app.command("analogy")
def analogy_cmd(a: str, b: str, c: str, slice_name: Slice = None, k: K = 10) -> None:
    """a : b :: c : ?  (3CosAdd)."""
    from langsuite.geom.space import analogy, load_replicates

    spaces = load_replicates(_ws(), slice_name)
    rows = analogy(spaces, a.casefold(), b.casefold(), c.casefold(), k)
    emit(
        [
            {"word": n.word, "sim": n.mean_sim, "sd": n.sd_sim, "replicates": n.replicates}
            for n in rows
        ]
    )


@app.command("axis")
def axis(
    words: Annotated[list[str], typer.Argument(help="Words to place on the axis.")],
    pos: Annotated[list[str], typer.Option("--pos", help="Positive pole word (repeatable).")],
    neg: Annotated[list[str], typer.Option("--neg", help="Negative pole word (repeatable).")],
    slice_name: Slice = None,
) -> None:
    """Project words onto a semantic axis, e.g. --pos good --neg bad."""
    from langsuite.geom.space import axis_scores, load_replicates

    spaces = load_replicates(_ws(), slice_name)
    f = str.casefold
    emit(axis_scores(spaces, [f(w) for w in words], [f(w) for w in pos], [f(w) for w in neg]))
