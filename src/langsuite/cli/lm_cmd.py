"""`langsuite lm ...`: language-model surprisal, offline after one pull."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import polars as pl
import typer

from langsuite.cli.output import emit, info, warn
from langsuite.corpus.workspace import Workspace
from langsuite.errors import InputError, UsageError
from langsuite.lm.model import DEFAULT_MODEL

app = typer.Typer(help="LM surprisal: per-word bits, slice predictability, surprising words.")

Slice = Annotated[str | None, typer.Option("--slice", "-s", help="Slice name (default: all).")]
Model = Annotated[str, typer.Option("--model", "-m", help="Pulled model name or local dir.")]
Stride = Annotated[int, typer.Option(help="Sliding-window stride in tokens.")]
Groups = Annotated[int, typer.Option("--groups", "-g", help="Jackknife groups for the 95% CI.")]


def _ws() -> Workspace:
    from langsuite.cli.commands import ws

    return ws()


def _frame(ws: Workspace, slice_name: str | None, model: str, stride: int) -> pl.DataFrame:
    from langsuite.lm.model import load
    from langsuite.lm.surprisal import surprisal_frame

    if stride < 1:
        raise UsageError("--stride must be at least 1")
    tokens = ws.resolve_slice(slice_name).collect()
    if tokens.height == 0:
        raise InputError(f"slice '{slice_name or 'all'}' has no tokens")
    lm = load(model)
    n_docs = tokens["doc_id"].n_unique()
    info(f"scoring {tokens.height} words in {n_docs} documents with {lm.name}")
    return surprisal_frame(ws, tokens, lm, stride)


def _jack(per_doc: pl.DataFrame, groups: int) -> tuple[float, float, float]:
    """Ratio estimator sum(bits)/sum(words) with a delete-a-group jackknife interval."""
    b = per_doc["bits"].to_numpy()
    n = per_doc["words"].to_numpy().astype(float)
    value = float(b.sum() / n.sum())
    g = min(groups, len(b))
    if g < 2:
        return value, float("nan"), float("nan")
    label = np.random.default_rng(0).permutation(len(b)) % g
    th = np.array([b[label != k].sum() / n[label != k].sum() for k in range(g)])
    se = float(np.sqrt((g - 1) / g * ((th - th.mean()) ** 2).sum()))
    return value, value - 1.96 * se, value + 1.96 * se


def _per_doc(f: pl.DataFrame) -> pl.DataFrame:
    return f.group_by("doc_id").agg(pl.col("bits").sum(), pl.len().alias("words"))


@app.command("pull")
def pull_cmd(
    name: Annotated[str, typer.Argument(help="Hugging Face model id.")] = DEFAULT_MODEL,
    accept_licence: Annotated[bool, typer.Option("--accept-licence")] = False,
) -> None:
    """Download a model once; every other lm command then runs offline."""
    from langsuite.lm.model import pull

    path, lic = pull(name, accept_licence)
    emit({"model": name, "licence": lic, "path": str(path)}, title="pulled")


@app.command("score")
def score_cmd(
    slices: Annotated[
        str | None, typer.Option(help="Comma-separated slices (default: all).")
    ] = None,
    model: Model = DEFAULT_MODEL,
    stride: Stride = 512,
    groups: Groups = 20,
) -> None:
    """Mean surprisal (bits/word) and word perplexity per slice, with 95% intervals."""
    ws = _ws()
    names: list[str | None] = [s.strip() for s in slices.split(",")] if slices else [None]
    rows = []
    for name in names:
        f = _frame(ws, name, model, stride)
        v, lo, hi = _jack(_per_doc(f), groups)
        rows.append(
            {
                "slice": name or "all",
                "bits/word": v,
                "ci_low": lo,
                "ci_high": hi,
                "perplexity": 2.0**v,
                "words": f.height,
                "docs": f["doc_id"].n_unique(),
            }
        )
    if len(rows) == 2:
        a, b = rows
        overlap = not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])
        if overlap:
            warn("the two intervals overlap: the difference may be noise")
    emit(rows, title=f"surprisal under {model}")


@app.command("top")
def top_cmd(
    slice_name: Slice = None,
    model: Model = DEFAULT_MODEL,
    stride: Stride = 512,
    n: Annotated[int, typer.Option("-n", help="Number of words.")] = 15,
    context: Annotated[int, typer.Option(help="Words of context each side.")] = 5,
) -> None:
    """The most surprising words in a slice, in context (KWIC-style)."""
    ws = _ws()
    f = _frame(ws, slice_name, model, stride).sort(["doc_id", "sent", "i"])
    forms = f["form"].to_list()
    order = np.argsort(-f["bits"].to_numpy(), kind="stable")[:n]
    docs = f["doc_id"].to_list()
    rows = []
    for j in order:
        lo, hi = max(0, j - context), min(len(forms), j + context + 1)
        while docs[lo] != docs[j]:
            lo += 1
        while docs[hi - 1] != docs[j]:
            hi -= 1
        rows.append(
            {
                "bits": float(f["bits"][int(j)]),
                "left": " ".join(forms[lo:j]),
                "word": forms[j],
                "right": " ".join(forms[j + 1 : hi]),
                "doc": docs[j],
            }
        )
    emit(rows, title="most surprising")


@app.command("words")
def words_cmd(
    slice_name: Slice = None,
    model: Model = DEFAULT_MODEL,
    stride: Stride = 512,
    min_count: Annotated[int, typer.Option(help="Minimum frequency.")] = 3,
    n: Annotated[int, typer.Option("-n", help="Number of types.")] = 20,
) -> None:
    """Word types ranked by mean surprisal, net of what their frequency predicts."""
    ws = _ws()
    f = _frame(ws, slice_name, model, stride)
    t = (
        f.group_by("norm")
        .agg(pl.col("bits").mean().alias("mean_bits"), pl.len().alias("count"))
        .filter(pl.col("count") >= min_count)
    )
    if t.height < 3:
        raise InputError("too few word types reach --min-count", "lower --min-count")
    x = np.log(t["count"].to_numpy().astype(float))
    y = t["mean_bits"].to_numpy()
    slope, icept = np.polyfit(x, y, 1)
    t = t.with_columns(excess=pl.Series(y - (icept + slope * x))).sort("excess", descending=True)
    emit(t.head(n).to_dicts(), title="surprisal above the frequency trend")
