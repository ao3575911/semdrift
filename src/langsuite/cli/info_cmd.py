"""`langsuite info ...`: information measures with bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated

import numpy as np
import typer

from langsuite.cli.output import emit
from langsuite.corpus.workspace import Workspace
from langsuite.errors import InputError, UsageError
from langsuite.metrics import info as m
from langsuite.metrics.info import Estimate, IntArray
from langsuite.metrics.load import encode, load_frame

app = typer.Typer(help="Information measures: entropy, diversity, Zipf/Heaps, divergence.")

Slice = Annotated[str | None, typer.Option("--slice", "-s", help="Slice name (default: all).")]
Unit = Annotated[str, typer.Option(help="Token unit: norm (casefolded), lemma or form.")]
Groups = Annotated[
    int, typer.Option("--groups", "-g", help="Jackknife document groups for the 95% CI (0 = off).")
]
Seed = Annotated[int, typer.Option(help="Random seed.")]
Punct = Annotated[bool, typer.Option("--keep-punct", help="Count punctuation tokens.")]


def _ws() -> Workspace:
    from langsuite.cli.commands import ws

    return ws()


def _few(n_docs: int, groups: int) -> None:
    from langsuite.cli.output import warn

    if groups and n_docs < 10:
        warn(f"only {n_docs} documents: the interval is rough; split long texts into documents")


def _row(name: str, e: Estimate) -> dict[str, object]:
    return {
        "measure": name,
        "value": e.value,
        "ci_low": e.ci_low,
        "ci_high": e.ci_high,
        "groups": e.resamples,
    }


def _cat(ds: Sequence[IntArray]) -> IntArray:
    return np.concatenate(list(ds)) if ds else np.empty(0, np.int64)


def _measures(window: int) -> dict[str, Callable[[Sequence[IntArray]], float]]:
    return {
        "H1 (bits/token)": lambda ds: m.unigram_entropy(_cat(ds)),
        "H(X,Y) bigram": lambda ds: m.bigram_entropies(ds)[0],
        "H(Y|X) conditional": lambda ds: m.bigram_entropies(ds)[1],
        "TTR": lambda ds: m.ttr(_cat(ds)),
        f"MATTR (W={window})": lambda ds: m.mattr(_cat(ds), window),
        "Zipf s": lambda ds: m.zipf_exponent(_cat(ds)),
        "Heaps beta": lambda ds: m.heaps_fit(_cat(ds))[1],
    }


@app.command("summary")
def summary(
    slice_name: Slice = None,
    unit: Unit = "norm",
    groups: Groups = 20,
    seed: Seed = 0,
    window: Annotated[int, typer.Option(help="MATTR window.")] = 500,
    keep_punct: Punct = False,
) -> None:
    """All measures for one slice, each with a 95% group-jackknife interval."""
    if window < 1:
        raise UsageError("--window must be at least 1")
    coded = encode([load_frame(_ws(), slice_name, unit, keep_punct)])[0]
    _few(len(coded.docs), groups)
    rows = [
        _row(
            k,
            m.group_jackknife(coded.docs, f, groups, seed)
            if groups
            else m.Estimate(f(coded.docs), float("nan"), float("nan"), 0),
        )
        for k, f in _measures(window).items()
    ]
    rows.insert(
        0,
        {
            "measure": "tokens",
            "value": coded.tokens,
            "ci_low": None,
            "ci_high": None,
            "groups": 0,
        },
    )
    rows.insert(
        1,
        {
            "measure": "types",
            "value": len(coded.vocab),
            "ci_low": None,
            "ci_high": None,
            "groups": 0,
        },
    )
    emit(rows, title=f"slice: {slice_name or 'all'}")


@app.command("divergence")
def divergence(
    a: Annotated[str, typer.Argument(help="First slice.")],
    b: Annotated[str, typer.Argument(help="Second slice.")],
    measure: Annotated[str, typer.Option(help="js (symmetric, [0,1]) or kl (D(A||B)).")] = "js",
    unit: Unit = "norm",
    groups: Groups = 20,
    seed: Seed = 0,
    alpha: Annotated[float, typer.Option(help="Add-alpha smoothing for KL.")] = 0.5,
    keep_punct: Punct = False,
    top: Annotated[int, typer.Option(help="Show the words contributing most.")] = 10,
) -> None:
    """Divergence between two slices, with a 95% jackknife CI and the top contributing words."""
    if measure not in ("js", "kl"):
        raise UsageError(f"unknown measure '{measure}'", "use js or kl")
    if alpha <= 0 and measure == "kl":
        raise UsageError("--alpha must be > 0 for KL (zero counts make it infinite)")
    w = _ws()
    ca, cb = encode([load_frame(w, a, unit, keep_punct), load_frame(w, b, unit, keep_punct)])
    v = len(ca.vocab)
    _few(min(len(ca.docs), len(cb.docs)), groups)

    def counts(ds: Sequence[IntArray]) -> np.ndarray:
        return np.bincount(_cat(ds), minlength=v).astype(np.float64)

    def div(p: np.ndarray, q: np.ndarray) -> float:
        return m.js_divergence(p, q) if measure == "js" else m.kl_divergence(p, q, alpha)

    value = div(counts(ca.docs), counts(cb.docs))
    g = min(groups, len(ca.docs), len(cb.docs))
    lo = hi = float("nan")
    if g >= 2:
        rng = np.random.default_rng(seed)
        la = rng.permutation(len(ca.docs)) % g
        lb = rng.permutation(len(cb.docs)) % g
        th = np.array(
            [
                div(
                    counts([d for d, x in zip(ca.docs, la, strict=True) if x != k]),
                    counts([d for d, x in zip(cb.docs, lb, strict=True) if x != k]),
                )
                for k in range(g)
            ]
        )
        se = float(np.sqrt((g - 1) / g * ((th - th.mean()) ** 2).sum()))
        lo, hi = value - 1.959964 * se, value + 1.959964 * se
    vals = list(range(g if g >= 2 else 0))
    p, q = counts(ca.docs), counts(cb.docs)
    if p.sum() == 0 or q.sum() == 0:
        raise InputError("one of the slices is empty after filtering")
    pn, qn = p / p.sum(), q / q.sum()
    mm = 0.5 * (pn + qn)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = 0.5 * np.nan_to_num(pn * np.log2(pn / mm)) + 0.5 * np.nan_to_num(
            qn * np.log2(qn / mm)
        )
    order = np.argsort(-contrib)[:top]
    emit(
        {
            "measure": measure,
            "a": a,
            "b": b,
            "value": value,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "groups": len(vals),
            "top_contributors": [
                {
                    "word": ca.vocab[i],
                    "share": float(contrib[i] / max(contrib.sum(), 1e-300)),
                    "more_in": a if pn[i] > qn[i] else b,
                }
                for i in order
            ],
        }
    )
    if top > 0:
        from langsuite.cli.output import SETTINGS

        if not SETTINGS.json:
            emit(
                [
                    {
                        "word": ca.vocab[i],
                        "JS share": float(contrib[i] / max(contrib.sum(), 1e-300)),
                        "more in": a if pn[i] > qn[i] else b,
                    }
                    for i in order
                ],
                title="top contributors (JS)",
            )
