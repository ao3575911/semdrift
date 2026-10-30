"""Evidence cards: every measurement semdrift has for one word, with provenance.

Each section either has data or a `skipped` reason, so a card never fails as a whole
because one source is missing (no vectors, no pulled model, word too rare).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import platform
from collections.abc import Sequence
from importlib import metadata
from typing import Any

import numpy as np
import polars as pl

from semdrift._meta import __version__
from semdrift.corpus.workspace import Workspace
from semdrift.errors import InputError, UserError

Section = dict[str, Any]

METHODS = {
    "frequency": "Relative frequency per million tokens with a delete-a-group jackknife 95% "
    "interval over documents (Efron & Stein 1981).",
    "change": "Seed-replicated SGNS embeddings (Mikolov et al. 2013) aligned with orthogonal "
    "Procrustes (Hamilton et al. 2016); change is tested against a shuffled-slice null "
    "(Dubossarsky et al. 2017) per word on run-normalised distances, with Benjamini-Hochberg "
    "q-values (1995) and neighbour-overlap corroboration.",
    "surprisal": "Word surprisal in bits from a causal language model, subword surprisals "
    "summed per word (Pimentel et al. 2021), with sliding-window context.",
}


def _versions() -> dict[str, str]:
    out = {"semdrift": __version__, "python": platform.python_version()}
    for pkg in ("polars", "numpy", "gensim", "torch", "transformers"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[pkg] = metadata.version(pkg)
    return out


def _skip(e: Exception) -> Section:
    msg = e.message if isinstance(e, UserError) else f"{type(e).__name__}: {e}"
    return {"skipped": msg}


def frequency(ws: Workspace, word: str, slices: Sequence[str], groups: int = 20) -> Section:
    rows = []
    for s in slices:
        t = ws.resolve_slice(s).select("doc_id", "norm").collect()
        per = t.group_by("doc_id").agg(
            pl.len().alias("n"), (pl.col("norm") == word).sum().alias("hits")
        )
        n = per["n"].to_numpy().astype(float)
        h = per["hits"].to_numpy().astype(float)
        value = 1e6 * h.sum() / max(n.sum(), 1.0)
        g = min(groups, len(n))
        lo = hi = float("nan")
        if g >= 2:
            lab = np.random.default_rng(0).permutation(len(n)) % g
            th = np.array([1e6 * h[lab != k].sum() / n[lab != k].sum() for k in range(g)])
            se = float(np.sqrt((g - 1) / g * ((th - th.mean()) ** 2).sum()))
            lo, hi = max(0.0, value - 1.96 * se), value + 1.96 * se
        rows.append(
            {
                "slice": s,
                "count": int(h.sum()),
                "per_million": value,
                "ci_low": lo,
                "ci_high": hi,
                "docs_with": int((h > 0).sum()),
                "docs": len(n),
            }
        )
    return {"rows": rows}


def examples(ws: Workspace, word: str, slices: Sequence[str], n: int = 3, ctx: int = 6) -> Section:
    rows = []
    for s in slices:
        t = ws.resolve_slice(s).select("doc_id", "sent", "i", "form", "norm").collect()
        t = t.sort(["doc_id", "sent", "i"])
        hits = np.flatnonzero(t["norm"].to_numpy() == word)
        if hits.size == 0:
            continue
        pick = hits[np.linspace(0, hits.size - 1, min(n, hits.size)).astype(int)]
        forms, docs = t["form"].to_list(), t["doc_id"].to_list()
        for j in pick:
            lo, hi = max(0, j - ctx), min(len(forms), j + ctx + 1)
            left = [f for f, d in zip(forms[lo:j], docs[lo:j], strict=True) if d == docs[j]]
            right = [
                f for f, d in zip(forms[j + 1 : hi], docs[j + 1 : hi], strict=True) if d == docs[j]
            ]
            rows.append(
                {
                    "slice": s,
                    "left": " ".join(left),
                    "word": forms[j],
                    "right": " ".join(right),
                    "doc": docs[j],
                }
            )
    return {"rows": rows}


def change(
    ws: Workspace, word: str, slices: Sequence[str], null_runs: int, min_count: int
) -> Section:
    from semdrift.change.score import drift, score

    out: Section = {"timeline": drift(ws, word, slices)}
    rows, meta = score(ws, slices[0], slices[-1], null_runs, min_count)
    hit = [r for r in rows if r.word == word]
    if not hit:
        out["test"] = {"skipped": f"'{word}' is below --min-count {min_count} in a slice"}
    else:
        r = hit[0]
        rank = rows.index(r) + 1
        out["test"] = {
            "a": slices[0],
            "b": slices[-1],
            "change": r.change,
            "q": r.q,
            "p_empirical": r.p_empirical,
            "nn_overlap": r.nn_overlap,
            "stable": r.stable,
            "rank": rank,
            "tested": meta["tested_words"],
        }
    return out


def surprisal(ws: Workspace, word: str, slices: Sequence[str], model: str) -> Section:
    from semdrift.lm.model import load
    from semdrift.lm.surprisal import surprisal_frame

    lm = load(model)
    rows = []
    for s in slices:
        f = surprisal_frame(ws, ws.resolve_slice(s).collect(), lm, 512)
        w = f.filter(pl.col("norm") == word)["bits"]
        rows.append(
            {
                "slice": s,
                "word_bits": float(w.to_numpy().mean()) if w.len() else float("nan"),
                "slice_bits": float(f["bits"].to_numpy().mean()),
                "n": w.len(),
            }
        )
    return {"model": model, "rows": rows}


def verdict(card: dict[str, Any]) -> list[str]:
    """Plain-language findings; each one cites the section it comes from."""
    notes: list[str] = []
    fr = card["frequency"].get("rows", [])
    if len(fr) >= 2:
        a, b = fr[0], fr[-1]
        if a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]:
            way = "rises" if b["per_million"] > a["per_million"] else "falls"
            notes.append(
                f"Frequency {way} from {a['per_million']:.0f} to {b['per_million']:.0f} per "
                f"million ({a['slice']} -> {b['slice']}; intervals don't overlap)."
            )
        else:
            notes.append("Frequency differences are within the jackknife intervals.")
    t = card.get("change", {}).get("test", {})
    if "q" in t:
        if t["q"] < 0.05 and t["nn_overlap"] < 0.5 and t["stable"]:
            notes.append(
                f"Meaning change is supported: q={t['q']:.3g}, neighbour overlap "
                f"{t['nn_overlap']:.2f}, stable across seeds (rank {t['rank']}/{t['tested']})."
            )
        elif t["q"] < 0.05:
            notes.append(
                f"Change is significant (q={t['q']:.3g}) but not corroborated "
                f"(overlap {t['nn_overlap']:.2f}, stable={t['stable']}): treat with caution."
            )
        else:
            notes.append(f"No change beyond the shuffled null (q={t['q']:.2f}).")
    sr = card.get("surprisal", {}).get("rows", [])
    for r in sr:
        if r["n"] and r["word_bits"] > r["slice_bits"] + 3:
            notes.append(
                f"The LM finds '{card['word']}' unexpected in {r['slice']} "
                f"({r['word_bits']:.1f} vs {r['slice_bits']:.1f} bits/word)."
            )
    return notes


def build(
    ws: Workspace,
    word: str,
    slices: Sequence[str],
    model: str | None = None,
    null_runs: int = 3,
    min_count: int = 10,
) -> dict[str, Any]:
    word = word.casefold()
    known = ws.slices()
    for s in slices:
        if s not in known:
            raise InputError(f"unknown slice '{s}'", "see: semdrift slice ls")
    card: dict[str, Any] = {"word": word, "slices": list(slices)}
    card["frequency"] = frequency(ws, word, slices)
    if not any(r["count"] for r in card["frequency"]["rows"]):
        raise InputError(f"'{word}' does not occur in these slices")
    card["examples"] = examples(ws, word, slices)
    try:
        card["change"] = (
            change(ws, word, slices, null_runs, min_count)
            if len(slices) >= 2
            else {"skipped": "needs two or more slices"}
        )
    except Exception as e:
        card["change"] = _skip(e)
    if model is None:
        card["surprisal"] = {"skipped": "no --model given"}
    else:
        try:
            card["surprisal"] = surprisal(ws, word, slices, model)
        except Exception as e:
            card["surprisal"] = _skip(e)
    card["findings"] = verdict(card)
    card["methods"] = [
        METHODS[k] for k in ("frequency", "change", "surprisal") if "skipped" not in card[k]
    ]
    card["provenance"] = {
        "corpus_hash": ws.corpus_hash(),
        "slices": {s: known[s] for s in slices},
        "parameters": {"null_runs": null_runs, "min_count": min_count, "model": model},
        "versions": _versions(),
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    return card


def _f(v: Any) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    return str(v)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_none_\n"
    keys = list(rows[0])
    head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    return head + "".join("| " + " | ".join(_f(r[k]) for k in keys) + " |\n" for r in rows)


def markdown(card: dict[str, Any]) -> str:
    out = [f"# Evidence card: *{card['word']}*\n", f"Slices: {', '.join(card['slices'])}\n"]
    out.append("## Findings\n" + "".join(f"- {n}\n" for n in card["findings"]))
    out.append("## Frequency\n" + _table(card["frequency"]["rows"]))
    out.append("## Examples\n" + _table(card["examples"]["rows"]))
    ch = card["change"]
    if "skipped" in ch:
        out.append(f"## Semantic change\n_skipped: {ch['skipped']}_\n")
    else:
        t = ch["test"]
        test = _table([t]) if "skipped" not in t else f"_skipped: {t['skipped']}_\n"
        out.append("## Semantic change\n" + test + "\n" + _table(ch["timeline"]))
    su = card["surprisal"]
    if "skipped" in su:
        out.append(f"## LM surprisal\n_skipped: {su['skipped']}_\n")
    else:
        out.append(f"## LM surprisal ({su['model']})\n" + _table(su["rows"]))
    out.append("## Methods\n" + " ".join(card["methods"]) + "\n")
    p = card["provenance"]
    out.append(
        "## Provenance\n"
        + f"- corpus `{p['corpus_hash'][:16]}`, created {p['created']}\n"
        + "".join(f"- slice `{k}`: `{v}`\n" for k, v in p["slices"].items())
        + f"- parameters: {p['parameters']}\n"
        + "- versions: "
        + ", ".join(f"{k} {v}" for k, v in p["versions"].items())
        + "\n"
    )
    return "\n".join(out)
