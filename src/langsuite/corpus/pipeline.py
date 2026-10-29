"""Tokenisation pipelines: a dependency-free 'basic' one, spaCy, and Stanza."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass

from langsuite.corpus.readers import RawToken
from langsuite.errors import InputError, ModelError, missing_extra

_APOS = "'\u2019"
_TOKEN = re.compile(r"\w+(?:[-" + _APOS + r"]\w+)*|[^\w\s]", re.UNICODE)
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]" + "\u201d\u2019" + r"]*\s+")


@dataclass(frozen=True)
class Pipeline:
    id: str
    run: Callable[[str], list[RawToken]]


def basic(text: str) -> list[RawToken]:
    out: list[RawToken] = []
    for s, sent in enumerate(p for p in _SENT_END.split(text) if p.strip()):
        out.extend(RawToken(s, m.group(0)) for m in _TOKEN.finditer(sent))
    return out


def resolve(spec: str) -> Pipeline:
    """Turn 'auto' | 'basic' | 'spacy:MODEL' | 'stanza:LANG' into a pipeline."""
    if spec == "basic":
        return Pipeline("basic/1", basic)
    if spec == "auto":
        if importlib.util.find_spec("spacy") and importlib.util.find_spec("en_core_web_sm"):
            return _spacy("en_core_web_sm")
        return Pipeline("basic/1", basic)
    kind, _, arg = spec.partition(":")
    if kind == "spacy" and arg:
        return _spacy(arg)
    if kind == "stanza" and arg:
        return _stanza(arg)
    raise InputError(
        f"unknown pipeline '{spec}'", "use auto, basic, spacy:<model> or stanza:<lang>"
    )


def _spacy(model: str) -> Pipeline:
    try:
        import spacy  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise missing_extra("spacy", "spacy") from None
    try:
        nlp = spacy.load(model, disable=["ner", "parser"])
    except OSError:
        raise ModelError(
            f"spaCy model '{model}' is not installed",
            f"install it once with: python -m spacy download {model}",
        ) from None
    if "senter" in nlp.pipe_names or "sentencizer" not in nlp.pipe_names:
        try:
            nlp.enable_pipe("senter")
        except ValueError:
            nlp.add_pipe("sentencizer")
    nlp.max_length = 10_000_000

    def run(text: str) -> list[RawToken]:
        doc = nlp(text)
        return [
            RawToken(s, t.text, t.lemma_ or None, t.pos_ or None)
            for s, sent in enumerate(doc.sents)
            for t in sent
            if not t.is_space
        ]

    version = str(nlp.meta.get("version", "?"))
    return Pipeline(f"spacy/{_dist_version('spacy')}/{model}/{version}", run)


def _stanza(lang: str) -> Pipeline:
    try:
        import stanza  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise missing_extra("stanza", "stanza") from None
    try:
        nlp = stanza.Pipeline(
            lang, processors="tokenize,pos,lemma", download_method=None, verbose=False
        )
    except Exception as e:
        raise ModelError(
            f"Stanza model for '{lang}' could not be loaded ({type(e).__name__})",
            f"download it once with: python -c \"import stanza; stanza.download('{lang}')\"",
        ) from None

    def run(text: str) -> list[RawToken]:
        doc = nlp(text)
        return [
            RawToken(s, w.text, w.lemma, w.upos)
            for s, sent in enumerate(doc.sentences)
            for w in sent.words
        ]

    return Pipeline(f"stanza/{_dist_version('stanza')}/{lang}", run)


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "?"
