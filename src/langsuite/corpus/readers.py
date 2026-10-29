"""Readers for plain text, JSONL, CSV and CoNLL-U. Each yields RawDoc records."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langsuite.errors import InputError

FORMATS = ("txt", "jsonl", "csv", "conllu")
Scalar = str | int | float | bool | None


@dataclass
class RawToken:
    sent: int
    form: str
    lemma: str | None = None
    upos: str | None = None


@dataclass
class RawDoc:
    doc_id: str
    text: str | None
    meta: dict[str, Scalar] = field(default_factory=dict[str, Scalar])
    tokens: list[RawToken] | None = None  # pre-tokenised input (CoNLL-U)


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    aliases = {"text": "txt", "md": "txt", "ndjson": "jsonl", "conll": "conllu", "tsv": "csv"}
    fmt = aliases.get(suffix, suffix)
    if fmt not in FORMATS:
        raise InputError(
            f"can't tell the format of {path.name}",
            f"pass --format with one of: {', '.join(FORMATS)}",
        )
    return fmt


def read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise InputError(f"file not found: {path}") from None
    except IsADirectoryError:
        raise InputError(f"{path} is a directory", "pass files, or a glob like dir/*.txt") from None
    except PermissionError:
        raise InputError(f"permission denied: {path}") from None
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            break
        return text
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        raise InputError(
            f"{path.name} is not valid UTF-8, UTF-16 or cp1252 text",
            "convert it to UTF-8 first (e.g. iconv -f <enc> -t utf-8)",
        ) from None


def _scalar(v: object) -> Scalar:
    if v is None or isinstance(v, str | int | float | bool):
        return v
    return json.dumps(v, ensure_ascii=False)


def iter_docs(path: Path, fmt: str, text_field: str = "text") -> Iterator[RawDoc]:
    if fmt == "txt":
        yield RawDoc(path.stem, read_text(path))
    elif fmt == "jsonl":
        yield from _jsonl(path, text_field)
    elif fmt == "csv":
        yield from _csv(path, text_field)
    elif fmt == "conllu":
        yield from _conllu(path)
    else:
        raise InputError(f"unknown format '{fmt}'", f"use one of: {', '.join(FORMATS)}")


def _jsonl(path: Path, text_field: str) -> Iterator[RawDoc]:
    for n, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise InputError(f"{path.name}:{n}: invalid JSON ({e.msg})") from None
        if not isinstance(rec, dict):
            raise InputError(f"{path.name}:{n}: each line must be a JSON object")
        text = rec.get(text_field)
        if not isinstance(text, str):
            raise InputError(
                f"{path.name}:{n}: no string field '{text_field}'",
                "set the field name with --text-field",
            )
        doc_id = str(rec.get("id", f"{path.stem}:{n}"))
        meta = {str(k): _scalar(v) for k, v in rec.items() if k not in (text_field, "id")}
        yield RawDoc(doc_id, text, meta)


def _csv(path: Path, text_field: str) -> Iterator[RawDoc]:
    text = read_text(path)
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    if reader.fieldnames is None or text_field not in reader.fieldnames:
        raise InputError(
            f"{path.name}: no column '{text_field}'",
            f"columns are: {', '.join(reader.fieldnames or [])}; set --text-field",
        )
    for n, row in enumerate(reader, 2):
        doc_id = row.get("id") or f"{path.stem}:{n}"
        meta: dict[str, Scalar] = {
            k: coerce(v) for k, v in row.items() if k not in (text_field, "id") and k
        }
        yield RawDoc(doc_id, row[text_field] or "", meta)


def coerce(v: str | None) -> Scalar:
    if v is None or v == "":
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _conllu(path: Path) -> Iterator[RawDoc]:
    doc_id = path.stem
    toks: list[RawToken] = []
    sent = 0
    in_sent = False
    for n, line in enumerate(read_text(path).splitlines(), 1):
        line = line.rstrip("\n")
        if line.startswith("# newdoc"):
            if toks:
                yield RawDoc(doc_id, None, tokens=toks)
                toks, sent = [], 0
            _, _, rest = line.partition("=")
            doc_id = rest.strip() or f"{path.stem}:{n}"
            continue
        if line.startswith("#"):
            continue
        if not line.strip():
            if in_sent:
                sent += 1
                in_sent = False
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            raise InputError(f"{path.name}:{n}: expected 10 tab-separated CoNLL-U columns")
        if not cols[0].isdigit():  # skip multiword ranges (1-2) and empty nodes (1.1)
            continue
        in_sent = True
        toks.append(RawToken(sent, cols[1], _u(cols[2]), _u(cols[3])))
    if toks:
        yield RawDoc(doc_id, None, tokens=toks)


def _u(v: str) -> str | None:
    return None if v == "_" else v
