from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FIXTURES
from hypothesis import given
from hypothesis import strategies as st

from langsuite.cli.app import main
from langsuite.corpus.pipeline import basic
from langsuite.corpus.workspace import hash_bytes


def run_json(capsys: pytest.CaptureFixture[str], *args: str) -> object:
    capsys.readouterr()
    code = main(["--json", *args])
    out = capsys.readouterr().out
    assert code == 0, out
    return json.loads(out)


def test_ingest_jsonl_golden(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    r = run_json(capsys, "ingest", str(FIXTURES / "mini.jsonl"), "--pipeline", "basic")
    assert r == {
        "files": 1,
        "docs_added": 4,
        "docs_skipped": 0,
        "tokens": 48,  # hand-counted: 12 + 14 + 11 + 11
        "cache_hits": 0,
        "pipeline": "basic/1",
    }
    info = run_json(capsys, "corpus", "info")
    assert isinstance(info, dict)
    assert info["documents"] == 4
    assert info["sentences"] == 9
    assert info["metadata"] == "year"


def test_reingest_hits_cache_and_skips(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ("ingest", str(FIXTURES / "mini.jsonl"), "--pipeline", "basic")
    run_json(capsys, *args)
    r = run_json(capsys, *args)
    assert isinstance(r, dict)
    assert r["cache_hits"] == 1
    assert r["docs_added"] == 0
    assert r["docs_skipped"] == 4


def test_conllu_skips_ranges(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_json(capsys, "ingest", str(FIXTURES / "mini.conllu"))
    info = run_json(capsys, "corpus", "info")
    assert isinstance(info, dict)
    assert (info["tokens"], info["sentences"]) == (6, 2)


def test_slices(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_json(capsys, "ingest", str(FIXTURES / "mini.jsonl"), "--pipeline", "basic")
    r = run_json(capsys, "slice", "define", "early", "year < 1900")
    assert isinstance(r, dict)
    assert r["documents"] == 2
    rows = run_json(capsys, "slice", "ls")
    assert rows == [{"slice": "early", "expr": "year < 1900", "documents": 2}]


@pytest.mark.parametrize(
    ("args", "code", "needle"),
    [
        (["ingest", "missing.txt", "--pipeline", "basic"], 3, "file not found"),
        (["ingest", "x.weird", "--pipeline", "basic"], 3, "can't tell the format"),
        (["ingest", "x.txt", "--pipeline", "nope"], 3, "unknown pipeline"),
        (["slice", "define", "s", "year <<< 3"], 3, "no corpus"),
        (["corpus", "info"], 3, "no corpus"),
    ],
)
def test_errors_are_clean(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    code: int,
    needle: str,
) -> None:
    assert main(args) == code
    err = capsys.readouterr().err
    assert needle in err
    assert "Traceback" not in err


def test_bad_slice_expr(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_json(capsys, "ingest", str(FIXTURES / "mini.jsonl"), "--pipeline", "basic")
    assert main(["slice", "define", "s", "year <<< 3"]) == 2
    assert "bad slice expression" in capsys.readouterr().err


def test_bad_jsonl_line(
    workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "ok"}\n{oops\n', encoding="utf-8")
    assert main(["ingest", str(bad), "--pipeline", "basic"]) == 3
    assert "bad.jsonl:2: invalid JSON" in capsys.readouterr().err


def test_no_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["corpus", "info"]) == 2


@given(st.text(max_size=300))
def test_basic_tokeniser_is_lossless_on_non_space(text: str) -> None:
    toks = basic(text)
    joined = "".join(t.form for t in toks)
    assert joined == "".join(ch for ch in text if not ch.isspace())
    sents = [t.sent for t in toks]
    assert sents == sorted(sents)


@given(st.lists(st.binary(max_size=20), max_size=5))
def test_hash_is_unambiguous(parts: list[bytes]) -> None:
    assert hash_bytes(*parts) != hash_bytes(b"".join(parts), b"x")
