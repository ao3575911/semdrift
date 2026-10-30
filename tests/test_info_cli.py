from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FIXTURES

from semdrift.cli.app import main


@pytest.fixture
def sliced(workspace: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    assert main(["ingest", str(FIXTURES / "mini.jsonl"), "--pipeline", "basic"]) == 0
    assert main(["slice", "define", "early", "year < 1900"]) == 0
    assert main(["slice", "define", "late", "year >= 1900"]) == 0
    capsys.readouterr()
    return workspace


def test_summary_json(sliced: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "info", "summary", "-s", "early"]) == 0
    rows = json.loads(capsys.readouterr().out)
    by = {r["measure"]: r for r in rows}
    assert by["tokens"]["value"] == 22  # (12 - 2) + (14 - 2) after dropping punctuation
    assert by["TTR"]["value"] == pytest.approx(by["types"]["value"] / 22)


def test_divergence_json(sliced: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "info", "divergence", "early", "late"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert 0 < r["value"] < 1
    assert r["top_contributors"][0]["word"] == "news"


@pytest.mark.parametrize(
    ("args", "code", "needle"),
    [
        (["info", "summary", "-s", "nope"], 2, "unknown slice 'nope'"),
        (["info", "summary", "--unit", "lemma"], 3, "no 'lemma' tokens"),
        (["info", "summary", "--unit", "xyz"], 2, "unknown unit"),
        (["info", "divergence", "early", "late", "--measure", "x"], 2, "unknown measure"),
        (["info", "divergence", "early", "late", "--measure", "kl", "--alpha", "0"], 2, "alpha"),
    ],
)
def test_info_errors(
    sliced: Path, capsys: pytest.CaptureFixture[str], args: list[str], code: int, needle: str
) -> None:
    assert main(args) == code
    err = capsys.readouterr().err
    assert needle in err
    assert "Traceback" not in err
