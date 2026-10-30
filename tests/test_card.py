from __future__ import annotations

import json
from pathlib import Path

import pytest
from synth import write_change

from langsuite.cli.app import main

TRAIN = ["--seeds", "2", "--dim", "24", "--epochs", "8", "--min-count", "5"]
CARD = ["--slices", "early,late", "--null-runs", "2", "--min-count", "5"]


@pytest.fixture
def sliced(workspace: Path, tmp_path: Path) -> Path:
    src = write_change(tmp_path / "c.jsonl")
    assert main(["ingest", str(src), "--pipeline", "basic"]) == 0
    assert main(["slice", "define", "early", "year < 1850"]) == 0
    assert main(["slice", "define", "late", "year >= 1850"]) == 0
    return workspace


def test_card_without_vectors_or_model_degrades(
    sliced: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["--json", "card", "pivot", *CARD, "--model", "not-pulled"]) == 0
    card = json.loads(capsys.readouterr().out)
    assert "skipped" in card["change"]
    assert "skipped" in card["surprisal"]
    assert card["frequency"]["rows"][0]["count"] > 0
    assert card["examples"]["rows"]
    assert len(card["provenance"]["corpus_hash"]) >= 16
    assert (sliced / ".langsuite" / "cards" / "pivot.md").exists()


def test_card_reports_supported_change(sliced: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("gensim")
    for s in ("early", "late"):
        assert main(["vec", "train", "--slice", s, *TRAIN]) == 0
    capsys.readouterr()
    assert main(["--json", "card", "pivot", *CARD]) == 0
    card = json.loads(capsys.readouterr().out)
    assert card["change"]["test"]["q"] < 0.05
    assert any("supported" in f or "significant" in f for f in card["findings"])
    md = (sliced / ".langsuite" / "cards" / "pivot.md").read_text()
    assert "## Methods" in md
    assert "Procrustes" in md


def test_card_unknown_word_and_slice(sliced: Path) -> None:
    assert main(["card", "zzzz", "--slices", "early,late"]) == 3
    assert main(["card", "pivot", "--slices", "early,nope"]) == 3
