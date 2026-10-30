from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from synth import write_change

from semdrift.change.score import bh, procrustes
from semdrift.cli.app import main

TRAIN = ["--seeds", "2", "--dim", "24", "--epochs", "8", "--min-count", "5"]


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 10_000))
def test_procrustes_recovers_rotation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    y = (x @ q).astype(np.float32)
    assert np.abs(procrustes(x, y) - q).max() < 1e-4


def test_bh_monotone_and_known() -> None:
    q = bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5])


@pytest.fixture
def changed(workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    pytest.importorskip("gensim")
    src = write_change(tmp_path / "c.jsonl")
    assert main(["ingest", str(src), "--pipeline", "basic"]) == 0
    assert main(["slice", "define", "early", "year < 1850"]) == 0
    assert main(["slice", "define", "late", "year >= 1850"]) == 0
    for s in ("early", "late"):
        assert main(["vec", "train", "--slice", s, *TRAIN]) == 0
    capsys.readouterr()
    return workspace


def test_injected_change_is_found(changed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--json", "change", "score", "--slices", "early,late", "--null-runs", "2", "--all"]
    assert main([*args, "--min-count", "5"]) == 0
    r = json.loads(capsys.readouterr().out)
    words = r["words"]
    assert words[0]["word"] == "pivot"
    assert words[0]["q"] < 0.05
    false_pos = [w["word"] for w in words if w["q"] < 0.05 and w["word"] != "pivot"]
    assert len(false_pos) <= 0.05 * r["tested_words"] + 1


def test_drift_timeline(changed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "change", "drift", "pivot", "--slices", "early,late"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["distance_from_first"] == 0.0
    assert rows[1]["distance_from_first"] > 0.3
    assert "c0w" in rows[0]["neighbours"]
    assert "c3w" in rows[1]["neighbours"]


@pytest.mark.parametrize(
    ("args", "code", "needle"),
    [
        (["change", "score", "--slices", "early"], 2, "at least two"),
        (["change", "score", "--slices", "early,early"], 2, "two different"),
        (["change", "score", "--slices", "early,nope"], 2, "unknown slice"),
        (["change", "score", "--slices", "early,late", "--min-count", "100000"], 3, "min-count"),
    ],
)
def test_change_errors(
    changed: Path, capsys: pytest.CaptureFixture[str], args: list[str], code: int, needle: str
) -> None:
    assert main(args) == code
    err = capsys.readouterr().err
    assert needle in err
    assert "Traceback" not in err


def test_no_change_corpus_flags_little(
    workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("gensim")
    src = write_change(tmp_path / "n.jsonl", seed=3, shift=False)
    assert main(["ingest", str(src), "--pipeline", "basic"]) == 0
    assert main(["slice", "define", "early", "year < 1850"]) == 0
    assert main(["slice", "define", "late", "year >= 1850"]) == 0
    for s in ("early", "late"):
        assert main(["vec", "train", "--slice", s, *TRAIN]) == 0
    capsys.readouterr()
    args = ["--json", "change", "score", "--slices", "early,late", "--null-runs", "2", "--all"]
    assert main([*args, "--min-count", "5"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["significant_q05"] <= max(1, 0.05 * r["tested_words"])
