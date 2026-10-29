from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from synth import A, write_clusters

from langsuite.cli.app import main
from langsuite.geom.space import Space, normalise

pytest.importorskip("gensim")


@pytest.fixture
def trained(workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    src = write_clusters(tmp_path / "c.jsonl")
    assert main(["ingest", str(src), "--pipeline", "basic"]) == 0
    assert (
        main(["vec", "train", "--seeds", "2", "--dim", "20", "--epochs", "10", "--min-count", "2"])
        == 0
    )
    capsys.readouterr()
    return workspace


def test_neighbours_find_cluster(trained: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "vec", "nn", "alpha0", "-k", "5"]) == 0
    r = json.loads(capsys.readouterr().out)
    words = [n["word"] for n in r["neighbours"]]
    assert sum(w in A for w in words) >= 4
    assert 0 <= r["stability_jaccard"] <= 1


def test_training_is_deterministic(trained: Path) -> None:
    d = trained / ".langsuite" / "vec" / "_all"
    first = Space.load(d / "seed0.npz").vectors.copy()
    assert (
        main(["vec", "train", "--seeds", "2", "--dim", "20", "--epochs", "10", "--min-count", "2"])
        == 0
    )
    assert np.array_equal(first, Space.load(d / "seed0.npz").vectors)


def test_csls_and_axis(trained: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "vec", "nn", "alpha0", "--csls", "-k", "5"]) == 0
    capsys.readouterr()
    assert (
        main(["--json", "vec", "axis", "alpha3", "beta3", "--pos", "alpha1", "--neg", "beta1"]) == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["word"] == "alpha3"


def test_unknown_word_suggests(trained: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["vec", "nn", "alpah0"]) == 3
    err = capsys.readouterr().err
    assert "did you mean" in err
    assert "alpha0" in err


def test_untrained_slice(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["vec", "nn", "x"]) == 3
    assert "vec train" in capsys.readouterr().err


def test_normalise_zero_row() -> None:
    m = normalise(np.zeros((2, 3), np.float32))
    assert not np.isnan(m).any()
