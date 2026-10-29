from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSUITE_CACHE_DIR", str(tmp_path / "cache"))
    from langsuite.cli.output import SETTINGS

    SETTINGS.json = False
    SETTINGS.debug = False
    SETTINGS.workspace = "."


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from langsuite.cli.app import main

    ws = tmp_path / "ws"
    assert main(["init", str(ws)]) == 0
    monkeypatch.chdir(ws)
    return ws
