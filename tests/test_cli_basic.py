from __future__ import annotations

import pytest

from semdrift._meta import __version__
from semdrift.cli.app import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_command_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["no-such-command"]) == 2
    assert "Traceback" not in capsys.readouterr().err
