"""Command registration. Each milestone adds a sub-app here."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    if getattr(app, "_langsuite_registered", False):
        return
    app._langsuite_registered = True  # type: ignore[attr-defined]
