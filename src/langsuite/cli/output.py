"""Output helpers: one place decides between JSON and human tables."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

err_console = Console(stderr=True, highlight=False)
out_console = Console(highlight=False)


@dataclass
class Settings:
    json: bool = False
    debug: bool = False
    workspace: str = "."


SETTINGS = Settings()


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def emit(data: Mapping[str, Any] | Sequence[Mapping[str, Any]], title: str | None = None) -> None:
    """Print data as JSON (with --json) or as a table / key-value list."""
    if SETTINGS.json:
        sys.stdout.write(json.dumps(_clean(data), ensure_ascii=False) + "\n")
        return
    if isinstance(data, Mapping):
        table = Table(title=title, show_header=False, box=None)
        table.add_column(style="bold")
        table.add_column()
        for k, v in data.items():
            if isinstance(v, list | dict):
                continue
            table.add_row(str(k), _fmt(v))
        out_console.print(table)
        return
    rows = list(data)
    if not rows:
        out_console.print("(no results)")
        return
    table = Table(title=title)
    for col in rows[0]:
        table.add_column(str(col), justify="right" if _is_num(rows[0][col]) else "left")
    for row in rows:
        table.add_row(*(_fmt(row[c]) for c in rows[0]))
    out_console.print(table)


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def info(msg: str) -> None:
    if not SETTINGS.json:
        err_console.print(msg)


def warn(msg: str) -> None:
    err_console.print(f"[yellow]warning:[/] {msg}")
