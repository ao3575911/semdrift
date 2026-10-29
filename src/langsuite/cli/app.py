"""Top-level CLI. Every command routes errors through `main()`."""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from langsuite._meta import BIN, NAME, __version__
from langsuite.cli.output import SETTINGS, err_console
from langsuite.errors import ResourceError, UserError

app = typer.Typer(
    name=BIN,
    help=f"{NAME}: information, geometry and time in one terminal.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"{BIN} {__version__}")
        raise typer.Exit


@app.callback()
def root(
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks.")] = False,
    workspace: Annotated[
        str,
        typer.Option(
            "--workspace", "-w", help="Workspace directory.", envvar="LANGSUITE_WORKSPACE"
        ),
    ] = ".",
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    SETTINGS.json = json_out
    SETTINGS.debug = debug
    SETTINGS.workspace = workspace


def _register() -> None:
    from langsuite.cli import commands

    commands.register(app)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns an exit code; never leaks tracebacks unless --debug."""
    args = sys.argv[1:] if argv is None else argv
    debug = "--debug" in args
    _register()
    try:
        app(args=args, standalone_mode=False)
    except typer.Exit as e:
        return e.exit_code
    except typer.Abort:
        err_console.print("aborted")
        return 130
    except KeyboardInterrupt:
        err_console.print("interrupted")
        return 130
    except UserError as e:
        return _report(e, debug)
    except MemoryError:
        return _report(
            ResourceError("out of memory", "use a smaller --slice or lower --batch"), debug
        )
    except Exception as e:  # click usage errors and unexpected failures
        # Typer vendors click; detect its exceptions by shape, not import path.
        show = getattr(e, "show", None)
        code = getattr(e, "exit_code", None)
        if callable(show) and isinstance(code, int):
            show()
            return code
        if debug:
            raise
        err_console.print(f"[red]error:[/] unexpected failure: {type(e).__name__}: {e}")
        err_console.print("[dim]hint:[/] rerun with --debug and report the traceback")
        return 70
    return 0


def _report(e: UserError, debug: bool) -> int:
    if debug:
        raise e
    err_console.print(f"[red]error:[/] {e.message}")
    if e.hint:
        err_console.print(f"[dim]hint:[/] {e.hint}")
    return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
