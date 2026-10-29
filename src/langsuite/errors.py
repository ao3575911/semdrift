"""User-facing errors. Each carries an exit code and a hint; no tracebacks unless --debug."""

from __future__ import annotations


class UserError(Exception):
    """Base class for errors shown to the user as `error:` + `hint:`."""

    exit_code = 1

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class UsageError(UserError):
    exit_code = 2


class InputError(UserError):
    exit_code = 3


class ModelError(UserError):
    exit_code = 4


class ResourceError(UserError):
    exit_code = 5


def missing_extra(module: str, extra: str) -> ModelError:
    return ModelError(
        f"optional dependency '{module}' is not installed",
        f'install it with: pip install "langsuite[{extra}]"',
    )
