"""Map doppel's errors to one-line messages and exit codes.

Core code raises `ValueError` (and its subclass `InvalidInputError`) for input
that doppel rejects. The CLI shows those as a usage error, exit code 2."""

from __future__ import annotations

import functools
from collections.abc import Callable

import typer


def map_errors[**P](command: Callable[P, None]) -> Callable[P, None]:
    """Wrap a Typer command so core errors exit cleanly. Typer reads the wrapped signature."""

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    return wrapper
