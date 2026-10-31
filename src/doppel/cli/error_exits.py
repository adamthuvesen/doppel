"""Map doppel's errors to one-line messages and exit codes.

- `ValueError`, which covers `InvalidInputError` and Pydantic's `ValidationError`:
  the input is one doppel rejects. Usage error, exit code 2.
- Any other `DoppelError`: a failure at run time. `error: <message>` on stderr,
  exit code 1, no traceback."""

from __future__ import annotations

import functools
from collections.abc import Callable

import typer

from doppel.errors import DoppelError

RUNTIME_ERROR_EXIT_CODE = 1


def map_errors[**P](command: Callable[P, None]) -> Callable[P, None]:
    """Wrap a Typer command so core errors exit cleanly. Typer reads the wrapped signature."""

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except DoppelError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=RUNTIME_ERROR_EXIT_CODE) from None

    return wrapper
