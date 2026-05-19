"""`doppel doctor` — environment check: versions, extras, importability."""

from __future__ import annotations

import importlib
import platform
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.table import Table

from doppel import __version__

console = Console()


@dataclass(frozen=True)
class _Probe:
    label: str
    module: str
    required: bool


_CORE_DEPS = (
    _Probe("polars", "polars", required=True),
    _Probe("duckdb", "duckdb", required=True),
    _Probe("scikit-learn", "sklearn", required=True),
    _Probe("scipy", "scipy", required=True),
    _Probe("numpy", "numpy", required=True),
    _Probe("pydantic", "pydantic", required=True),
    _Probe("typer", "typer", required=True),
    _Probe("rich", "rich", required=True),
)

_EXTRAS = (
    _Probe("pii / presidio-analyzer", "presidio_analyzer", required=False),
    _Probe("pii / faker", "faker", required=False),
)


def run() -> None:
    """Report environment health and exit non-zero if a required dep is broken."""
    console.print(f"[bold]doppel[/] {__version__}")
    console.print(
        f"[dim]python {platform.python_version()} ({platform.system()} {platform.machine()})[/]"
    )

    table = Table(title=None, show_header=True, header_style="bold")
    table.add_column("dep", no_wrap=True)
    table.add_column("status")
    table.add_column("version")
    table.add_column("kind")

    any_required_missing = False
    for probe in (*_CORE_DEPS, *_EXTRAS):
        status_ok, version_str = _probe(probe.module)
        kind = "core" if probe.required else "extra"
        if status_ok:
            mark = "[green]ok[/]"
        elif probe.required:
            mark = "[red]missing[/]"
            any_required_missing = True
        else:
            mark = "[yellow]not installed[/]"
        table.add_row(probe.label, mark, version_str, kind)

    console.print(table)

    if any_required_missing:
        console.print(
            "[red]one or more required dependencies are missing. "
            "reinstall with `uv tool install doppeldata` or `pip install doppeldata`.[/]"
        )
        raise typer.Exit(code=1)


def _probe(module: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False, "-"
    version = getattr(mod, "__version__", None)
    if version is None:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as md_version

        try:
            version = md_version(module.replace("_", "-"))
        except PackageNotFoundError:
            version = "?"
    return True, str(version)


if __name__ == "__main__":  # pragma: no cover — script convenience
    typer.run(run)


__all__ = ["run"]
