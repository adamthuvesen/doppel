"""Phase 0 scaffold smoke test: every documented subcommand is reachable from --help."""

from __future__ import annotations

from typer.testing import CliRunner

from doppel import __version__
from doppel.cli import app

runner = CliRunner()


def test_root_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("gen", "fit", "sample", "diff", "schema"):
        assert cmd in result.stdout, f"`{cmd}` missing from root help"


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_schema_subapp_help() -> None:
    result = runner.invoke(app, ["schema", "--help"])
    assert result.exit_code == 0
    assert "infer" in result.stdout
    assert "check" in result.stdout
