"""Unit tests for the SourceSpec / SinkSpec parser.

Covers Section 9 (URI parser) and Section 10 (auth) of the SQL-connectors
change. End-to-end DuckDB reads live in test_sql_duckdb.py; mocked driver
behaviour lives in test_sql_pushdown.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer

from doppel.sources.spec import (
    DatabaseUri,
    DuckDbSink,
    FilePath,
    expand_env_vars,
    parse_sink_spec,
    parse_source_spec,
    redact_uri,
)


def test_parse_file_path(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")
    spec = parse_source_spec(str(p), table=None, query=None, password_cmd=None)
    assert isinstance(spec, FilePath)
    assert spec.path == p


def test_parse_file_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="does not exist"):
        parse_source_spec(str(tmp_path / "missing.csv"), table=None, query=None, password_cmd=None)


def test_parse_file_path_with_table_rejected(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(typer.BadParameter, match="apply only to URI sources"):
        parse_source_spec(str(p), table="users", query=None, password_cmd=None)


def test_parse_database_uri_with_table() -> None:
    spec = parse_source_spec(
        "snowflake://user@account/db/schema?warehouse=WH",
        table="USERS",
        query=None,
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)
    assert spec.scheme == "snowflake"
    assert spec.table == "USERS"
    assert spec.query is None
    assert "WH" in spec.uri


def test_parse_database_uri_with_query() -> None:
    spec = parse_source_spec(
        "postgres://user@host/db",
        table=None,
        query="SELECT * FROM users WHERE active = true",
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)
    assert spec.scheme == "postgres"
    assert spec.query == "SELECT * FROM users WHERE active = true"
    assert spec.table is None


def test_database_uri_both_table_and_query_rejected() -> None:
    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        parse_source_spec(
            "snowflake://user@a/db/s?warehouse=WH",
            table="T",
            query="SELECT 1",
            password_cmd=None,
        )


def test_database_uri_neither_table_nor_query_rejected() -> None:
    with pytest.raises(typer.BadParameter, match="exactly one of --table or --query"):
        parse_source_spec(
            "snowflake://user@a/db/s?warehouse=WH",
            table=None,
            query=None,
            password_cmd=None,
        )


def test_unknown_scheme_rejected() -> None:
    with pytest.raises(typer.BadParameter, match="unsupported URI scheme"):
        parse_source_spec(
            "bigquery://proj/ds/tbl",
            table="users",
            query=None,
            password_cmd=None,
        )


def test_env_var_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_USER", "adam")
    monkeypatch.setenv("SF_PASS", "hunter2")
    spec = parse_source_spec(
        "snowflake://${SF_USER}:${SF_PASS}@account/db/schema?warehouse=WH",
        table="USERS",
        query=None,
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)
    assert "adam" in spec.raw_uri
    assert "hunter2" in spec.raw_uri
    assert "hunter2" not in spec.uri
    assert ":***@" in spec.uri


def test_env_var_missing_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SF_PASS", raising=False)
    with pytest.raises(typer.BadParameter, match="SF_PASS"):
        parse_source_spec(
            "snowflake://user:${SF_PASS}@account/db/s?warehouse=WH",
            table="USERS",
            query=None,
            password_cmd=None,
        )


def test_password_redaction() -> None:
    spec = parse_source_spec(
        "postgres://user:hunter2@host/db",
        table="users",
        query=None,
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)
    assert ":***@" in spec.uri
    assert "hunter2" not in spec.uri
    assert "hunter2" in spec.raw_uri


def test_redact_uri_no_password_unchanged() -> None:
    uri = "snowflake://user@account/db/s?warehouse=WH"
    assert redact_uri(uri) == uri


def test_redact_uri_with_password() -> None:
    redacted = redact_uri("postgres://user:secret@host:5432/db")
    assert "secret" not in redacted
    assert ":***@" in redacted


def test_expand_env_vars_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # No ${} markers means unchanged.
    s = "snowflake://user@account/db/s?warehouse=WH"
    assert expand_env_vars(s) == s


def test_expand_env_vars_only_braced_form(monkeypatch: pytest.MonkeyPatch) -> None:
    # `$VAR` without braces should NOT be expanded — passwords can contain $.
    monkeypatch.setenv("FOO", "bar")
    assert expand_env_vars("$FOO is literal") == "$FOO is literal"


def test_uri_embedded_password_emits_warning(capfd: pytest.CaptureFixture[str]) -> None:
    parse_source_spec(
        "postgres://user:hunter2@host/db",
        table="users",
        query=None,
        password_cmd=None,
    )
    err = capfd.readouterr().err
    assert "shell history" in err.lower()


def test_password_cmd_overrides_uri_password(capfd: pytest.CaptureFixture[str]) -> None:
    spec = parse_source_spec(
        "postgres://user:wrongpass@host/db",
        table="users",
        query=None,
        password_cmd="echo secret",
    )
    assert isinstance(spec, DatabaseUri)
    assert "secret" in spec.raw_uri
    assert "wrongpass" not in spec.raw_uri
    err = capfd.readouterr().err
    assert "--password-cmd" in err.lower()


def test_password_cmd_failure_raises() -> None:
    with pytest.raises(typer.BadParameter, match="--password-cmd"):
        parse_source_spec(
            "postgres://user@host/db",
            table="users",
            query=None,
            password_cmd="false",
        )


def test_password_cmd_empty_output_raises() -> None:
    with pytest.raises(typer.BadParameter, match="no output"):
        parse_source_spec(
            "postgres://user@host/db",
            table="users",
            query=None,
            password_cmd="true",
        )


def test_password_cmd_stderr_quoted() -> None:
    with pytest.raises(typer.BadParameter) as exc_info:
        parse_source_spec(
            "postgres://user@host/db",
            table="users",
            query=None,
            # `sh -c` so we can write to stderr deterministically.
            password_cmd="sh -c 'echo my-error-msg >&2; exit 1'",
        )
    assert "my-error-msg" in str(exc_info.value)


def test_postgresql_alias_accepted() -> None:
    spec = parse_source_spec(
        "postgresql://user@host/db",
        table="users",
        query=None,
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)
    assert spec.scheme == "postgresql"


# ---------- SinkSpec ----------


def test_parse_sink_file_path() -> None:
    spec = parse_sink_spec("out.parquet")
    assert isinstance(spec, FilePath)
    assert spec.path == Path("out.parquet")


def test_parse_sink_duckdb_uri() -> None:
    spec = parse_sink_spec("duckdb:///tmp/synth.db?table=users")
    assert isinstance(spec, DuckDbSink)
    assert spec.path == Path("/tmp/synth.db")
    assert spec.table == "users"


def test_duckdb_sink_requires_table() -> None:
    with pytest.raises(typer.BadParameter, match="table=NAME"):
        parse_sink_spec("duckdb:///tmp/synth.db")


def test_duckdb_sink_requires_path() -> None:
    with pytest.raises(typer.BadParameter, match="file path"):
        parse_sink_spec("duckdb://?table=t")


def test_snowflake_sink_rejected() -> None:
    with pytest.raises(typer.BadParameter, match="Snowflake sinks are not supported"):
        parse_sink_spec("snowflake://user@a/db/s?warehouse=WH")


def test_postgres_sink_rejected() -> None:
    with pytest.raises(typer.BadParameter, match="Postgres sinks are not supported"):
        parse_sink_spec("postgres://user@host/db")


def test_unknown_sink_scheme_rejected() -> None:
    with pytest.raises(typer.BadParameter, match="unknown sink URI scheme"):
        parse_sink_spec("bigquery://x")


def test_info_log_redacted(capfd: pytest.CaptureFixture[str]) -> None:
    """Ensure emit_redacted_log goes to stderr with the redacted form only."""
    from doppel.sources.spec import emit_redacted_log

    spec = parse_source_spec(
        "postgres://user:secret@host/db",
        table="users",
        query=None,
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)
    # Drain any warnings already emitted.
    capfd.readouterr()
    emit_redacted_log(spec)
    err = capfd.readouterr().err
    assert "secret" not in err
    assert ":***@" in err
    assert "reading from" in err


def test_python_version_sanity() -> None:
    """Sanity check that the tests run on the supported interpreter."""
    assert sys.version_info >= (3, 11)
