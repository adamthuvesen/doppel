"""SQL-backed source — read from DuckDB / Snowflake / Postgres via a `DatabaseUri`.

The driver layer is intentionally thin: per-vendor sample-pushdown SQL and
row-count probes live here as **pure functions** so they can be unit-tested
without a database. The actual driver invocation is one call site per scheme:
DuckDB runs in-process, Snowflake goes through `snowflake-connector-python`
(ConnectorX has no Snowflake source), and Postgres goes through
`pl.read_database_uri` with ConnectorX.

Pyright strictness: ConnectorX has no stubs. The narrow `Any` casts live at
the driver boundary; nothing else relaxes."""

from __future__ import annotations

import importlib
import threading
import urllib.parse
import warnings
from typing import Any, cast

import polars as pl

from doppel.errors import InvalidInputError, RowCountProbeError, WarehouseConnectionError
from doppel.locations import quote_identifier, quote_table_name, table_name_parts
from doppel.sources.spec import DatabaseUri, emit_redacted_log, redact_text

ROW_COUNT_THRESHOLD = 1_000_000
# Fraction samples (Postgres BERNOULLI, Snowflake SYSTEM) return approximate row
# counts; oversample by 5% and `LIMIT N` so users get the row count they asked
# for. Documented in docs/sql-connectors.md.
_FRACTION_OVERSAMPLE = 1.05
# Snowflake accepts a SEED from 0 to 2147483647 inclusive.
_SNOWFLAKE_SEED_MODULUS = 2**31
_SQL_EXTRA_HINT = 'Install it with: uv tool install "doppeldata[sql] @ git+https://github.com/adamthuvesen/doppel"'


def build_pushdown_sql(
    scheme: str,
    *,
    table: str | None,
    query: str | None,
    fit_rows: int | None,
    seed: int | None,
    row_count_estimate: int | None = None,
) -> tuple[str, bool]:
    """Build the read SQL, with a per-vendor sample clause when `fit_rows` is set.

    Returns `(sql, unseeded)`. `unseeded` tells callers that the vendor cannot
    honour the seed for this source, so they emit a determinism-caveat warning.

    Postgres `TABLESAMPLE` and Snowflake `SEED` apply only to a base table, so
    a `--query` read on those vendors uses an unseeded form."""
    base_query = _base_query(scheme, table=table, query=query)
    # fit_rows = 0 is the documented full-fit opt-in: read the whole table, unsampled.
    if not fit_rows:
        return base_query, False
    inner = base_query.strip().rstrip(";")
    if scheme == "duckdb":
        if seed is None:
            return f"SELECT * FROM ({inner}) AS _doppel_t USING SAMPLE {fit_rows} ROWS", False
        return (
            f"SELECT * FROM ({inner}) AS _doppel_t "
            f"USING SAMPLE reservoir({fit_rows} ROWS) REPEATABLE ({seed})",
            False,
        )
    if scheme == "snowflake":
        if table is None or seed is None:
            return f"SELECT * FROM ({inner}) SAMPLE ({fit_rows} ROWS)", seed is not None
        probability = _sample_percent(fit_rows, row_count_estimate)
        return (
            f"SELECT * FROM {quote_table_name(scheme, table)} "
            f"SAMPLE SYSTEM ({probability:.4f}) SEED ({seed % _SNOWFLAKE_SEED_MODULUS}) "
            f"LIMIT {fit_rows}",
            False,
        )
    if scheme in {"postgres", "postgresql"} and table is not None:
        probability = _sample_percent(fit_rows, row_count_estimate)
        seed_clause = f" REPEATABLE ({seed})" if seed is not None else ""
        return (
            f"SELECT * FROM {quote_table_name(scheme, table)} "
            f"TABLESAMPLE BERNOULLI ({probability:.4f}){seed_clause} LIMIT {fit_rows}",
            False,
        )
    sql = f"SELECT * FROM ({inner}) AS _doppel_t ORDER BY RANDOM() LIMIT {fit_rows}"
    return sql, True


def _sample_percent(fit_rows: int, row_count_estimate: int | None) -> float:
    """Sample percentage for a fraction sample, oversampled so `LIMIT` gets `fit_rows`."""
    if row_count_estimate is None or row_count_estimate <= 0:
        return 100.0
    return min(100.0, 100.0 * fit_rows * _FRACTION_OVERSAMPLE / row_count_estimate)


def build_count_sql(scheme: str, *, table: str | None, query: str | None) -> str:
    """Build the cheapest row-count query the vendor supports for table reads,
    falling back to `SELECT COUNT(*)` wrapping for `--query` reads."""
    if query is not None:
        inner = query.strip().rstrip(";")
        return f"SELECT COUNT(*) FROM ({inner}) AS _doppel_probe"
    if table is None:
        raise ValueError("build_count_sql requires either `table` or `query`")
    if scheme == "snowflake":
        return _snowflake_count_sql(table_name_parts(scheme, table))
    if scheme in {"postgres", "postgresql"}:
        # reltuples is the planner's estimate: -1 until the table is analysed.
        regclass = _string_literal(quote_table_name(scheme, table))
        return f"SELECT reltuples::BIGINT AS c FROM pg_class WHERE oid = {regclass}::regclass"
    return f"SELECT COUNT(*) AS c FROM {quote_table_name(scheme, table)}"


def _snowflake_count_sql(parts: list[str]) -> str:
    """ROW_COUNT in INFORMATION_SCHEMA is millisecond-cheap. It is NULL for a view.

    A `db.schema.table` name reads that database's INFORMATION_SCHEMA. Every form
    filters on the schema too, the session's current one when the name has none, so a
    table of the same name in another schema does not answer."""
    *qualifiers, name = parts
    if len(qualifiers) > 2:
        raise InvalidInputError(f"Snowflake table name has too many parts: {parts}")
    catalog = f"{quote_identifier(qualifiers[0])}." if len(qualifiers) == 2 else ""
    schema = _string_literal(qualifiers[-1]) if qualifiers else "CURRENT_SCHEMA()"
    return (
        "SELECT MAX(ROW_COUNT) AS c "
        f"FROM {catalog}INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_SCHEMA = {schema} AND TABLE_NAME = {_string_literal(name)}"
    )


def _string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read(
    spec: DatabaseUri,
    *,
    fit_rows: int | None,
    seed: int | None,
    timeout: int,
) -> pl.DataFrame:
    """Top-level entry: read `spec` into a Polars DataFrame.

    Applies sample pushdown (when `fit_rows` is set) and the row-count probe
    (Snowflake/Postgres only). DuckDB reads run in-process via `duckdb.connect`,
    Snowflake reads through `snowflake-connector-python`, and Postgres reads through
    `pl.read_database_uri` with ConnectorX.

    Wraps every driver exception as `WarehouseConnectionError` with the
    redacted URI so traceback chains never carry the raw password."""
    emit_redacted_log(spec)

    row_count_estimate: int | None = None
    if spec.scheme in {"snowflake", "postgres", "postgresql"}:
        row_count_estimate = _probe_row_count(spec, timeout=timeout)
        _enforce_row_count_threshold(spec, row_count_estimate, fit_rows)

    sql, unseeded = build_pushdown_sql(
        spec.scheme,
        table=spec.table,
        query=spec.query,
        fit_rows=fit_rows,
        seed=seed,
        row_count_estimate=row_count_estimate,
    )
    if unseeded and seed is not None:
        warnings.warn(
            f"the {spec.scheme!r} sample for this source cannot take a seed; "
            "output may not be byte-identical across runs",
            UserWarning,
            stacklevel=2,
        )

    return _run_sql(spec, sql, timeout=timeout)


def _run_sql(spec: DatabaseUri, sql: str, *, timeout: int) -> pl.DataFrame:
    if spec.scheme == "duckdb":
        return _read_duckdb(spec, sql, timeout=timeout)
    if spec.scheme == "snowflake":
        return _read_snowflake(spec, sql, timeout=timeout)
    return _read_via_connectorx(spec, sql, timeout=timeout)


def _base_query(scheme: str, *, table: str | None, query: str | None) -> str:
    """The user's selection — either `SELECT * FROM <table>` or the raw --query."""
    if query is not None:
        return query
    if table is not None:
        return f"SELECT * FROM {quote_table_name(scheme, table)}"
    raise ValueError("DatabaseUri must have either `table` or `query` set")


def _enforce_row_count_threshold(
    spec: DatabaseUri, estimate: int | None, fit_rows: int | None
) -> None:
    """If the table is huge or its size is unknown and the user didn't ask to
    sample, refuse to read.

    The 1M threshold is a safety net: streaming a billion-row Snowflake table
    to fit 25k rows is a $5k mistake. An unknown count (`None`: a view, or a
    Postgres table that was never analysed) counts as over the threshold.
    Users with intent can opt in via `--fit-rows N` (sample to N) or
    `--fit-rows 0` (fit on the whole thing)."""
    if estimate is not None and estimate <= ROW_COUNT_THRESHOLD:
        return
    size = "an unknown number of" if estimate is None else f"~{estimate:,}"
    if fit_rows is not None:
        if fit_rows == 0:
            warnings.warn(
                f"reading full table from {spec.uri} ({size} rows); "
                "this streams every row over the network — consider --fit-rows N",
                UserWarning,
                stacklevel=3,
            )
        return
    raise InvalidInputError(
        f"table at {spec.uri} has {size} rows; pass --fit-rows N to "
        "sample, or --fit-rows 0 to fit on the whole table "
        "(in a multi-table schema, set `fit_rows` on the table)"
    )


def _probe_row_count(spec: DatabaseUri, *, timeout: int) -> int | None:
    """Run the cheap row-count query. Returns None when the count is unknown:
    no row, NULL, a negative Postgres `reltuples`, or a Snowflake 0 (which a
    view or a missing catalog entry also returns)."""
    sql = build_count_sql(spec.scheme, table=spec.table, query=spec.query)
    try:
        df = _run_sql(spec, sql, timeout=timeout)
    except WarehouseConnectionError:
        # Re-raise — the user-facing error is more useful than a silent skip.
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RowCountProbeError(f"row-count probe failed for {spec.uri}: {exc}") from exc
    if df.height == 0 or df.width == 0:
        return None
    raw = df.row(0)[0]
    if raw is None:
        return None
    count = int(raw)
    if count < 0 or (count == 0 and spec.scheme == "snowflake"):
        return None
    return count


def _read_duckdb(spec: DatabaseUri, sql: str, *, timeout: int) -> pl.DataFrame:
    """In-process DuckDB read. Avoids the ConnectorX hop for the local case
    (and works without the `[sql]` extra installed)."""
    parsed = urllib.parse.urlsplit(spec.raw_uri)
    # `duckdb:///abs/path.db` → path='/abs/path.db'.
    # `duckdb://?table=T` (in-memory) → path='', netloc=''.
    db_path = parsed.path or ":memory:"
    if db_path == "/":
        db_path = ":memory:"

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - DuckDB is a top-level dep
        raise WarehouseConnectionError(
            f"duckdb is not installed; cannot read from {spec.uri}"
        ) from exc

    def _do_read() -> pl.DataFrame:
        try:
            con = duckdb.connect(db_path, read_only=db_path != ":memory:")
        except duckdb.Error as exc:
            raise WarehouseConnectionError(
                f"failed to open {spec.uri}: {str(exc).splitlines()[0]}"
            ) from None
        try:
            result = con.execute(sql)
            description = result.description
            rows = result.fetchall()
        except duckdb.Error as exc:
            # DuckDB errors span several lines (hint, query, caret); the first is the cause.
            raise WarehouseConnectionError(
                f"failed to read from {spec.uri}: {str(exc).splitlines()[0]}"
            ) from None
        finally:
            con.close()
        if not description:
            return pl.DataFrame()
        columns = [str(d[0]) for d in description]
        if not rows:
            return pl.DataFrame({c: [] for c in columns})
        # `orient='row'` lets polars infer the schema from the first non-null
        # row of each column. Nulls are preserved as `None`.
        return pl.DataFrame(rows, schema=columns, orient="row")

    return _run_with_timeout(_do_read, timeout=timeout, spec=spec)


def snowflake_connect_args(raw_uri: str, *, timeout: int) -> dict[str, Any]:
    """Keyword arguments for `snowflake.connector.connect` from
    `snowflake://user[:password]@account/database[/schema]?warehouse=WH&role=R`.

    Each query parameter passes through as a connection parameter, so a key-pair or
    SSO setup can name `authenticator` or `private_key_file`. `timeout` bounds the login
    and each statement; 0 means no limit."""
    parsed = urllib.parse.urlsplit(raw_uri)
    if not parsed.hostname:
        raise InvalidInputError("a Snowflake URI needs an account: snowflake://user@account/db")
    args: dict[str, Any] = {
        "account": parsed.hostname,
        "login_timeout": timeout,
        "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": timeout},
    }
    if parsed.username is not None:
        args["user"] = urllib.parse.unquote(parsed.username)
    if parsed.password is not None:
        args["password"] = urllib.parse.unquote(parsed.password)
    path = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if len(path) > 2:
        raise InvalidInputError("a Snowflake URI path is /database or /database/schema")
    args.update(zip(("database", "schema"), path, strict=False))
    for key, values in urllib.parse.parse_qs(parsed.query, keep_blank_values=True).items():
        if len(values) != 1:
            raise InvalidInputError(f"Snowflake URI parameter {key!r} is set more than once")
        args[key] = values[0]
    return args


def _read_snowflake(spec: DatabaseUri, sql: str, *, timeout: int) -> pl.DataFrame:
    """Snowflake read via `snowflake-connector-python`, fetched as one Arrow table."""
    try:
        import snowflake.connector as connector
    except ImportError as exc:
        raise InvalidInputError(
            f"the snowflake:// scheme requires the optional [sql] extra. {_SQL_EXTRA_HINT}"
        ) from exc
    connect_args = snowflake_connect_args(spec.raw_uri, timeout=max(timeout, 0))

    def _do_read() -> pl.DataFrame:
        try:
            with connector.connect(**connect_args) as connection:
                cursor = connection.cursor()
                cursor.execute(sql)
                frame = pl.from_arrow(cursor.fetch_arrow_all(force_return_table=True))
        except Exception as exc:
            detail = redact_text(str(exc), spec.raw_uri)
            raise WarehouseConnectionError(f"failed to read from {spec.uri}: {detail}") from None
        return cast("pl.DataFrame", frame)

    return _run_with_timeout(_do_read, timeout=timeout, spec=spec)


def _read_via_connectorx(spec: DatabaseUri, sql: str, *, timeout: int) -> pl.DataFrame:
    """Postgres read via Polars + ConnectorX.

    The `[sql]` extra is required here. We catch the ImportError raised by
    Polars when ConnectorX isn't present and re-raise a clear install hint."""
    try:
        # Importing connectorx eagerly gives us a deterministic error message
        # before Polars' own optional-import path runs.
        importlib.import_module("connectorx")
    except ImportError as exc:
        raise InvalidInputError(
            f"the {spec.scheme}:// scheme requires the optional [sql] extra. {_SQL_EXTRA_HINT}"
        ) from exc

    def _do_read() -> pl.DataFrame:
        try:
            # `Any` cast: connectorx has no stubs; the Polars call returns
            # `DataFrame` at runtime, which we re-narrow.
            df = cast(
                "Any",
                pl.read_database_uri(
                    query=sql,
                    uri=spec.raw_uri,
                    engine="connectorx",
                ),
            )
            return cast("pl.DataFrame", df)
        except Exception as exc:
            detail = redact_text(str(exc), spec.raw_uri)
            raise WarehouseConnectionError(f"failed to read from {spec.uri}: {detail}") from None

    return _run_with_timeout(_do_read, timeout=timeout, spec=spec)


def _run_with_timeout(
    fn: Any,
    *,
    timeout: int,
    spec: DatabaseUri,
) -> pl.DataFrame:
    """Python-side watchdog: run `fn` in a thread, raise if it doesn't return
    within `timeout` seconds. The thread keeps running in the background after
    a timeout (we can't cancel arbitrary C-extension calls), but the user gets
    a clean error and the CLI process exits.

    `timeout <= 0` disables the watchdog."""
    if timeout <= 0:
        return cast("pl.DataFrame", fn())
    result: list[pl.DataFrame] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(fn())
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise WarehouseConnectionError(
            f"connection or query timed out after {timeout}s against {spec.uri}; "
            "raise --connection-timeout if the warehouse is slow"
        )
    if error:
        # Re-raise the original exception (preserves WarehouseConnectionError
        # chain when the inner function already wrapped).
        raise error[0]
    return result[0]


__all__ = [
    "ROW_COUNT_THRESHOLD",
    "build_count_sql",
    "build_pushdown_sql",
    "read",
    "snowflake_connect_args",
]
