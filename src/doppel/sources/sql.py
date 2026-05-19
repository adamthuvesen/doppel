"""SQL-backed source — read from DuckDB / Snowflake / Postgres via a `DatabaseUri`.

The driver layer is intentionally thin: per-vendor sample-pushdown SQL and
row-count probes live here as **pure functions** so they can be unit-tested
without a database. The actual driver invocation is one call site per scheme
(DuckDB → in-process; Snowflake/Postgres → `pl.read_database_uri`).

Pyright strictness: ConnectorX has no stubs. The narrow `Any` casts live at
the driver boundary; nothing else relaxes."""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from doppel.sources.errors import RowCountProbeError, WarehouseConnectionError
from doppel.sources.spec import DatabaseUri, emit_redacted_log

if TYPE_CHECKING:
    pass

ROW_COUNT_THRESHOLD = 1_000_000
# Postgres TABLESAMPLE BERNOULLI returns approximate row counts; oversample
# by 5% and `LIMIT N` client-side so users get exactly the row count they
# asked for. Documented in docs/sql-connectors.md.
_POSTGRES_OVERSAMPLE = 1.05


def build_pushdown_sql(
    scheme: str,
    base_query: str,
    fit_rows: int | None,
    seed: int | None,
    row_count_estimate: int | None = None,
) -> tuple[str, bool]:
    """Wrap `base_query` in a per-vendor sample clause when `fit_rows` is set.

    Returns `(sql, ansi_fallback_used)`. The fallback flag tells callers to
    emit a determinism-caveat warning."""
    if fit_rows is None:
        return base_query, False
    inner = base_query.strip().rstrip(";")
    if scheme == "snowflake":
        seed_clause = f" SEED ({seed})" if seed is not None else ""
        sql = f"SELECT * FROM ({inner}) SAMPLE ({fit_rows} ROWS){seed_clause}"
        return sql, False
    if scheme in {"postgres", "postgresql"}:
        # Compute probability from row-count estimate; clamp to (0, 100].
        if row_count_estimate and row_count_estimate > 0:
            probability = min(100.0, 100.0 * fit_rows * _POSTGRES_OVERSAMPLE / row_count_estimate)
        else:
            probability = 100.0
        seed_clause = f" REPEATABLE({seed})" if seed is not None else ""
        # Quote the probability as a fixed number (Postgres requires it inline).
        sql = (
            f"SELECT * FROM ({inner}) AS _doppel_t "
            f"TABLESAMPLE BERNOULLI({probability:.4f}){seed_clause} "
            f"LIMIT {fit_rows}"
        )
        return sql, False
    if scheme == "duckdb":
        seed_clause = f" REPEATABLE {seed}" if seed is not None else ""
        sql = (
            f"SELECT * FROM ({inner}) AS _doppel_t "
            f"USING SAMPLE {fit_rows} ROWS ({seed_clause.strip()})"
            if seed is not None
            else f"SELECT * FROM ({inner}) AS _doppel_t USING SAMPLE {fit_rows} ROWS"
        )
        return sql, False
    # ANSI fallback for future / unknown vendors. Determinism depends on the
    # vendor's RANDOM() seedability — we cannot guarantee it. Caller warns.
    sql = f"SELECT * FROM ({inner}) AS _doppel_t ORDER BY RANDOM() LIMIT {fit_rows}"
    return sql, True


def build_count_sql(scheme: str, *, table: str | None, query: str | None) -> str:
    """Build the cheapest row-count query the vendor supports for table reads,
    falling back to `SELECT COUNT(*)` wrapping for `--query` reads."""
    if query is not None:
        inner = query.strip().rstrip(";")
        return f"SELECT COUNT(*) FROM ({inner}) AS _doppel_probe"
    if table is None:
        raise ValueError("build_count_sql requires either `table` or `query`")
    if scheme == "snowflake":
        # ROW_COUNT in INFORMATION_SCHEMA is millisecond-cheap. Strip schema
        # qualification if the user passed `SCHEMA.TABLE`.
        last_segment = table.split(".")[-1].upper()
        return (
            "SELECT COALESCE(MAX(ROW_COUNT), 0) AS c "
            "FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE UPPER(TABLE_NAME) = '{last_segment}'"
        )
    if scheme in {"postgres", "postgresql"}:
        # reltuples is the planner's estimate; cheap and good enough for the
        # safety net. Quoting with `::regclass` resolves schema-qualified names.
        return f"SELECT reltuples::BIGINT AS c FROM pg_class WHERE oid = '{table}'::regclass"
    if scheme == "duckdb":
        return f"SELECT COUNT(*) AS c FROM {table}"
    return f"SELECT COUNT(*) AS c FROM {table}"


def read(
    spec: DatabaseUri,
    *,
    fit_rows: int | None,
    seed: int | None,
    timeout: int,
) -> pl.DataFrame:
    """Top-level entry: read `spec` into a Polars DataFrame.

    Applies sample pushdown (when `fit_rows` is set) and the row-count probe
    (Snowflake/Postgres only). DuckDB reads run in-process via `duckdb.connect`;
    Snowflake/Postgres reads go through `pl.read_database_uri` with ConnectorX.

    Wraps every driver exception as `WarehouseConnectionError` with the
    redacted URI so traceback chains never carry the raw password."""
    emit_redacted_log(spec)
    base_query = _base_query(spec)

    row_count_estimate: int | None = None
    if spec.scheme in {"snowflake", "postgres", "postgresql"}:
        row_count_estimate = _probe_row_count(spec, timeout=timeout)
        if row_count_estimate is not None:
            _enforce_row_count_threshold(spec, row_count_estimate, fit_rows)

    sql, fallback = build_pushdown_sql(spec.scheme, base_query, fit_rows, seed, row_count_estimate)
    if fallback and seed is not None:
        warnings.warn(
            f"using ANSI sample fallback for scheme {spec.scheme!r}; determinism "
            "depends on the vendor's RANDOM() seedability — output may not be "
            "byte-identical across runs",
            UserWarning,
            stacklevel=2,
        )

    if spec.scheme == "duckdb":
        return _read_duckdb(spec, sql, timeout=timeout)
    return _read_via_connectorx(spec, sql, timeout=timeout)


def _base_query(spec: DatabaseUri) -> str:
    """The user's selection — either `SELECT * FROM <table>` or the raw --query.

    Single-table source. The pushdown wrapper turns this into a sampled query
    if `fit_rows` is set."""
    if spec.query is not None:
        return spec.query
    if spec.table is not None:
        return f"SELECT * FROM {spec.table}"
    raise ValueError("DatabaseUri must have either `table` or `query` set")


def _enforce_row_count_threshold(spec: DatabaseUri, estimate: int, fit_rows: int | None) -> None:
    """If the table is huge and the user didn't ask to sample, refuse to read.

    The 1M threshold is a safety net: streaming a billion-row Snowflake table
    to fit 25k rows is a $5k mistake. Users with intent can opt in via
    `--fit-rows N` (sample to N) or `--fit-rows 0` (fit on the whole thing)."""
    import typer

    if estimate <= ROW_COUNT_THRESHOLD:
        return
    if fit_rows is not None:
        # User opted in explicitly. Emit a one-line stderr warning for full-fit.
        if fit_rows == 0:
            warnings.warn(
                f"reading full table from {spec.uri} (~{estimate:,} rows); "
                "this streams every row over the network — consider --fit-rows N",
                UserWarning,
                stacklevel=3,
            )
        return
    raise typer.BadParameter(
        f"table at {spec.uri} has ~{estimate:,} rows; pass --fit-rows N to "
        "sample, or --fit-rows 0 to fit on the whole table"
    )


def _probe_row_count(spec: DatabaseUri, *, timeout: int) -> int | None:
    """Run the cheap row-count query. Returns None on benign probe failure
    (unknown table, permission issue) so we don't block the user from reading
    a table whose catalog we can't see — the threshold check then doesn't fire."""
    sql = build_count_sql(spec.scheme, table=spec.table, query=spec.query)
    try:
        if spec.scheme == "duckdb":
            df = _read_duckdb(spec, sql, timeout=timeout)
        else:
            df = _read_via_connectorx(spec, sql, timeout=timeout)
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
    return int(raw)


def _read_duckdb(spec: DatabaseUri, sql: str, *, timeout: int) -> pl.DataFrame:
    """In-process DuckDB read. Avoids the ConnectorX hop for the local case
    (and works without the `[sql]` extra installed)."""
    import urllib.parse

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
        con = duckdb.connect(db_path)
        try:
            result = con.execute(sql)
            description = result.description
            rows = result.fetchall()
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


def _read_via_connectorx(spec: DatabaseUri, sql: str, *, timeout: int) -> pl.DataFrame:
    """Snowflake / Postgres / future-vendor read via Polars + ConnectorX.

    The `[sql]` extra is required here. We catch the ImportError raised by
    Polars when ConnectorX isn't present and re-raise a clear install hint."""
    try:
        # Importing connectorx eagerly gives us a deterministic error message
        # before Polars' own optional-import path runs.
        import importlib

        importlib.import_module("connectorx")
    except ImportError as exc:
        import typer

        raise typer.BadParameter(
            f"the {spec.scheme}:// scheme requires the optional [sql] extra. "
            'Install it with: pip install "doppeldata[sql]"'
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
            raise WarehouseConnectionError(f"failed to read from {spec.uri}: {exc}") from exc

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
]
