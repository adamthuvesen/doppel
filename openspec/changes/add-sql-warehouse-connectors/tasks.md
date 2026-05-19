## 1. Pre-flight

- [ ] 1.1 Confirm `connectorx>=0.3` is the right pin (check the latest stable on PyPI; ARM-mac wheels available).
- [ ] 1.2 Confirm Polars `read_database_uri` signature on the pinned Polars version; document `engine="connectorx"` invocation.
- [ ] 1.3 Decide on a doppel-internal exception hierarchy (`WarehouseConnectionError`, `UnsupportedSinkError`, `RowCountProbeError`); add to `src/doppel/__init__.py` or a new `src/doppel/sources/errors.py`.

## 2. Dependency wiring

- [ ] 2.1 Add `[project.optional-dependencies].sql = ["connectorx>=0.3"]` to `pyproject.toml`.
- [ ] 2.2 Update the `[all]` extra to include `[sql]`'s contents.
- [ ] 2.3 Update CI matrix (or doc the manual step) so the SQL tests run with the `[sql]` extra installed; keep a job that runs without it to verify the "extra not installed" rejection path.

## 3. `SourceSpec` and `parse_spec`

- [ ] 3.1 Create `src/doppel/sources/spec.py`. Define `FilePath`, `DatabaseUri`, and `SourceSpec = FilePath | DatabaseUri` as frozen dataclasses.
- [ ] 3.2 Define `SinkSpec = FilePath | DuckDbSink` analogously in `src/doppel/sinks/spec.py` (or share `sources/spec.py`).
- [ ] 3.3 Implement `${VAR}` expansion via `re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", ...)`. Missing env var raises `BadParameter`.
- [ ] 3.4 Implement password redaction: `_redact_uri(parsed) -> str` substitutes `:***@` into the netloc.
- [ ] 3.5 Implement `parse_spec(value: str | Path, *, table: str | None, query: str | None, password_cmd: str | None) -> SourceSpec`. Routes file paths to `FilePath`, URIs to `DatabaseUri`. Enforces `table` xor `query` for URIs and that neither is set for files.
- [ ] 3.6 Implement `_resolve_password(uri, password_cmd, env_resolved)`: precedence `--password-cmd > URI-embedded > error if neither and the scheme requires auth`. Warn when `--password-cmd` overrides a URI password.
- [ ] 3.7 Unit tests for every parse_spec branch; especially redaction, missing env vars, both/neither selector errors.

## 4. Source dispatcher

- [ ] 4.1 Refactor `src/doppel/sources/file.py` to expose `read_file(path: Path) -> pl.DataFrame` (current `read` renamed).
- [ ] 4.2 Create `src/doppel/sources/__init__.py` with top-level `read(spec: SourceSpec, *, fit_rows, seed, timeout) -> pl.DataFrame` that dispatches on `isinstance(spec, FilePath | DatabaseUri)`.
- [ ] 4.3 Update all six call sites (`gen.py`, `fit.py`, `artifact.py`, `diff.py`, `schema.py`, `schema/multi.py`) to pass `SourceSpec` instead of `Path`. Internal imports change from `from doppel.sources import file as source_file` to `from doppel.sources import read as source_read`.

## 5. SQL source (`sources/sql.py`)

- [ ] 5.1 Create `src/doppel/sources/sql.py` exposing `read(spec: DatabaseUri, *, fit_rows: int | None, seed: int | None, timeout: int) -> pl.DataFrame`.
- [ ] 5.2 Implement `build_pushdown_sql(scheme, base_query, fit_rows, seed, row_count_estimate) -> tuple[str, bool]` as a pure function. Returns `(sql, determinism_warning_needed)`. Per-vendor branches per the design table.
- [ ] 5.3 Implement `build_count_sql(scheme, table: str | None, query: str | None) -> str`. Snowflake `INFORMATION_SCHEMA.TABLES.ROW_COUNT`; Postgres `pg_class.reltuples`; for `--query`, wrap in `SELECT COUNT(*) FROM (<query>) AS _doppel_probe`.
- [ ] 5.4 Implement `_probe_row_count(spec, timeout) -> int | None`. Returns `None` for DuckDB (caller skips the threshold check). On Snowflake/Postgres, runs the count query and returns the estimate.
- [ ] 5.5 Implement threshold check: if `row_count > 1_000_000` and `fit_rows is None`, raise `BadParameter` with the row count and the suggested flag values.
- [ ] 5.6 Implement the DuckDB read path via `duckdb.connect(...).execute(sql).pl()` (avoiding the ConnectorX hop for the local case).
- [ ] 5.7 Implement Snowflake/Postgres reads via `pl.read_database_uri(uri, query, engine="connectorx")`.
- [ ] 5.8 Wrap driver exceptions in `WarehouseConnectionError` with redacted URI in the message.
- [ ] 5.9 Detect missing `[sql]` extra (ImportError on `connectorx`) and re-raise as `BadParameter` with the install hint.

## 6. Sink dispatcher

- [ ] 6.1 Refactor `src/doppel/sinks/file.py` to expose `write_file(df, path: Path)` (current `write` renamed).
- [ ] 6.2 Create `src/doppel/sinks/sql.py` with `write_duckdb(df, spec: DuckDbSink)`; raises `UnsupportedSinkError` if given a non-DuckDB scheme.
- [ ] 6.3 Create `src/doppel/sinks/__init__.py` with top-level `write(df, spec: SinkSpec)`. Routes on tag; Snowflake/Postgres URIs raise `UnsupportedSinkError` at the dispatcher.
- [ ] 6.4 Update all sink call sites (`gen.py:231`, `gen.py:332`, `fit.py:156`).

## 7. CLI wiring

- [ ] 7.1 Create `SourceSpecParam(typer.ParamType)` (or click-level equivalent) in `src/doppel/cli/_common.py`. The `convert` method calls `parse_spec` and returns a `SourceSpec`. Empty string and `--`-only values raise BadParameter.
- [ ] 7.2 Create `SinkSpecParam` analogously.
- [ ] 7.3 Register the new shared flags in `_common.py`: `--table`, `--query`, `--password-cmd`, `--connection-timeout`. Compose into the affected subcommands.
- [ ] 7.4 Update `gen`, `fit`, `schema infer`, `diff` to accept `SourceSpec` (via `SourceSpecParam`) instead of `Path`.
- [ ] 7.5 Update `gen` and `fit` `-o`/`--output` to accept `SinkSpec` (via `SinkSpecParam`).
- [ ] 7.6 `doppel sample` accepts a `SinkSpec` for `-o` and `SourceSpec` is irrelevant (its source is the `.doppel` artifact, not raw data).
- [ ] 7.7 Enforce `--table` xor `--query` at the click-type level when the resolved spec is a `DatabaseUri`; raise BadParameter on the both-set and neither-set cases. Reject `--table`/`--query` on `FilePath` specs.
- [ ] 7.8 Wire `--password-cmd` execution via `subprocess.run(shell-string, shell=True, capture_output=True, timeout=connection_timeout)`. Stdout is the password; stderr is captured and surfaced on non-zero exit.
- [ ] 7.9 Add the one-line `[info] reading from <redacted-uri>` log at the start of each SQL read; route through the existing Rich console.

## 8. Multi-table SQL in `schema.toml`

- [ ] 8.1 Extend the Pydantic model in `src/doppel/schema/multi.py` to accept `uri: str | None` and `query: str | None` alongside `path`. Validator enforces exactly-one-of `path` / `uri`; if `uri`, exactly-one-of `table` / `query`.
- [ ] 8.2 In `multi.to_dataset`, route through `parse_spec` per table; share a single connection per URI across tables.
- [ ] 8.3 Document the multi-table TOML example in `docs/sql-connectors.md` and inline in README.

## 9. Tests — URI parser

- [ ] 9.1 `parse_spec("/tmp/users.csv")` returns FilePath with the path.
- [ ] 9.2 `parse_spec("snowflake://user@account/db/schema?warehouse=WH", table="USERS")` returns DatabaseUri with all fields populated and a redacted URI.
- [ ] 9.3 `${ENV}` expansion: set env var, parse, assert expanded; unset env var raises BadParameter naming the variable.
- [ ] 9.4 Password redaction: parse a URI with `user:hunter2@`; assert `.uri` contains `:***@`; assert `.raw_uri` contains `:hunter2@`.
- [ ] 9.5 Unknown scheme (`bigquery://...`) raises BadParameter naming supported schemes.
- [ ] 9.6 `--table` and `--query` both set → BadParameter.
- [ ] 9.7 `--table` and `--query` both unset on a URI → BadParameter.
- [ ] 9.8 `--table` or `--query` set on a FilePath → BadParameter.

## 10. Tests — auth

- [ ] 10.1 `--password-cmd` overrides URI password; warning emitted; raw URI to driver has the command's stdout.
- [ ] 10.2 `--password-cmd` exits non-zero → BadParameter quoting stderr.
- [ ] 10.3 No password mechanism set + scheme requires auth → BadParameter naming the three mechanisms.
- [ ] 10.4 Redaction in error paths: trigger a connection error; assert raw password is absent from the message.

## 11. Tests — DuckDB end-to-end

- [ ] 11.1 Fixture: create a tempfile DuckDB with a seeded fixture table (e.g. 1000 rows, mixed-dtype).
- [ ] 11.2 `doppel gen "duckdb:///<tmp>" --table users -n 100 -o <out>.csv --seed 1` → exit 0, output has 100 rows, schema matches.
- [ ] 11.3 `doppel fit "duckdb:///<tmp>" --table users -o <model>.doppel --seed 1` → exit 0, artifact loads.
- [ ] 11.4 `doppel sample <model>.doppel -n 100 -o "duckdb:///<tmp2>?table=synth"` → DuckDB sink writes; `synth` table readable.
- [ ] 11.5 `doppel diff "duckdb:///<tmp>?table=users" "duckdb:///<tmp2>?table=synth"` → quality report produced.
- [ ] 11.6 `doppel schema infer "duckdb:///<tmp>" --table users -o <schema>.toml` → TOML produced; types match.

## 12. Tests — mocked Snowflake/Postgres

- [ ] 12.1 Patch `pl.read_database_uri` to return a fixture DataFrame; assert the URI passed has the expected redacted-vs-raw distinction.
- [ ] 12.2 Patch the driver; assert the pushdown SQL emitted for Snowflake contains `SAMPLE (N ROWS) SEED (S)`.
- [ ] 12.3 Same for Postgres `TABLESAMPLE BERNOULLI(p) REPEATABLE(S)`.
- [ ] 12.4 Same for DuckDB `USING SAMPLE N ROWS (REPEATABLE S)` (direct `duckdb` path).
- [ ] 12.5 ANSI fallback: hypothetical `unknown://` scheme (or test-only registered scheme) → SQL contains `ORDER BY RANDOM() LIMIT N` + warning.
- [ ] 12.6 Row-count probe: mock returns 5_000_000; no `--fit-rows` → BadParameter with the row count.
- [ ] 12.7 Row-count probe with `--fit-rows 0` → proceeds + stderr warning.
- [ ] 12.8 Row-count probe with `--query`: assert probe SQL is `SELECT COUNT(*) FROM (<query>) AS _doppel_probe`.

## 13. Tests — sink rejection

- [ ] 13.1 `doppel gen ... -o "snowflake://..."` → BadParameter naming supported sinks.
- [ ] 13.2 `doppel gen ... -o "postgres://..."` → BadParameter.
- [ ] 13.3 File sink behavior unchanged (regression: existing tests stay green).

## 14. Tests — multi-table SQL

- [ ] 14.1 `schema.toml` with mixed `path` + `uri` tables loads; both synthesize; FK integrity preserved.
- [ ] 14.2 `path` + `uri` in the same block → TOML load error naming the conflict.
- [ ] 14.3 `uri` without `table`/`query` → TOML load error.
- [ ] 14.4 Single connection reuse: multiple tables on the same URI share one connection (assert via mock call count).

## 15. Tests — `diff` symmetric URIs

- [ ] 15.1 `doppel diff duckdb:///real duckdb:///synth --table users` produces a quality report.
- [ ] 15.2 Mixed file + URI diff works.

## 16. Tests — connection lifecycle and `[sql]` extra

- [ ] 16.1 `--connection-timeout 1` against a slow mock → clear timeout error with redacted URI.
- [ ] 16.2 With the `[sql]` extra uninstalled, Snowflake URI → BadParameter pointing at the install command.
- [ ] 16.3 With `[sql]` uninstalled, DuckDB URI works (uses top-level `duckdb`, not connectorx).
- [ ] 16.4 Redacted URI appears in the `[info] reading from ...` log; raw URI does not.

## 17. Docs

- [ ] 17.1 README: add SQL Quickstart block with DuckDB, Snowflake, Postgres one-liners; add `--password-cmd "op read op://..."` example.
- [ ] 17.2 README Limitations: warehouse sinks are explicitly out (use file or DuckDB).
- [ ] 17.3 SECURITY.md: add a SQL-connector section. Cover password redaction at the parser boundary, `--query` as developer-trust input, ConnectorX→ADBC migration plan, and the residual risk of vendor driver vulnerabilities.
- [ ] 17.4 New `docs/sql-connectors.md`: URI format reference per scheme, auth model, sample pushdown SQL examples, row-count probe semantics, multi-table schema TOML example, Postgres `TABLESAMPLE` accuracy note, Snowflake key-pair auth as a v2 follow-up.
- [ ] 17.5 `docs/determinism.md` (if it exists): note that `--seed` propagates through SQL pushdown for Snowflake/Postgres/DuckDB; ANSI fallback warns.
- [ ] 17.6 Update CLI `--help` text for `gen`, `fit`, `sample`, `diff`, `schema infer` to mention URI inputs.

## 18. CI gates

- [ ] 18.1 `uv run ruff check src tests` clean.
- [ ] 18.2 `uv run ruff format --check src tests` clean.
- [ ] 18.3 `uv run pyright` 0 errors in strict mode. Narrow `Any` casts only at the connectorx boundary.
- [ ] 18.4 `uv run pytest` green, full coverage including DuckDB end-to-end suite.
- [ ] 18.5 CI matrix: one job with `uv sync --all-extras`, one job with `uv sync` (no extras) to verify the rejection path.
- [ ] 18.6 `uv run doppel gen --help` shows the new flags with the expected descriptions.
- [ ] 18.7 `uv run doppel --help` is reasonable for users browsing the new surface.
