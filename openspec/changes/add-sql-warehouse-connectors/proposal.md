## Why

doppel reads/writes CSV/TSV/Parquet/JSON/Arrow only ([src/doppel/sources/file.py](../../../src/doppel/sources/file.py),
[src/doppel/sinks/file.py](../../../src/doppel/sinks/file.py)). The full-codebase audit
([openspec/custom/reviews/full-codebase-audit-2026-05-17.md](../../custom/reviews/full-codebase-audit-2026-05-17.md))
flagged "README advertises SQL, but the source/sink layer does not support it" as a
truth-in-advertising bug AND named it the largest workflow friction wall. Real datasets
live in DuckDB, Snowflake, Postgres, BigQuery — forcing users to "export to Parquet → run
doppel → load back" is the reason a curious developer bounces. This change closes the
gap with the lowest-risk slice: SQL read for the four most common backends, plus DuckDB
write. Warehouse writes are explicitly deferred.

## What Changes

- Sources accept either a file path or a database URI:
  - `duckdb:///path.db?table=T` (or in-memory `duckdb://?table=T`)
  - `snowflake://user@account/db/schema?warehouse=W`
  - `postgres://user@host:5432/dbname`
- Sinks accept file paths (unchanged) and DuckDB URIs. Snowflake/Postgres sinks are
  rejected with a clear "use file or DuckDB sink" message.
- New `--table T` and `--query "SELECT ..."` flags on `gen`, `fit`, `schema infer`, and
  `diff`. Mutually exclusive; exactly one is required when the source is a URI.
- New `--password-cmd "op read op://..."`, `--connection-timeout SECONDS` flags. URI-embedded
  passwords work with a one-line shell-history warning. `${ENV_VAR}` interpolation in the URI
  is supported.
- A `SourceSpec` tagged union (`FilePath | DatabaseUri`) parsed once at the CLI boundary via
  a custom Typer click param type. `source/sink` modules dispatch on the tag — no new ABCs,
  no string-contains dispatch.
- Sample pushdown: when `--fit-rows N` is set, the SQL connector pushes the sample into the
  warehouse — Snowflake `SAMPLE (N ROWS) SEED (S)`, Postgres `TABLESAMPLE BERNOULLI(p)
  REPEATABLE(S)`, DuckDB `USING SAMPLE N ROWS (REPEATABLE)`. ANSI fallback (`ORDER BY
  RANDOM() LIMIT N`) for any other vendor, with a warning that determinism is vendor-dependent.
- Row-count probe for Snowflake/Postgres sources only: if the table has > 1M rows and
  `--fit-rows` is unset, hard-fail with a message naming the row count and suggesting
  `--fit-rows N` or `--fit-rows 0`. DuckDB/file sources keep the current auto-cap behavior.
- Multi-table SQL: `schema.toml` `[[tables]]` blocks accept either `path` or `uri` (with
  `table` / `query` keys). Same dispatch at multiple call sites.
- `diff` accepts URIs symmetrically — both arguments use the same `SourceSpec`.
- New `[sql]` optional extra adds `connectorx>=0.3`. DuckDB is already a top-level
  dependency.

**Out of scope** (deferred): Snowflake/Postgres sink writes; ADBC drivers as the default
(noted as v2 once `adbc-driver-snowflake` matures); vendor-native clients
(`snowflake-connector-python`, `psycopg`); transient-failure retry; connection pooling;
async; `--password-env` flag; TTY-prompt auth; SQL injection sanitization on `--query`
(developer-trust input).

## Capabilities

### New Capabilities

- `warehouse-connectors`: read tabular data from DuckDB, Snowflake, and Postgres warehouses
  via database URIs; write synthetic output to file or DuckDB. Includes URI dispatch,
  auth model, sample pushdown, row-count probe, multi-table SQL schema support, and
  symmetric `diff` URI handling.

### Modified Capabilities

None. The `conditional-generation` capability proposed in
[add-conditional-where-filter](../add-conditional-where-filter/proposal.md) is orthogonal
to source/sink dispatch; the two changes do not modify each other's requirements.

## Impact

**Code**
- `src/doppel/sources/__init__.py` — top-level `read(spec: SourceSpec) -> pl.DataFrame`
  dispatcher (replaces today's flat module import).
- `src/doppel/sources/file.py` — kept as the file branch; unchanged behavior on `Path` input.
- `src/doppel/sources/spec.py` (NEW) — `SourceSpec` tagged union (`FilePath | DatabaseUri`),
  `parse_spec(value: str | Path) -> SourceSpec`, `${ENV}` expander, password-redaction helper.
- `src/doppel/sources/sql.py` (NEW) — `read(spec: DatabaseUri, *, table, query, fit_rows, seed,
  timeout) -> pl.DataFrame`. Per-vendor pushdown SQL generation. Uses `pl.read_database_uri`
  with ConnectorX as the driver.
- `src/doppel/sinks/__init__.py` — top-level `write(df, spec: SinkSpec)` dispatcher.
- `src/doppel/sinks/file.py` — extension dispatch unchanged.
- `src/doppel/sinks/sql.py` (NEW) — DuckDB-only writer. Snowflake/Postgres URIs raise
  `UnsupportedSinkError` at parse time.
- `src/doppel/cli/_common.py` — `SourceSpecParam`, `SinkSpecParam` Typer custom click types;
  `--table`, `--query`, `--password-cmd`, `--connection-timeout` shared flag registration.
- `src/doppel/cli/gen.py`, `fit.py`, `artifact.py`, `diff.py`, `schema.py` — wire new
  parameters; route through dispatcher.
- `src/doppel/schema/multi.py` — `[[tables]]` accept either `path` or `uri` + `table` / `query`.

**Tests**
- DuckDB end-to-end (no credentials): tempfile DuckDB with seeded fixture; full `gen` / `fit` /
  `sample` / `diff` lifecycle.
- Mocked Snowflake/Postgres: patch `pl.read_database_uri`; assert URI passed correctly,
  pushdown SQL emitted, redaction in error paths.
- URI parser: scheme extraction, `${ENV}` expansion, password redaction.
- Auth precedence: `--password-cmd` overrides URI password (with warning); missing env var
  raises clearly.
- Selectors: `--table` / `--query` mutual-exclusion matrix.
- Sample pushdown: per-vendor SQL inspection via mock; `--seed` reproducibility.
- Row-count probe: > 1M rows + missing `--fit-rows` hard-fails; explicit `--fit-rows N` or
  `--fit-rows 0` proceeds.
- Sink rejection: `-o snowflake://...` and `-o postgres://...` raise BadParameter.
- Multi-table SQL: mixed `path` + `uri` tables in one schema.
- `diff` symmetric URIs.
- Connection timeout against a slow mock raises a clear error.

**Dependencies**
- New optional extra `[sql]`: `connectorx>=0.3`. DuckDB stays top-level.

**Docs**
- README: SQL Quickstart block with `doppel gen "duckdb:///data.db" --table users -n 1000`;
  Snowflake/Postgres examples; auth section showing `--password-cmd "op read op://..."`.
- SECURITY.md: SQL-credentials threat model — passwords redacted at parser boundary, no
  unredacted logging anywhere; `--query` documented as developer-trust input.
- New `docs/sql-connectors.md`: URI format, auth model, sample pushdown, row-count probe,
  vendor caveats (Snowflake key-pair auth, Postgres TABLESAMPLE accuracy), determinism
  contract.

**Risk**
- ConnectorX's bus-factor is moderate (small maintainer set). Mitigation: ADBC migration
  documented as v2; the abstraction surface (`SourceSpec` + per-vendor SQL generators) is
  driver-agnostic.
- Type stubs for `connectorx` may be absent — narrow `Any` casts at the boundary, no
  pyright-strict relaxation elsewhere.
- Multi-table SQL extends the schema TOML format; existing single-table file schemas
  continue to parse unchanged.
