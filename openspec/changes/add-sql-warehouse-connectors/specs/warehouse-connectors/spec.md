## ADDED Requirements

### Requirement: Database-URI sources

The `doppel gen`, `doppel fit`, `doppel schema infer`, and `doppel diff` commands
SHALL accept database URIs as input alongside file paths. The supported URI
schemes are `duckdb://`, `snowflake://`, and `postgres://` (with `postgresql://`
accepted as an alias). When the input is a URI, the user MUST provide exactly
one of `--table T` or `--query "SELECT ..."`. Passing both, or neither, MUST
raise `typer.BadParameter` at CLI parse time.

When the input is a file path, the `--table` and `--query` flags MUST be
rejected with `typer.BadParameter`.

#### Scenario: DuckDB URI with `--table`

- **WHEN** the user runs `doppel gen "duckdb:///tmp/test.db" --table users -n 100 -o out.csv`
- **AND** the `users` table exists in `/tmp/test.db`
- **THEN** the command MUST exit 0
- **AND** the output file MUST contain 100 synthetic rows whose schema matches the source table

#### Scenario: Snowflake URI with `--query`

- **WHEN** the user runs `doppel gen "snowflake://user@account/db/schema?warehouse=WH" --query "SELECT * FROM USERS WHERE plan='enterprise'" -n 1000 -o out.parquet`
- **THEN** the driver MUST be invoked with the user's query (wrapped for pushdown as documented)
- **AND** the output file MUST contain 1000 synthetic rows

#### Scenario: Postgres URI with `--table`

- **WHEN** the user runs `doppel gen "postgres://user@host:5432/dbname" --table public.users -n 100 -o out.csv`
- **THEN** the driver MUST receive the URI with the `public.users` table reference

#### Scenario: Both `--table` and `--query` rejected

- **WHEN** the user passes both `--table users` and `--query "SELECT ..."` with any URI source
- **THEN** the command MUST exit with `BadParameter` whose message names both flags as mutually exclusive

#### Scenario: Neither `--table` nor `--query` with URI source

- **WHEN** the user passes a URI input but no `--table` or `--query`
- **THEN** the command MUST exit with `BadParameter` whose message states that one of `--table` or `--query` is required for URI sources

#### Scenario: `--table` rejected with file source

- **WHEN** the user passes `--table users` alongside a file path input
- **THEN** the command MUST exit with `BadParameter` stating that `--table` applies only to URI sources

### Requirement: `SourceSpec` tagged-union dispatch

A `SourceSpec` algebraic type (`FilePath | DatabaseUri`) SHALL be the single
representation that `sources.read` and `sinks.write` accept. The CLI layer
MUST parse the raw `str | Path` argument into a `SourceSpec` exactly once,
via a custom Typer click param type, before any other code sees it. Modules
downstream of the CLI MUST NOT perform string-based dispatch on `"://" in s`
or `path.suffix == "..."`.

#### Scenario: File path parses to `FilePath`

- **WHEN** `parse_spec("/tmp/users.csv")` is called
- **THEN** it MUST return a `FilePath` with `path = Path("/tmp/users.csv")`
- **AND** the path MUST be validated for existence

#### Scenario: Database URI parses to `DatabaseUri`

- **WHEN** `parse_spec("snowflake://user@account/db/schema?warehouse=WH")` is called with `--table USERS`
- **THEN** it MUST return a `DatabaseUri` with `scheme="snowflake"`, `table="USERS"`, `query=None`
- **AND** the redacted form MUST be available as `.uri` (used for logs)
- **AND** the raw form (passed to the driver) MUST be available as `.raw_uri` (never logged)

#### Scenario: Unknown scheme rejected

- **WHEN** the user passes `bigquery://...` (out of v1 scope)
- **THEN** `parse_spec` MUST raise `BadParameter` naming the supported schemes (`duckdb`, `snowflake`, `postgres`)

### Requirement: Authentication mechanisms

The connector SHALL support three authentication mechanisms for SQL sources,
which compose in this precedence order:

1. `--password-cmd "<shell command>"` (highest): the command is executed,
   stdout is captured and substituted into the URI's password slot. If the
   command exits non-zero, the CLI MUST exit with `BadParameter` quoting the
   subprocess stderr.
2. `${VAR}` interpolation anywhere in the URI: expanded from the process
   environment before parsing. Missing env vars MUST raise `BadParameter`
   naming the variable.
3. URI-embedded password (`scheme://user:pass@host/...`): used as a fallback;
   the parser MUST emit a one-line stderr warning that the password appears
   in shell history.

The raw password MUST never appear in logs, error messages, or `--explain`
output. Redaction MUST be enforced at the parser boundary by substituting
`:***@` into the netloc of the log-safe URI form.

#### Scenario: `--password-cmd` overrides URI password

- **WHEN** the user passes `--password-cmd "echo secret"` AND the URI is `snowflake://user:wrongpass@account/db/schema`
- **THEN** the driver MUST receive the URI with `user:secret@`
- **AND** a stderr warning MUST be emitted noting that `--password-cmd` overrode the URI password

#### Scenario: `${ENV}` expansion

- **WHEN** the URI is `snowflake://${SF_USER}:${SF_PASS}@account/db/schema?warehouse=WH`
- **AND** the environment has `SF_USER=adam` and `SF_PASS=hunter2`
- **THEN** the driver MUST receive the URI with `adam:hunter2@`

#### Scenario: Missing env var rejected

- **WHEN** the URI references `${SF_PASS}` and the variable is unset
- **THEN** `parse_spec` MUST raise `BadParameter` naming `SF_PASS`

#### Scenario: URI-embedded password warning

- **WHEN** the URI contains a password slot (`user:pass@`) and `--password-cmd` is not set
- **THEN** a one-line stderr warning MUST be emitted noting that the password is in shell history

#### Scenario: Redaction in error paths

- **WHEN** a connection error is raised
- **THEN** the error message MUST contain only the redacted URI (`:***@` form)
- **AND** MUST NOT contain the raw password anywhere

#### Scenario: `--password-cmd` failure

- **WHEN** `--password-cmd "false"` is passed (exits non-zero)
- **THEN** the command MUST exit with `BadParameter`
- **AND** the message MUST quote the subprocess's stderr output

### Requirement: Sample pushdown to the warehouse

When `--fit-rows N` is set on a SQL source AND `--seed S` is provided, the
connector SHALL emit per-vendor SQL that pushes the sample into the
warehouse using the vendor's seedable sampling clause:

- Snowflake: `SAMPLE (N ROWS) SEED (S)`
- Postgres: `TABLESAMPLE BERNOULLI(p) REPEATABLE(S)` where `p` is computed from `N` and the row-count estimate; client-side `LIMIT N` applied after to guarantee exact row count.
- DuckDB: `USING SAMPLE N ROWS (REPEATABLE S)`
- Unknown / future vendor: ANSI `ORDER BY RANDOM() LIMIT N`, with a stderr warning that determinism is vendor-dependent.

When `--seed` is omitted, the same syntax MUST be used without the seed
clause (Snowflake `SAMPLE (N ROWS)`, Postgres `TABLESAMPLE BERNOULLI(p)`,
DuckDB `USING SAMPLE N ROWS`).

#### Scenario: Snowflake pushdown SQL

- **WHEN** the user runs `doppel gen "snowflake://..." --table USERS --fit-rows 25000 --seed 42 -n 10000 -o out.csv`
- **THEN** the SQL submitted to the driver MUST contain `SAMPLE (25000 ROWS) SEED (42)`

#### Scenario: Postgres pushdown SQL

- **WHEN** the user runs the equivalent against `postgres://...` with `--fit-rows 10000 --seed 42` against a table with ~1M rows
- **THEN** the SQL MUST contain `TABLESAMPLE BERNOULLI(<computed-p>) REPEATABLE(42)`
- **AND** the client-side `LIMIT 10000` MUST be applied after (to guarantee exact row count)

#### Scenario: DuckDB pushdown SQL

- **WHEN** the user runs the equivalent against `duckdb://...` with `--fit-rows 1000 --seed 42`
- **THEN** the SQL MUST contain `USING SAMPLE 1000 ROWS (REPEATABLE 42)`

#### Scenario: ANSI fallback warns about determinism

- **WHEN** the URI scheme is not Snowflake / Postgres / DuckDB (hypothetical future vendor) AND `--seed` is set
- **THEN** the SQL MUST use `ORDER BY RANDOM() LIMIT N`
- **AND** a stderr warning MUST be emitted naming the vendor and the determinism caveat

#### Scenario: `--fit-rows` omitted on a small DuckDB table

- **WHEN** the user runs `doppel gen "duckdb:///tmp/small.db" --table users -n 100 -o out.csv` (no `--fit-rows`)
- **AND** the table has < 1M rows
- **THEN** the connector MUST read the full table without sampling
- **AND** no warning MUST be emitted

### Requirement: Row-count probe for warehouse sources

For Snowflake and Postgres sources only, the connector SHALL probe the row
count before reading. The probe MUST use the vendor's catalog
(`INFORMATION_SCHEMA.TABLES` for Snowflake, `pg_class.reltuples` for
Postgres) when reading a single table, and `SELECT COUNT(*) FROM (<query>)`
when reading via `--query`.

If the estimated row count exceeds 1,000,000 AND `--fit-rows` is unset, the
command MUST exit with `BadParameter` naming the row count and suggesting
`--fit-rows N` or `--fit-rows 0`. DuckDB and file sources MUST NOT run this
probe.

#### Scenario: Snowflake table with 5M rows, no `--fit-rows`

- **WHEN** the user runs `doppel gen "snowflake://..." --table BIG_TABLE -n 1000 -o out.csv`
- **AND** the row-count probe returns ~5,000,000
- **THEN** the command MUST exit with `BadParameter`
- **AND** the message MUST name the row count and suggest `--fit-rows N` or `--fit-rows 0`
- **AND** no rows MUST be streamed to the client

#### Scenario: Snowflake table with 5M rows, `--fit-rows 0`

- **WHEN** the user passes `--fit-rows 0` (opt-in to full fit)
- **AND** the row-count probe returns ~5,000,000
- **THEN** the command MUST proceed
- **AND** a stderr warning MUST be emitted noting the large row count

#### Scenario: Postgres table with 500k rows

- **WHEN** the user runs `doppel gen "postgres://..." --table small_table -n 1000 -o out.csv`
- **AND** the row-count probe returns ~500,000
- **THEN** the command MUST proceed (under the 1M threshold)
- **AND** no warning MUST be emitted

#### Scenario: DuckDB source skips the probe

- **WHEN** the user runs `doppel gen "duckdb:///big.db" --table huge -n 1000 -o out.csv`
- **THEN** no row-count probe MUST run
- **AND** the existing client-side auto-cap behavior MUST apply

#### Scenario: `--query` row-count probe

- **WHEN** the user uses `--query` against Snowflake or Postgres
- **THEN** the probe MUST wrap the user's query as `SELECT COUNT(*) FROM (<query>) AS _doppel_probe`
- **AND** the threshold check MUST apply to that count

### Requirement: Sinks accept files and DuckDB only

The `-o`/`--output` option SHALL accept either a file path (any extension
currently supported by `sinks.file`) or a DuckDB URI of the form
`duckdb:///path.db?table=T`. Snowflake and Postgres sinks MUST be rejected
at parse time with `BadParameter` naming the supported sink kinds.

#### Scenario: File sink unchanged

- **WHEN** the user passes `-o out.parquet`
- **THEN** the sink MUST behave exactly as today

#### Scenario: DuckDB sink writes a local file

- **WHEN** the user passes `-o "duckdb:///tmp/synth.db?table=users_synth"`
- **THEN** the sink MUST create or open `/tmp/synth.db` and write the DataFrame as `users_synth`
- **AND** if `users_synth` already exists, the sink MUST replace it (matching file overwrite semantics)

#### Scenario: Snowflake sink rejected

- **WHEN** the user passes `-o "snowflake://..."`
- **THEN** the command MUST exit with `BadParameter`
- **AND** the message MUST state "Snowflake sinks are not supported; use file (.csv/.parquet/...) or DuckDB (duckdb:///path.db?table=T)"

#### Scenario: Postgres sink rejected

- **WHEN** the user passes `-o "postgres://..."`
- **THEN** the command MUST exit with `BadParameter`
- **AND** the message MUST state that Postgres sinks are not supported

### Requirement: Multi-table SQL in `schema.toml`

The `[[tables]]` blocks in `schema.toml` SHALL accept either a `path` field
or a `uri` field. When `uri` is given, exactly one of `table` or `query`
MUST also be given. Mixing `path` and `uri` in the same block MUST be
rejected at TOML-load time. A single multi-table run MAY mix `path`-based
and `uri`-based tables.

The CLI's `--password-cmd` and `--connection-timeout` flags apply globally
to all SQL tables in a multi-table run.

#### Scenario: Mixed `path` + `uri` tables

- **WHEN** `schema.toml` declares one table with `path = "data/users.parquet"` and another with `uri = "duckdb:///tmp/o.db"` + `table = "orders"`
- **THEN** the loader MUST parse both
- **AND** `doppel gen --schema schema.toml -n 100 -o out/` MUST synthesize both tables

#### Scenario: `path` + `uri` in same block rejected

- **WHEN** a single `[[tables]]` block has both `path` and `uri`
- **THEN** the TOML loader MUST raise a validation error naming the conflict

#### Scenario: `uri` without `table` or `query` rejected

- **WHEN** a `[[tables]]` block has `uri` but no `table` or `query`
- **THEN** the TOML loader MUST raise a validation error naming the missing key

### Requirement: `diff` accepts URIs symmetrically

The `doppel diff <real> <synth>` command SHALL accept either argument as a
file path or a database URI. When either argument is a URI, the `--table`
and `--query` flags SHALL apply identically to all URI inputs in that
invocation. Asymmetric per-input selection is not supported in v1; users
needing it MUST generate to an intermediate file.

#### Scenario: Two DuckDB URIs

- **WHEN** the user runs `doppel diff "duckdb:///real.db" "duckdb:///synth.db" --table users`
- **THEN** the command MUST read both tables from their respective DuckDB files
- **AND** produce the same quality report shape as a file-vs-file diff

#### Scenario: Mixed file + URI

- **WHEN** the user runs `doppel diff users_real.parquet "duckdb:///synth.db?table=users"`
- **THEN** the command MUST succeed
- **AND** the `--table` flag (if present) MUST apply only to the URI argument; the file argument MUST ignore it

### Requirement: Connection lifecycle and timeout

The connector SHALL open one connection per source URI per CLI invocation.
Multiple multi-table reads targeting the same URI MUST share a single
connection. The `--connection-timeout SECONDS` flag (default 300) MUST be
threaded into the driver's timeout parameter where supported; a Python-side
watchdog MUST enforce it where the driver does not.

The redacted URI MUST be printed to stderr at info level before the
connection is opened, so users can correlate failures with the connection
target.

#### Scenario: Single connection across multi-table read

- **WHEN** `schema.toml` declares three tables all using the same `snowflake://...` URI
- **THEN** exactly one connection MUST be opened
- **AND** the connection MUST be closed cleanly after the last read

#### Scenario: Timeout against slow query

- **WHEN** `--connection-timeout 5` is set AND the driver hangs > 5s on connection or query
- **THEN** the command MUST exit with a clear error mentioning the timeout
- **AND** the error MUST contain the redacted URI (no password)

#### Scenario: Redacted URI in stderr log

- **WHEN** any SQL read begins
- **THEN** stderr MUST contain a one-line `[info] reading from <redacted-uri>` log
- **AND** the redacted URI MUST NOT contain the raw password

### Requirement: `[sql]` optional extra and dependency posture

The `connectorx>=0.3` package SHALL be added as an optional extra
(`[project.optional-dependencies].sql`). The base `pip install doppeldata`
MUST NOT pull connectorx. DuckDB MUST remain a top-level dependency.

When a SQL URI is used and the extra is not installed, the command MUST
exit with `BadParameter` naming the install command (`pip install
"doppeldata[sql]"`).

#### Scenario: Base install rejects SQL URIs

- **WHEN** doppel is installed without the `[sql]` extra
- **AND** the user runs `doppel gen "snowflake://..." --table T -n 100 -o out.csv`
- **THEN** the command MUST exit with `BadParameter`
- **AND** the message MUST include the install command for the `[sql]` extra

#### Scenario: DuckDB works without `[sql]` extra

- **WHEN** doppel is installed without the `[sql]` extra
- **AND** the user runs `doppel gen "duckdb:///local.db" --table users -n 100 -o out.csv`
- **THEN** the command MUST succeed (DuckDB uses the top-level `duckdb` dep, not ConnectorX)

### Requirement: Documentation and threat model updates

The README SHALL include at least one SQL Quickstart example for each
supported scheme (DuckDB, Snowflake, Postgres) and at least one
`--password-cmd "op read ..."` example.

The SECURITY.md threat model SHALL document:
- Password redaction at the parser boundary.
- `--query` as developer-trust input (no SQL injection sanitization).
- ConnectorX as the v1 driver, with ADBC migration noted as v2.

A new `docs/sql-connectors.md` SHALL document URI formats, auth mechanisms,
sample pushdown, the row-count probe, and per-vendor caveats.

#### Scenario: README has SQL examples

- **WHEN** a reader scans the README
- **THEN** they MUST find at least one `doppel gen "duckdb://..."` example, one `snowflake://` example, and one `postgres://` example
- **AND** at least one `--password-cmd "op read op://..."` example

#### Scenario: SECURITY.md covers the SQL surface

- **WHEN** a reader scans SECURITY.md
- **THEN** they MUST find a section describing password redaction, `--query` as developer-trust input, and the ConnectorX→ADBC migration plan
