# SQL connectors

Doppel reads from DuckDB, Snowflake, and Postgres via database URIs. Install
the optional extra to enable Snowflake / Postgres support:

```bash
pip install "doppeldata[sql]"
```

DuckDB reads work without the extra (they use the top-level `duckdb`
dependency directly).

## URI formats

| Scheme        | URI shape                                                       |
| ------------- | --------------------------------------------------------------- |
| DuckDB        | `duckdb:///abs/path/to/file.db`                                 |
| Snowflake     | `snowflake://user@account/db/schema?warehouse=WH&role=R`        |
| Postgres      | `postgres://user@host:5432/dbname` (alias: `postgresql://...`)  |

For every URI source you must pass **exactly one** of:

- `--table NAME` — reads the whole table
- `--query "SELECT ..."` — reads the result of a custom query (developer-trust input)

For DuckDB the path goes in the URI path component; for Snowflake/Postgres
the database, schema, and warehouse routing live in the path and query
string respectively.

## Auth

Three mechanisms, applied in precedence order:

1. **`--password-cmd "<shell-cmd>"`** (recommended). The command's stdout
   is captured and substituted into the URI's password slot. Example:
   ```
   --password-cmd "op read op://vault/snowflake/password"
   ```
   If the command exits non-zero, doppel exits with `BadParameter` quoting
   the subprocess stderr.

2. **`${ENV_VAR}` interpolation** anywhere in the URI:
   ```
   "snowflake://${SF_USER}:${SF_PASS}@account/db/schema?warehouse=WH"
   ```
   Missing variables raise a clear error naming the variable. Only the
   braced `${VAR}` form is expanded — bare `$VAR` is left literal so
   passwords with `$` in them survive.

3. **URI-embedded** (`scheme://user:pass@host/...`): supported but emits a
   one-line stderr warning that the password appears in shell history.
   Prefer (1) or (2) outside throwaway development.

Passwords are redacted at the parser boundary by substituting `:***@`
into a log-safe URI form. The raw URI is held separately and passed
straight to the driver — it never appears in logs, error messages, or
`--explain` output.

## Sample pushdown

When `--fit-rows N` is set on a SQL source, doppel pushes the sample down
to the warehouse using vendor-native syntax:

| Vendor    | Generated SQL                                                              |
| --------- | -------------------------------------------------------------------------- |
| Snowflake | `SELECT * FROM (<base>) SAMPLE (N ROWS) SEED (S)`                          |
| Postgres  | `SELECT * FROM (<base>) AS t TABLESAMPLE BERNOULLI(p) REPEATABLE(S) LIMIT N` |
| DuckDB    | `SELECT * FROM (<base>) AS t USING SAMPLE N ROWS (REPEATABLE S)`           |
| Other     | `SELECT * FROM (<base>) AS t ORDER BY RANDOM() LIMIT N` (with warning)     |

The probability `p` for Postgres is computed from the row-count estimate
with a 5% oversample (since `TABLESAMPLE BERNOULLI` returns approximate
row counts); the client-side `LIMIT N` then guarantees the exact row count
the user asked for. Determinism for the ANSI fallback depends on the
vendor's `RANDOM()` seedability; doppel emits a warning when the fallback
is used.

Setting `--seed` propagates through every supported vendor's seed clause.

## Row-count probe

Before reading from Snowflake or Postgres, doppel runs a cheap row-count
query against the catalog:

- Snowflake: `SELECT ROW_COUNT FROM INFORMATION_SCHEMA.TABLES`
- Postgres: `SELECT reltuples::BIGINT FROM pg_class`
- For `--query`: `SELECT COUNT(*) FROM (<query>) AS _doppel_probe`

If the estimate exceeds **1,000,000 rows** AND `--fit-rows` is not set,
doppel hard-fails with a message naming the row count and suggesting
`--fit-rows N` (to sample) or `--fit-rows 0` (to fit on the whole thing,
with a warning that the network egress will be paid).

The probe is skipped for DuckDB and file sources (the auto-cap behavior
applies for files, and DuckDB is local).

## Multi-table SQL

`schema.toml` `[[tables]]` blocks accept either `path` or `uri`:

```toml
[tables.users]
path = "data/users.parquet"
primary_key = "user_id"

[tables.orders]
uri = "snowflake://${SF_USER}@account/db/schema?warehouse=WH"
table = "ORDERS"
primary_key = "order_id"

# Or use --query in place of `table`:
# query = "SELECT * FROM ORDERS WHERE created_at >= '2025-01-01'"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
```

Each `[[tables]]` block must declare exactly one of `path` / `uri`, and
URI-backed tables must additionally declare exactly one of `table` /
`query`. The CLI's `--password-cmd` and `--connection-timeout` apply
globally to every SQL table in the run.

## Sinks: file and DuckDB only

The `-o`/`--output` flag accepts:

- a file path (any extension supported by `sinks.file`); or
- a DuckDB URI of the form `duckdb:///path.db?table=NAME`.

Snowflake/Postgres sinks raise `BadParameter` at parse time. Warehouse
writes have their own design surface (transactions, idempotency, table
existence, schema permissions, recovery) and are out of scope for v1.
Write to a file or DuckDB and load with your normal ELT tooling.

## Connection lifecycle

One connection per source URI per CLI invocation. `--connection-timeout
SECONDS` (default 300) wires into the driver's timeout where supported,
and a Python-side watchdog enforces it where not. The redacted URI is
logged to stderr at info level before the connection opens, so failures
correlate to a connection target without leaking credentials.

## Per-vendor caveats

- **Snowflake**: only password authentication in v1. Key-pair, OAuth, and
  SSO browser flow are forward-compatible at the URI level
  (`?authenticator=externalbrowser&...`) but not exercised. The
  `INFORMATION_SCHEMA.TABLES.ROW_COUNT` probe returns the value as of the
  last `ANALYZE`/`COMPACT`; in practice it's accurate enough for the 1M
  threshold safety net.
- **Postgres**: `TABLESAMPLE BERNOULLI(p)` returns approximately `p%` of
  rows, not exactly N. Doppel oversamples by 5% and applies `LIMIT N`
  client-side to guarantee the exact row count.
- **DuckDB**: works without the `[sql]` extra. The in-memory variant
  (`duckdb://?table=T`) is supported for sources but not for sinks
  (the sink must point at a persistable file).

## Driver story

ConnectorX is the v1 driver. ADBC (`adbc-driver-snowflake`,
`adbc-driver-postgresql`) is the v2 migration target once the Snowflake
ADBC driver is past 1.0 in production usage. The per-vendor SQL
generators are pure functions and driver-agnostic, so the migration is a
one-line swap in `sources/sql.py::_read_via_connectorx`. See
[SECURITY.md](../SECURITY.md) for the threat-model implications.
