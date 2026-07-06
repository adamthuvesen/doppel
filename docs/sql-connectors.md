# SQL connectors

Doppel can read source data from DuckDB, Snowflake, and Postgres. Use this
when the table already lives in a warehouse and exporting to Parquet first is
extra work.

DuckDB works with the base install. Snowflake and Postgres need the SQL extra:

```bash
pip install "doppeldata[sql]"
```

DuckDB writes are also supported. Snowflake and Postgres are read-only from
doppel's point of view; write a file or a DuckDB table, then load it with your
normal warehouse tooling.

## URI formats

| Scheme        | URI shape                                                       |
| ------------- | --------------------------------------------------------------- |
| DuckDB        | `duckdb:///abs/path/to/file.db`                                 |
| Snowflake     | `snowflake://user@account/db/schema?warehouse=WH&role=R`        |
| Postgres      | `postgres://user@host:5432/dbname` (alias: `postgresql://...`)  |

For every URI source, pass exactly one selector:

- `--table NAME` — reads the whole table
- `--query "SELECT ..."` — reads the result of a custom query (developer-trust input)

For DuckDB, the path goes in the URI path component. For Snowflake and Postgres,
the database, schema, and warehouse routing live in the path and query string.

## Auth

There are three password mechanisms. doppel applies them in this order:

1. **`--password-cmd "<shell-cmd>"`**. This is the safest default for local
   use. doppel captures stdout and puts it in the URI password slot.
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

3. **URI-embedded** (`scheme://user:pass@host/...`): supported, with a
   one-line stderr warning because the password can appear in shell history.
   Use this only for throwaway local runs.

doppel redacts passwords at the parser boundary by substituting `:***@` into
the log-safe URI. The raw URI is kept separately and passed straight to the
driver. It should not appear in logs, error messages, or `--explain` output.

## Sample pushdown

When you set `--fit-rows N` on a SQL source, doppel asks the database for the
sample instead of reading the full table first:

| Vendor    | Generated SQL                                                              |
| --------- | -------------------------------------------------------------------------- |
| Snowflake | `SELECT * FROM (<base>) SAMPLE (N ROWS) SEED (S)`                          |
| Postgres  | `SELECT * FROM (<base>) AS t TABLESAMPLE BERNOULLI(p) REPEATABLE(S) LIMIT N` |
| DuckDB    | `SELECT * FROM (<base>) AS t USING SAMPLE N ROWS (REPEATABLE S)`           |
| Other     | `SELECT * FROM (<base>) AS t ORDER BY RANDOM() LIMIT N` (with warning)     |

For Postgres, `p` is computed from the row-count estimate with a 5% oversample
because `TABLESAMPLE BERNOULLI` returns approximate row counts. The client-side
`LIMIT N` keeps the final fit set at the requested size. Determinism for the
ANSI fallback depends on the vendor's `RANDOM()` seedability; doppel warns when
that fallback is used.

Setting `--seed` propagates through every supported vendor's seed clause.

## Row-count probe

Before reading from Snowflake or Postgres, doppel runs a cheap row-count
query against the catalog:

- Snowflake: `SELECT ROW_COUNT FROM INFORMATION_SCHEMA.TABLES`
- Postgres: `SELECT reltuples::BIGINT FROM pg_class`
- For `--query`: `SELECT COUNT(*) FROM (<query>) AS _doppel_probe`

If the estimate exceeds **1,000,000 rows** and `--fit-rows` is not set, doppel
fails before reading the data. The error names the row count and suggests
`--fit-rows N` to sample or `--fit-rows 0` to fit on the whole table. The latter
can move a lot of data over the network.

The probe is skipped for DuckDB and file sources (the auto-cap behavior
applies for files, and DuckDB is local).

## Multi-table SQL

In `schema.toml`, each `[tables.<name>]` block can read from a file or a URI:

```toml
[tables.users]
file = "data/users.parquet"
primary_key = "user_id"

[tables.orders]
uri = "snowflake://${SF_USER}@account/db/schema?warehouse=WH"
table = "ORDERS"
primary_key = "order_id"

# Or use query in place of table:
# query = "SELECT * FROM ORDERS WHERE created_at >= '2025-01-01'"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
```

Each `[tables.<name>]` block must declare exactly one of `file` or `uri`.
URI-backed tables must also declare exactly one of `table` or `query`. The
CLI's `--password-cmd` and `--connection-timeout` apply to every SQL table in
the run.

## Sinks

The `-o`/`--output` flag accepts:

- a file path (any extension supported by `sinks.file`); or
- a DuckDB URI of the form `duckdb:///path.db?table=NAME`.

Snowflake and Postgres sink URIs raise `BadParameter` at parse time. Warehouse
writes need decisions about transactions, idempotency, table creation, schema
permissions, and recovery. doppel leaves that to your normal ELT tooling.

## Connection lifecycle

Each source URI opens one connection per CLI invocation. `--connection-timeout
SECONDS` (default 300) is passed to the driver where supported, with a
Python-side watchdog for the rest. Before opening the connection, doppel logs
the redacted URI to stderr at info level so failures still point at the right
target.

## Per-vendor caveats

- **Snowflake**: only password authentication is supported. The
  `INFORMATION_SCHEMA.TABLES.ROW_COUNT` probe returns the value as of the
  last `ANALYZE`/`COMPACT`; it is accurate enough for the 1M threshold guard.
- **Postgres**: `TABLESAMPLE BERNOULLI(p)` returns approximately `p%` of
  rows, not exactly N. Doppel oversamples by 5% and applies `LIMIT N`
  client-side to guarantee the exact row count.
- **DuckDB**: works without the `[sql]` extra. The in-memory variant
  (`duckdb://?table=T`) is supported for sources but not for sinks
  (the sink must point at a persistable file).

## Driver

ConnectorX is the SQL read driver for Snowflake and Postgres. DuckDB reads
use DuckDB directly. See [SECURITY.md](../SECURITY.md) for the threat-model
implications of native database drivers.
