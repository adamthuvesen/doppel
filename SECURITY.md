# Security

## Reporting

Report security issues privately to the maintainers, not as public GitHub issues.
Acknowledgement within 72 hours; fix or mitigation timeline depends on severity.

## Threat model

doppel reads tabular data and (via `fit` / `sample`) reads/writes `.doppel` artifact
files. Two trust boundaries:

### 1. `.doppel` artifact files

A `.doppel` artifact is a gzipped tar of a manifest, an optional schema, and a pickled
fitted synthesizer. A crafted pickle payload can execute arbitrary code on load.

**Mitigations:**

- Pickle is deserialised through a **restricted unpickler**
  ([src/doppel/artifact/safe_pickle.py](src/doppel/artifact/safe_pickle.py)) that
  refuses any class outside an explicit allowlist (sklearn, numpy, polars, scipy,
  the narrow doppel model classes needed by the artifact format, narrow stdlib).
  Standard `os.system` / `subprocess.Popen` / `builtins.eval` pickle-RCE payloads
  are blocked before any code runs.
- Manifest is Pydantic-validated and version-checked **before** the pickle is read.
- Tar extraction uses `getmember` + `extractfile` only — no `tarfile.extractall`, so
  tarbomb / `..` path traversal is closed.

**What you should still do:**

- Only load `.doppel` files from trusted sources. The restricted unpickler reduces
  risk but doesn't eliminate it — a novel exploit chain inside an allowed class is
  out of scope for v0.1.
- For an artifact from an unknown source, inspect the manifest first:
  `tar -xzOf model.doppel manifest.json | jq`. Or use `doppel artifact info <file>`,
  which never invokes the unpickler.

### 2. SQL connectors and credentials

When the `[sql]` extra is installed, doppel can read from DuckDB, Snowflake,
and Postgres via database URIs. The threat surface and mitigations:

**Password handling.** The CLI accepts three orthogonal auth mechanisms:

1. `--password-cmd "<shell>"` (recommended): stdout becomes the password.
   Doppel never sees the plaintext on argv or in shell history.
2. `${VAR}` interpolation in the URI: expanded from the environment before
   parsing. Missing variables raise a clear error naming the variable.
3. URI-embedded passwords (`scheme://user:pass@host/...`): supported because
   Polars accepts this form natively, but **doppel emits a one-line stderr
   warning** that the password appears in shell history. Prefer (1) or (2).

The parser substitutes `:***@` into a redacted form of the URI at the
parser boundary; **only** the redacted form appears in logs, error
messages, or `--explain` output. The raw form is held in a separate field
and passed straight to the driver. We have a regression test that drives a
connection failure and asserts the raw password is absent from the message.

**`--query` is developer-trust input.** Doppel does **not** sanitise the
`--query` argument against SQL injection. The user owns the credentials and
the warehouse, so a malicious query is an own-goal, not a doppel
vulnerability. We document this in CLI help and treat the query string the
same way `bash -c` treats its argument.

**Vendor driver vulnerabilities.** ConnectorX is the v1 read driver and
ships its own native code. A vulnerability in ConnectorX or its underlying
libraries (Arrow, libpq, etc.) is out of scope for the doppel threat
model — keep your `[sql]` extra current with `pip install -U
"doppeldata[sql]"`.

**ADBC migration plan.** ConnectorX has a moderate bus factor. The
per-vendor SQL generators (`sources/sql.py`) are driver-agnostic and the
URI dispatch happens entirely in `sources/spec.py`, so swapping ConnectorX
for ADBC (`adbc-driver-snowflake`, `adbc-driver-postgresql`) when the
ecosystem matures is a one-line change in `_read_via_connectorx`. v2
roadmap.

**Warehouse writes are explicitly out of scope.** Snowflake and Postgres
sinks raise at parse time. DuckDB writes are file writes — same blast
radius as Parquet. This keeps the v1 surface area honest: doppel is a
synth tool, not an ELT tool.

### 3. Synthetic-output privacy

doppel's privacy posture is **heuristic**, not formal:

- When the `[pii]` extra is installed and Presidio detects PII in a `gen` source,
  those columns are stripped before fit and regenerated via Faker at sample time —
  no real names / emails / phone numbers reach the output.
- `doppel fit` refuses any source where Presidio detects PII; the artifact format
  doesn't yet carry detection metadata to support round-trip regeneration (v0.2).
- Free-text columns without detected PII are sampled with replacement and **may leak
  original values**. `doppel diff` reports a distance-to-closest-record percentile
  and a per-column verbatim-text fraction so you can spot row-level memorisation.
- No differential privacy in v0.1. If you need a formal privacy guarantee, doppel
  is not the right tool yet — `--epsilon` is v0.2 roadmap.

### 3. Constraint expression evaluator

`doppel` exposes a small Python-expression DSL in two places:

- `[[constraints]]` of `kind = "derived"` in `schema.toml` (arithmetic only)
- `--where EXPR` on `doppel gen` and `kind = "where"` constraints (boolean)

Both are parsed with the stdlib `ast` module and walked under a strict allowlist.
**No `eval`, `exec`, or `compile()` of user input.**

**Numeric mode (`derived`).** Allowed nodes: `Name`, `Constant(int|float)`,
`UnaryOp(USub)`, `BinOp(Add|Sub|Mult|Div)`.

**Boolean mode (`where`).** Numeric subgrammar plus: `Compare` with one of
`Eq|NotEq|Lt|LtE|Gt|GtE` (single op only — chained `0 < x < 10` is rejected),
`BoolOp(And|Or)`, and `Constant(str|bool)` as comparands.

**Explicitly rejected** (each by AST node type, with a clear error message):
`Call`, `Attribute`, `Subscript`, `Lambda`, `IfExp`, list/set/dict/tuple
literals, comprehensions, `is`, `is not`, `in`, `not in`, `**`, `%`, `//`,
`<<`, `>>`, `&`, `|`, `^`, `not`, f-strings, walrus (`:=`). The `__import__`
RCE pattern is rejected because `Call` itself is not allowed.

Regression coverage: `tests/test_constraints.py` (numeric) +
`tests/test_where_expr.py` (boolean, parametrised over every rejected node).

## Reproducibility

`--seed` makes all fit + sample randomness deterministic: sklearn estimators,
leaf-sampling, UUID-typed keys, Faker-generated PII. Same seed = byte-identical
output. If you find a path that breaks this, it's a bug — please report.

Full contract in [docs/determinism.md](docs/determinism.md).
