"""The drives. Each returns a one-line detail on success and raises DriveFailure otherwise."""

import hashlib
import io
import json
import pickle
import re
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import polars as pl
from fixtures import ACCOUNT_ROWS, CUSTOMER_ROWS, FIXTURE_DIR
from harness import Ctx, squash

from doppel.cli.diff import THRESHOLD_BREACH_EXIT_CODE
from doppel.pii.detect import SPACY_MODEL, SPACY_MODEL_INSTALL_COMMAND

ACCOUNTS = FIXTURE_DIR / "accounts.csv"
CUSTOMERS = FIXTURE_DIR / "customers.csv"
SCHEMA = FIXTURE_DIR / "schema.toml"
HASH_SALT = "verify-salt"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Strings the product prints when it refuses. No constant exists in the code, so each is held here.
REFUSAL_PICKLE_MEMBER = "no longer loads pickles"
REFUSAL_OBJECT_ARRAY = "holds an array doppel cannot read"
REFUSAL_SINK_IS_SOURCE = "is the source table"
REFUSAL_INVALID_TABLE = "invalid table name"
# The install hint and the model name come from the code (`doppel.pii.detect`).
INSTALL_HINT = SPACY_MODEL_INSTALL_COMMAND

# Failure tokens the control and the reader match on.
NOT_REPRODUCIBLE = "output-differs-between-processes"
SEED_IGNORED = "different-seed-gave-same-output"
ROW_COUNT = "row-count-mismatch"
COLUMNS_MISSING = "source-columns-missing"
ARTIFACT_INFO = "artifact-info-incomplete"
PICKLE_MEMBER = "artifact-holds-a-pickle-member"
FK_ORPHAN = "foreign-key-orphan"
PII_NOT_DETECTED = "pii-not-detected"
REAL_VALUE_LEAKED = "real-value-in-output"
REPORT_MISSING = "report-metric-missing"
LEAK_NOT_CAUGHT = "verbatim-leak-not-caught"
NOT_REFUSED = "not-refused"
CODE_RAN = "planted-code-ran"
PAYLOAD_INERT = "planted-payload-is-inert"
SOURCE_CHANGED = "source-database-changed"
NOT_HARMLESS = "safe-path-failed"
HINT_MISSING = "install-hint-missing"
TRACEBACK = "traceback-in-output"


@dataclass(frozen=True)
class Drive:
    name: str
    proves: str
    paths: tuple[str, ...]
    run: Callable[[Ctx], str]


def _csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=None)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(
    ctx: Ctx, step: str, artifact: Path, out: Path, seed: int
) -> subprocess.CompletedProcess[str]:
    return ctx.doppel(
        step, "sample", str(artifact), "-o", str(out), "-n", "40", "--seed", str(seed)
    )


def _fit(ctx: Ctx, step: str, source: Path, out: Path, seed: int) -> None:
    ctx.doppel(step, "fit", str(source), "-o", str(out), "--seed", str(seed))


def gen(ctx: Ctx) -> str:
    rows = 60
    outs = [ctx.work / name for name in ("a.csv", "b.csv", "other.csv")]

    def run(out: Path, seed: int) -> Callable[[], None]:
        def job() -> None:
            ctx.doppel(
                f"gen-{out.stem}",
                *("gen", str(ACCOUNTS), "-o", str(out), "-n", str(rows), "--seed", str(seed)),
                *("--text-policy", "hash", "--hash-salt", HASH_SALT),
            )

        return job

    ctx.parallel([run(outs[0], 7), run(outs[1], 7), run(outs[2], 8)])
    first, second, other = outs
    ctx.check(_sha(first) == _sha(second), NOT_REPRODUCIBLE, "same seed and salt, different bytes")
    ctx.check(_sha(first) != _sha(other), SEED_IGNORED, "seed 7 and seed 8 gave the same bytes")
    frame = _csv(first)
    ctx.check(frame.height == rows, ROW_COUNT, f"{frame.height} rows, expected {rows}")
    source_columns = _csv(ACCOUNTS).columns
    ctx.check(frame.columns == source_columns, COLUMNS_MISSING, f"{frame.columns}")
    return f"two processes gave {_sha(first)[:12]}; {rows} rows; {len(source_columns)} columns"


def fit_sample(ctx: Ctx) -> str:
    artifact = ctx.work / "model.doppel"
    _fit(ctx, "fit", ACCOUNTS, artifact, 3)
    outs = [ctx.work / name for name in ("s1.csv", "s2.csv", "s3.csv")]
    info = ctx.parallel(
        [
            lambda: _sample(ctx, "sample-1", artifact, outs[0], 9),
            lambda: _sample(ctx, "sample-2", artifact, outs[1], 9),
            lambda: _sample(ctx, "sample-3", artifact, outs[2], 10),
            lambda: ctx.doppel("info", "artifact", "info", str(artifact)),
        ]
    )[3]
    ctx.check(_sha(outs[0]) == _sha(outs[1]), NOT_REPRODUCIBLE, "same seed, different sample")
    ctx.check(_sha(outs[0]) != _sha(outs[2]), SEED_IGNORED, "seed 9 and seed 10 matched")
    ctx.check(_csv(outs[0]).height == 40, ROW_COUNT, "sample is not 40 rows")
    wanted = ("artifact-version", f"training-rows:    {ACCOUNT_ROWS}", "synthesizer:")
    missing = [w for w in wanted if w not in info.stdout]
    ctx.check(not missing, ARTIFACT_INFO, f"artifact info lacks {missing}")
    with tarfile.open(artifact) as tar:
        members = tar.getnames()
        arrays = tar.extractfile("arrays.npz")
        ctx.check(arrays is not None, ARTIFACT_INFO, f"no arrays.npz in {members}")
        npz = np.load(io.BytesIO(cast("Any", arrays).read()), allow_pickle=False)
        loaded = [npz[name] for name in npz.files]
    pickled = [m for m in members if re.search(r"pickle|\.pkl$", m)]
    ctx.check(not pickled, PICKLE_MEMBER, f"members {pickled}")
    ctx.check(
        all(a.dtype != object for a in loaded), PICKLE_MEMBER, "arrays.npz holds an object array"
    )
    return f"members {sorted(members)}; {len(loaded)} plain arrays; same-seed samples match"


def multi_table(ctx: Ctx) -> str:
    out = ctx.work / "out"
    done = ctx.doppel(
        "gen", *("gen", "--schema", str(SCHEMA), "-o", str(out), "-n", "30", "--seed", "5")
    )
    detected = re.search(r"pii: customers: detected (.*)", done.stdout + done.stderr)
    ctx.check(
        bool(detected) and "PERSON" in detected.group(1) and "EMAIL_ADDRESS" in detected.group(1),
        PII_NOT_DETECTED,
        f"detected line: {detected.group(1) if detected else None}",
    )
    customers, orders = _csv(out / "customers.csv"), _csv(out / "orders.csv")
    ctx.check(customers.height == 30, ROW_COUNT, f"{customers.height} customers")
    ctx.check(orders.height > 0, ROW_COUNT, "no orders")
    orphans = set(orders["customer_id"]) - set(customers["customer_id"])
    ctx.check(not orphans, FK_ORPHAN, f"orders point at missing customers {sorted(orphans)[:5]}")
    real = _csv(CUSTOMERS)
    emails = customers["email"].drop_nulls().to_list()
    ctx.check(
        bool(emails) and all(EMAIL.fullmatch(e) for e in emails),
        PII_NOT_DETECTED,
        "output emails are missing or not email-shaped",
    )
    real_emails = set(real["email"])
    ctx.check(
        not real_emails & set(emails),
        REAL_VALUE_LEAKED,
        f"{len(real_emails & set(emails))} real emails in the output",
    )
    real_names = set(real["full_name"])
    ctx.check(
        not real_names & set(customers["full_name"]),
        REAL_VALUE_LEAKED,
        "a real person name is in the output",
    )
    ctx.check(not ctx.guard_hits(), "unexpected-guard-hit", f"{ctx.guard_hits()}")
    return (
        f"{customers.height} customers, {orders.height} orders, 0 orphans; "
        f"0 of {len(emails)} output emails among {CUSTOMER_ROWS} real ones"
    )


def diff(ctx: Ctx) -> str:
    clean, leak = ctx.work / "clean.csv", ctx.work / "leak.csv"
    ctx.doppel(
        "gen-clean",
        *("gen", str(ACCOUNTS), "-o", str(clean), "-n", "60", "--seed", "4"),
        *("--text-policy", "hash", "--hash-salt", HASH_SALT),
    )
    _csv(ACCOUNTS).head(30).write_csv(leak)

    def report(name: str, synth: Path, expect_exit: int, *flags: str) -> Any:
        done = ctx.doppel(
            f"diff-{name}",
            *("diff", str(ACCOUNTS), str(synth)),
            *("--json", str(ctx.work / f"{name}.json"), "-o", str(ctx.work / f"{name}.html")),
            *flags,
            expect_exit=expect_exit,
        )
        return done

    flag = "--fail-on-verbatim-text"
    clean_run, leak_run, leak_open = ctx.parallel(
        [
            lambda: report("clean", clean, 0, flag),
            lambda: report("leak", leak, THRESHOLD_BREACH_EXIT_CODE, flag),
            lambda: report("leak-open", leak, 0),
        ]
    )
    ctx.check("exact row matches" in clean_run.stdout, REPORT_MISSING, "terminal lacks the rate")
    clean_json = json.loads((ctx.work / "clean.json").read_text())
    ctx.check(
        isinstance(clean_json.get("exact_row_match_rate"), float),
        REPORT_MISSING,
        f"json keys {sorted(clean_json)}",
    )
    html = (ctx.work / "clean.html").read_text()
    ctx.check("Exact row matches" in html, REPORT_MISSING, "html lacks the rate")
    leak_json = json.loads((ctx.work / "leak-open.json").read_text())
    ctx.check(
        leak_json["exact_row_match_rate"] > 0.99,
        REPORT_MISSING,
        f"planted rows report exact_row_match_rate {leak_json['exact_row_match_rate']}",
    )
    gate = squash(leak_run.stderr + leak_run.stdout)
    ctx.check("verbatim_text" in gate, LEAK_NOT_CAUGHT, f"gate output: {gate[-200:]}")
    ctx.check(leak_open.returncode == 0, LEAK_NOT_CAUGHT, "leak passes without the flag")
    return (
        f"clean exit 0 (rate {clean_json['exact_row_match_rate']:.3f}); "
        f"planted leak exit {THRESHOLD_BREACH_EXIT_CODE}"
    )


class _PlantedCode:
    """Unpickling it runs `exec` and creates `marker`."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[str]]:
        return (exec, (f"open({str(self.marker)!r}, 'w').write('ran')",))


def _planted_marker_payload(marker: Path) -> bytes:
    return pickle.dumps(_PlantedCode(marker))


def _rewrite_artifact(source: Path, target: Path, replace: dict[str, bytes]) -> None:
    """Copy a valid artifact, replacing or adding members."""
    with tarfile.open(source) as src, tarfile.open(target, "w:gz") as out:
        for member in src:
            handle = src.extractfile(member)
            if handle is None or member.name in replace:
                continue
            out.addfile(member, io.BytesIO(handle.read()))
        for name, payload in replace.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            out.addfile(info, io.BytesIO(payload))


def _object_array_npz(marker: Path) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, evil=np.array([_PlantedCode(marker)], dtype=object), allow_pickle=True)
    return buffer.getvalue()


def artifact_safety(ctx: Ctx) -> str:
    good = ctx.work / "good.doppel"
    _fit(ctx, "fit", ACCOUNTS, good, 3)

    # The payload must be live: loading it in a child writes the marker. Without this, an absent
    # marker after the refusal would prove nothing.
    probe_marker = ctx.work / "probe.marker"
    payload = _planted_marker_payload(probe_marker)
    ctx.session.run(
        ctx.drive,
        "payload-is-live",
        [
            sys.executable,
            "-c",
            "import pickle,sys; pickle.loads(bytes.fromhex(sys.argv[1]))",
            payload.hex(),
        ],
        cwd=ctx.work,
    )
    ctx.check(probe_marker.exists(), PAYLOAD_INERT, "the planted pickle did not run when loaded")

    cases = {"pickle-member": REFUSAL_PICKLE_MEMBER, "object-array": REFUSAL_OBJECT_ARRAY}
    markers = {name: ctx.work / f"{name}.marker" for name in cases}
    _rewrite_artifact(
        good,
        ctx.work / "pickle-member.doppel",
        {"synth.pickle": _planted_marker_payload(markers["pickle-member"])},
    )
    _rewrite_artifact(
        good,
        ctx.work / "object-array.doppel",
        {"arrays.npz": _object_array_npz(markers["object-array"])},
    )
    details: list[str] = []
    for name, token in cases.items():
        bad = ctx.work / f"{name}.doppel"
        out = ctx.work / f"{name}.csv"
        done = ctx.doppel(
            f"sample-{name}",
            *("sample", str(bad), "-o", str(out), "-n", "5", "--seed", "1"),
            expect_exit=None,
        )
        text = squash(done.stderr + done.stdout)
        ctx.check(done.returncode != 0, NOT_REFUSED, f"{name}: sample exited 0")
        ctx.check(token in text, NOT_REFUSED, f"{name}: no {token!r} in {text[-300:]}")
        ctx.check("Traceback" not in text, TRACEBACK, f"{name}: traceback in output")
        ctx.check(not out.exists(), NOT_REFUSED, f"{name}: wrote an output file")
        ctx.check(not markers[name].exists(), CODE_RAN, f"{name}: the planted code ran")
        details.append(f"{name} exit {done.returncode}")
    return "; ".join(details) + "; no marker file"


def _duckdb_url(path: Path, table: str | None = None) -> str:
    return f"duckdb:///{path}" + (f"?table={table}" if table else "")


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    row = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    return int(row[0]) if row else -1


def _tables(path: Path) -> dict[str, int]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        names = [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()]
        return {n: _count(con, n) for n in sorted(names)}
    finally:
        con.close()


def build_source_database(path: Path) -> None:
    con = duckdb.connect(str(path))
    try:
        for table, csv_path in (("accounts", ACCOUNTS), ("customers", CUSTOMERS)):
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_csv(?, header=true)", [str(csv_path)]
            )
    finally:
        con.close()


def sink_equals_source(ctx: Ctx, source: Path, *, expect_exit: int | None) -> Any:
    """Write synthetic accounts back into the table they were read from, spelled differently."""
    return ctx.doppel(
        "sink-is-source",
        *("gen", _duckdb_url(source), "--table", "accounts", "-n", "5", "--seed", "1"),
        *("-o", _duckdb_url(source, "Main.ACCOUNTS")),
        expect_exit=expect_exit,
    )


def sql_safety(ctx: Ctx) -> str:
    source = ctx.work / "source.duckdb"
    build_source_database(source)
    before_sha, before_tables = _sha(source), _tables(source)
    # A name with a quote is refused as an invalid table name. A name with only a semicolon is a
    # valid quoted identifier, so it is harmless: DuckDB finds no such table.
    payloads = {
        "quote-break": ('accounts"; DROP TABLE accounts; --', REFUSAL_INVALID_TABLE),
        "semicolon": ("accounts; DROP TABLE accounts", None),
    }

    def injected(name: str, table: str) -> Any:
        return ctx.doppel(
            f"inject-{name}",
            *(
                "gen",
                _duckdb_url(source),
                "--table",
                table,
                "-n",
                "5",
                "-o",
                str(ctx.work / f"{name}.csv"),
            ),
            expect_exit=None,
        )

    def safe_copy() -> Any:
        return ctx.doppel(
            "safe-copy",
            *("gen", _duckdb_url(source), "--table", "accounts", "-n", "20", "--seed", "1"),
            *("-o", _duckdb_url(ctx.work / "dest.duckdb", "synth_accounts")),
        )

    results = ctx.parallel(
        [
            *(lambda n=n, t=t: injected(n, t) for n, (t, _) in payloads.items()),
            lambda: sink_equals_source(ctx, source, expect_exit=None),
            safe_copy,
        ]
    )
    for (name, (_, token)), done in zip(payloads.items(), results, strict=False):
        text = squash(done.stderr + done.stdout)
        ctx.check(done.returncode != 0, NOT_REFUSED, f"{name}: exit 0")
        ctx.check(token is None or token in text, NOT_REFUSED, f"{name}: {text[-300:]}")
        # A missing table prints a rich traceback today (reported as a product gap), so only the
        # refusal path is held to a clean message.
        ctx.check(token is None or "Traceback" not in text, TRACEBACK, f"{name}: traceback")
        ctx.check(not (ctx.work / f"{name}.csv").exists(), NOT_REFUSED, f"{name}: wrote output")
    sink_run = results[len(payloads)]
    sink_text = squash(sink_run.stderr + sink_run.stdout)
    ctx.check(sink_run.returncode != 0, NOT_REFUSED, "sink equal to source exited 0")
    ctx.check(REFUSAL_SINK_IS_SOURCE in sink_text, NOT_REFUSED, f"sink: {sink_text[-300:]}")
    ctx.check(
        _tables(ctx.work / "dest.duckdb") == {"synth_accounts": 20},
        NOT_HARMLESS,
        "the safe copy did not write 20 rows",
    )
    ctx.check(_tables(source) == before_tables, SOURCE_CHANGED, f"tables {_tables(source)}")
    ctx.check(_sha(source) == before_sha, SOURCE_CHANGED, "source file bytes changed")
    return f"{len(payloads)} injected names refused or harmless; sink==source refused; source bytes and {before_tables} unchanged"


def pii_model(ctx: Ctx) -> str:
    out = ctx.work / "out.csv"
    done = ctx.doppel(
        "gen-without-model",
        *("gen", str(CUSTOMERS), "-o", str(out), "-n", "10", "--seed", "1"),
        extra_env={"VERIFY_HIDE_DISTRIBUTION": SPACY_MODEL},
        expect_exit=None,
    )
    text = squash(done.stderr + done.stdout)
    ctx.check(done.returncode != 0, NOT_REFUSED, "gen exited 0 without the PII model")
    ctx.check(INSTALL_HINT in text and SPACY_MODEL in text, HINT_MISSING, text[-300:])
    ctx.check("Traceback" not in text, TRACEBACK, "traceback in output")
    ctx.check(not out.exists(), NOT_REFUSED, "wrote an output file without PII detection")
    ctx.check(not ctx.guard_hits(), "network-or-process-started", f"{ctx.guard_hits()}")
    return f"exit {done.returncode} with the hint {INSTALL_HINT!r}; no socket or process started"


DRIVES: dict[str, Drive] = {
    d.name: d
    for d in (
        Drive(
            "gen",
            "The same seed and salt give byte-identical output from two processes, with the requested rows and the source columns.",
            (
                "src/doppel/cli/gen.py",
                "src/doppel/pipeline/",
                "src/doppel/synth/",
                "src/doppel/text_policy.py",
            ),
            gen,
        ),
        Drive(
            "fit-sample",
            "fit then sample reproduces for the same seed; artifact info works; the artifact holds no pickle.",
            ("src/doppel/artifact/", "src/doppel/cli/fit.py", "src/doppel/cli/artifact.py"),
            fit_sample,
        ),
        Drive(
            "multi-table",
            "Every child foreign key exists in the parent output, and detected PII is regenerated.",
            (
                "src/doppel/schema/",
                "src/doppel/dataset.py",
                "src/doppel/pii/",
                "src/doppel/synth/hierarchy.py",
            ),
            multi_table,
        ),
        Drive(
            "diff",
            "Terminal, JSON, and HTML reports hold the exact-row-match rate; --fail-on-verbatim-text catches a planted leak.",
            (
                "src/doppel/quality/",
                "src/doppel/report/",
                "src/doppel/cli/diff.py",
                "src/doppel/pipeline/compare.py",
            ),
            diff,
        ),
        Drive(
            "artifact-safety",
            "A pickle member or an object-array npz is refused, and the planted code does not run.",
            ("src/doppel/artifact/",),
            artifact_safety,
        ),
        Drive(
            "sql-safety",
            "An injected DuckDB table name and a sink equal to the source are refused; the source file is unchanged.",
            (
                "src/doppel/sources/",
                "src/doppel/sinks/",
                "src/doppel/locations.py",
                "src/doppel/pipeline/sink_guard.py",
            ),
            sql_safety,
        ),
        Drive(
            "pii-model",
            "A missing spaCy model fails with the install hint and starts no download.",
            ("src/doppel/pii/",),
            pii_model,
        ),
    )
}
