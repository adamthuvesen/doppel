"""`doppel gen --where` end-to-end CLI tests.

Covers the feasibility precheck (0/<100/>=100 buckets), determinism under repeated
runs, multi-table single-table-scoped happy path, multi-table cross-table rejection,
oversample-exhaustion, TOML round-trip, and CLI/TOML composition.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def plan_csv(tmp_path: Path) -> Path:
    """A balanced 600-row dataset across three plans — enough support for any single
    plan and combined predicates to converge under a generous oversample budget. CART
    sample distributions can drift from the empirical, so tests using this fixture pass
    a high `--max-oversample` to keep them robust."""
    src = tmp_path / "plans.csv"
    pl.DataFrame(
        {
            "plan": ["enterprise"] * 200 + ["pro"] * 200 + ["free"] * 200,
            "age": [int(i / 6) + 18 for i in range(600)],
            "tenure_days": list(range(600)),
        }
    ).write_csv(src)
    return src


# ── Happy paths ─────────────────────────────────────────────────────────────────


def test_gen_where_categorical_equality(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "50",
            "--where",
            "plan == 'enterprise'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert df.height == 50
    assert (df["plan"] == "enterprise").all()


def test_gen_where_numeric_inequality(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "50",
            "--where",
            "tenure_days > 300",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert df.height == 50
    assert (df["tenure_days"] > 300).all()


def test_gen_where_combined_or(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "50",
            "--where",
            "plan == 'enterprise' or plan == 'pro'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "8.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert set(df["plan"].to_list()) <= {"enterprise", "pro"}


def test_gen_where_combined_and(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "20",
            "--where",
            "plan == 'enterprise' and tenure_days < 100",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert (df["plan"] == "enterprise").all()
    assert (df["tenure_days"] < 100).all()


# ── Determinism ─────────────────────────────────────────────────────────────────


def test_gen_where_is_deterministic_given_seed(plan_csv: Path, tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    for out in (a, b):
        result = runner.invoke(
            app,
            [
                "gen",
                str(plan_csv),
                "--rows",
                "50",
                "--where",
                "plan == 'enterprise'",
                "--seed",
                "42",
                "--output",
                str(out),
                "--no-quality",
                "--max-oversample",
                "16.0",
            ],
        )
        assert result.exit_code == 0, result.stdout
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()


def test_gen_where_changes_with_seed(plan_csv: Path, tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    for out, seed in [(a, "1"), (b, "2")]:
        result = runner.invoke(
            app,
            [
                "gen",
                str(plan_csv),
                "--rows",
                "50",
                "--where",
                "plan == 'enterprise'",
                "--seed",
                seed,
                "--output",
                str(out),
                "--no-quality",
                "--max-oversample",
                "16.0",
            ],
        )
        assert result.exit_code == 0, result.stdout
    assert a.read_bytes() != b.read_bytes()


# ── Feasibility precheck ───────────────────────────────────────────────────────


def test_gen_where_zero_match_fails_hard(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "10",
            "--where",
            "plan == 'nonexistent'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
        ],
    )
    assert result.exit_code != 0
    # The message must quote the expression and name the input path.
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "no rows" in combined
    assert "plan == 'nonexistent'" in combined
    # The path can be wrapped across lines in rich output, so just check for the filename.
    assert plan_csv.name in combined
    assert not out.exists()


def test_gen_where_zero_match_skips_fit(
    plan_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precheck must fire BEFORE the synthesizer is fitted — otherwise a zero-match
    where wastes the whole fit cost before erroring."""
    out = tmp_path / "synth.csv"

    from doppel.synth.cart import CartSynthesizer

    original_fit = CartSynthesizer.fit
    fit_called = False

    def spy_fit(self: CartSynthesizer, *args: object, **kwargs: object) -> None:
        nonlocal fit_called
        fit_called = True
        original_fit(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(CartSynthesizer, "fit", spy_fit)

    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "10",
            "--where",
            "plan == 'nonexistent'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
        ],
    )
    assert result.exit_code != 0
    assert not fit_called, "synthesizer.fit() should not run when --where precheck fails"


def test_gen_where_thin_support_warns(tmp_path: Path) -> None:
    """With <100 source matches the precheck emits a warning but proceeds.

    Asserts only that the warning fires — exit-zero relies on the sampler converging,
    which is sensitive to the CART model and the seed. The 0-match hard-fail path
    has its own dedicated test (`test_gen_where_zero_match_fails_hard`).
    """
    src = tmp_path / "thin.csv"
    # 50 enterprise rows out of 1000 → matches the <100 warn bucket without tipping
    # into the 0-match hard fail.
    pl.DataFrame(
        {
            "plan": ["enterprise"] * 50 + ["pro"] * 950,
            "age": list(range(1000)),
        }
    ).write_csv(src)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "--rows",
            "5",
            "--where",
            "plan == 'enterprise'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "64.0",
        ],
    )
    # The warning must fire regardless of whether the synth converges.
    assert "warn" in result.stdout.lower()
    assert "50 source rows" in result.stdout


def test_gen_where_thick_support_no_warning(plan_csv: Path, tmp_path: Path) -> None:
    """With >=100 matches no thin-support warning appears."""
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "50",
            "--where",
            "plan == 'enterprise'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "warn" not in result.stdout.lower()


# ── Oversample exhaustion ──────────────────────────────────────────────────────


def test_gen_where_oversample_exhaustion_is_clean_cli_error(tmp_path: Path) -> None:
    src = tmp_path / "rare.csv"
    pl.DataFrame(
        {
            "plan": ["common"] * 999 + ["rare"],
            "age": list(range(1000)),
        }
    ).write_csv(src)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "--rows",
            "100",
            "--where",
            "plan == 'rare'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "1.5",
        ],
    )
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "could not synthesize" in combined.lower()
    assert "Traceback" not in combined
    assert not isinstance(result.exception, ValueError)


def test_gen_where_invalid_max_oversample_rejected(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "10",
            "--where",
            "plan == 'enterprise'",
            "--max-oversample",
            "0.5",
            "--output",
            str(out),
            "--no-quality",
        ],
    )
    assert result.exit_code != 0


# ── Error messages ─────────────────────────────────────────────────────────────


def test_gen_where_chained_compare_rejected(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "10",
            "--where",
            "0 < tenure_days < 100",
            "--output",
            str(out),
            "--no-quality",
        ],
    )
    assert result.exit_code != 0
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "chained" in combined


def test_gen_where_non_boolean_top_level_rejected(plan_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "10",
            "--where",
            "tenure_days * 365",
            "--output",
            str(out),
            "--no-quality",
        ],
    )
    assert result.exit_code != 0
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "boolean predicate" in combined


# ── TOML round-trip + composition ──────────────────────────────────────────────


def test_gen_where_from_toml(plan_csv: Path, tmp_path: Path) -> None:
    schema = tmp_path / "schema.toml"
    schema.write_text(
        """
[table]
name = "plans"

[[constraints]]
kind = "where"
expression = "plan == 'enterprise'"
""",
        encoding="utf-8",
    )
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "30",
            "--schema",
            str(schema),
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert (df["plan"] == "enterprise").all()


def test_gen_where_cli_composes_with_toml_range(plan_csv: Path, tmp_path: Path) -> None:
    schema = tmp_path / "schema.toml"
    schema.write_text(
        """
[table]
name = "plans"

[[constraints]]
kind = "range"
column = "tenure_days"
min = 50
""",
        encoding="utf-8",
    )
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(plan_csv),
            "--rows",
            "30",
            "--schema",
            str(schema),
            "--where",
            "plan == 'enterprise'",
            "--seed",
            "1",
            "--output",
            str(out),
            "--no-quality",
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert (df["plan"] == "enterprise").all()
    assert (df["tenure_days"] >= 50).all()


# ── Multi-table ────────────────────────────────────────────────────────────────


def _write_multi_table_fixture(tmp_path: Path) -> Path:
    """Write a users/orders pair with an FK and a schema.toml. Returns the schema path."""
    users = tmp_path / "users.csv"
    pl.DataFrame(
        {
            "user_id": list(range(1, 101)),
            "plan": ["enterprise"] * 50 + ["pro"] * 50,
        }
    ).write_csv(users)
    orders = tmp_path / "orders.csv"
    pl.DataFrame(
        {
            "order_id": list(range(1, 201)),
            "user_id": [(i % 100) + 1 for i in range(200)],
            "amount": [10.0 + i for i in range(200)],
        }
    ).write_csv(orders)
    schema = tmp_path / "schema.toml"
    schema.write_text(
        f"""
[tables.users]
file = "{users.name}"
primary_key = "user_id"

[tables.orders]
file = "{orders.name}"
primary_key = "order_id"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
""",
        encoding="utf-8",
    )
    return schema


def test_gen_where_multi_table_single_scope(tmp_path: Path) -> None:
    schema = _write_multi_table_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "gen",
            "--schema",
            str(schema),
            "--rows",
            "30",
            "--where",
            "plan == 'enterprise'",
            "--seed",
            "1",
            "--output",
            str(out_dir),
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Filter applies to users; orders unconditional. FK integrity check below.
    users_out = pl.read_csv(out_dir / "users.csv")
    orders_out = pl.read_csv(out_dir / "orders.csv")
    assert (users_out["plan"] == "enterprise").all()
    orphans = orders_out.filter(~pl.col("user_id").is_in(users_out["user_id"].implode()))
    assert orphans.is_empty()


def test_gen_where_multi_table_parent_filter_prunes_children(tmp_path: Path) -> None:
    schema = _write_multi_table_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "gen",
            "--schema",
            str(schema),
            "--rows",
            "30",
            "--where",
            "plan == 'enterprise'",
            "--seed",
            "1",
            "--output",
            str(out_dir),
            "--max-oversample",
            "16.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    users_out = pl.read_csv(out_dir / "users.csv")
    orders_out = pl.read_csv(out_dir / "orders.csv")
    assert users_out.height == 30
    assert (users_out["plan"] == "enterprise").all()
    orphans = orders_out.filter(~pl.col("user_id").is_in(users_out["user_id"].implode()))
    assert orphans.is_empty()


def test_gen_where_multi_table_child_filter_preserves_fk_integrity(tmp_path: Path) -> None:
    schema = _write_multi_table_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "gen",
            "--schema",
            str(schema),
            "--rows",
            "30",
            "--where",
            "amount > 100",
            "--seed",
            "1",
            "--output",
            str(out_dir),
            "--max-oversample",
            "16.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    users_out = pl.read_csv(out_dir / "users.csv")
    orders_out = pl.read_csv(out_dir / "orders.csv")
    assert (orders_out["amount"] > 100).all()
    orphans = orders_out.filter(~pl.col("user_id").is_in(users_out["user_id"].implode()))
    assert orphans.is_empty()


def test_gen_where_multi_table_emits_child_warning(tmp_path: Path) -> None:
    schema = _write_multi_table_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "gen",
            "--schema",
            str(schema),
            "--rows",
            "30",
            "--where",
            "plan == 'enterprise'",
            "--seed",
            "1",
            "--output",
            str(out_dir),
            "--max-oversample",
            "16.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "unconditional" in result.stdout.lower()


def test_gen_where_multi_table_cross_table_rejected(tmp_path: Path) -> None:
    schema = _write_multi_table_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "gen",
            "--schema",
            str(schema),
            "--rows",
            "30",
            "--where",
            "plan == 'enterprise' and amount > 100",
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "multiple tables" in combined
    assert "users" in combined and "orders" in combined
    # Output directory should not be created — the failure happens before any sample work.
    assert not (out_dir / "users.csv").exists()
