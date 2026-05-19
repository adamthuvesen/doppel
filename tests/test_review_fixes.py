"""Regression tests for the issues raised in the Phase-6 review.

One test per finding so the fix is documented in code, not just memory.
"""

from __future__ import annotations

import gzip
import io
import pickle
import tarfile
from pathlib import Path

import polars as pl
import pytest

from doppel.artifact import ARTIFACT_VERSION, ArtifactError, load, save
from doppel.artifact.safe_pickle import UnsafeArtifactError, safe_loads
from doppel.dataset import Dataset
from doppel.schema import toml as schema_toml_mod
from doppel.schema.infer import infer_table
from doppel.schema.toml import ColumnSpec, SchemaToml, TableMeta
from doppel.schema.types import ColumnType
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng

# ---------------------------------------------------------------------------
# C1 — restricted unpickler refuses dangerous classes
# ---------------------------------------------------------------------------


class _Exploit:
    """A would-be pickle-RCE payload. `__reduce__` would normally be called on load."""

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        # os.system('echo PWNED') — without `os` in the allowlist this must NOT execute.
        import os

        return os.system, ("exit 0",)


def test_safe_pickle_refuses_os_system() -> None:
    payload = pickle.dumps(_Exploit())
    with pytest.raises(UnsafeArtifactError, match="not in doppel artifact allowlist"):
        safe_loads(payload)


def test_safe_pickle_refuses_builtins_eval() -> None:
    # Hand-craft a pickle that calls builtins.eval — pickle.dumps(eval) directly.
    payload = pickle.dumps(eval)
    with pytest.raises(UnsafeArtifactError, match="not in doppel artifact allowlist"):
        safe_loads(payload)


# Allowlist is the security boundary: any regression that whitelists a stdlib module
# under which a pickle gadget exists would bypass it. Parametrize across representative
# modules so a wider whitelist trips the suite immediately.
@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(pickle.dumps(__import__("subprocess").Popen), id="subprocess.Popen"),
        pytest.param(pickle.dumps(__import__("shutil").rmtree), id="shutil.rmtree"),
        pytest.param(
            pickle.dumps(__import__("importlib").import_module), id="importlib.import_module"
        ),
        pytest.param(pickle.dumps(__import__("builtins").exec), id="builtins.exec"),
        pytest.param(pickle.dumps(__import__("builtins").open), id="builtins.open"),
    ],
)
def test_safe_pickle_refuses_disallowed_stdlib_classes(obj: bytes) -> None:
    with pytest.raises(UnsafeArtifactError, match="not in doppel artifact allowlist"):
        safe_loads(obj)


def test_artifact_load_rejects_malicious_pickle(tmp_path: Path) -> None:
    """A tampered `.doppel` artifact carrying an exploit payload must fail before exec."""
    # First produce a legitimate artifact so we have a valid manifest.
    df = pl.DataFrame({"x": list(range(10)), "y": [float(i) for i in range(10)]})
    synth = CartSynthesizer()
    synth.fit(Dataset.single(infer_table("t", df)), Rng.from_seed(0))
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=df.height)

    # Now replace `synth.pickle` with an exploit blob while keeping the valid manifest.
    members: dict[str, bytes] = {}
    with tarfile.open(artifact, "r:gz") as tar:
        for m in tar.getmembers():
            handle = tar.extractfile(m)
            assert handle is not None
            members[m.name] = handle.read()
    members["synth.pickle"] = pickle.dumps(_Exploit())
    with tarfile.open(artifact, "w:gz") as tar:
        for name, blob in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    # Sanity: the file is still a valid gzip tar.
    with gzip.open(artifact, "rb") as gz:
        gz.read(1)

    with pytest.raises(ArtifactError, match="disallowed class"):
        load(artifact)


def test_safe_pickle_round_trips_legitimate_artifact(tmp_path: Path) -> None:
    """The restricted unpickler must NOT break loading of legitimate doppel artifacts."""
    df = pl.DataFrame(
        {
            "id": list(range(20)),
            "amount": [float(i) * 1.5 for i in range(20)],
            "tag": ["a", "b", "c", "d"] * 5,
        }
    )
    synth = CartSynthesizer()
    synth.fit(Dataset.single(infer_table("t", df)), Rng.from_seed(0))
    p = tmp_path / "m.doppel"
    save(synth, p, training_row_count=df.height)
    loaded, manifest, _ = load(p)
    assert manifest.version == ARTIFACT_VERSION
    out_a = synth.sample(10, Rng.from_seed(1)).only().data
    out_b = loaded.sample(10, Rng.from_seed(1)).only().data
    assert out_a is not None and out_b is not None
    assert out_a.equals(out_b)


# ---------------------------------------------------------------------------
# H1 — UUID-typed KEY columns must respect --seed
# ---------------------------------------------------------------------------


def test_uuid_key_column_is_deterministic_under_same_seed() -> None:
    df = pl.DataFrame(
        {
            "uuid": [f"uuid-original-{i}" for i in range(40)],
            "value": [float(i) for i in range(40)],
        }
    )
    table = infer_table("t", df)
    # Sanity: the inferrer flagged `uuid` as a KEY column.
    by_name = {c.name: c for c in table.columns}
    assert by_name["uuid"].type is ColumnType.KEY

    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(123))
    out1 = synth.sample(25, Rng.from_seed(7)).only().data
    out2 = synth.sample(25, Rng.from_seed(7)).only().data
    assert out1 is not None and out2 is not None
    assert out1["uuid"].to_list() == out2["uuid"].to_list()

    # And: different seeds produce different UUIDs.
    out3 = synth.sample(25, Rng.from_seed(99)).only().data
    assert out3 is not None
    assert out1["uuid"].to_list() != out3["uuid"].to_list()


def test_uuid_key_column_values_look_like_hex_uuids() -> None:
    df = pl.DataFrame(
        {
            "uuid": [f"uuid-source-{i}" for i in range(10)],  # unique → classified as KEY
            "v": [1.0] * 10,
        }
    )
    table = infer_table("t", df)
    by_name = {c.name: c for c in table.columns}
    assert by_name["uuid"].type is ColumnType.KEY  # sanity
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    out = synth.sample(10, Rng.from_seed(0)).only().data
    assert out is not None
    for value in out["uuid"].to_list():
        # 32 lowercase hex chars, version-4 nibble set.
        assert len(value) == 32
        assert all(c in "0123456789abcdef" for c in value)
        assert value[12] == "4"  # version nibble


# ---------------------------------------------------------------------------
# H2 — Faker generation deterministic across same-seed calls in one process
# ---------------------------------------------------------------------------


def test_faker_generate_same_seed_produces_same_output_in_same_process() -> None:
    from doppel.pii.fake import generate

    a = generate("EMAIL_ADDRESS", 8, Rng.from_seed(42))
    b = generate("EMAIL_ADDRESS", 8, Rng.from_seed(42))
    assert a == b


def test_faker_generate_different_seeds_differ() -> None:
    from doppel.pii.fake import generate

    a = generate("PERSON", 5, Rng.from_seed(1))
    b = generate("PERSON", 5, Rng.from_seed(2))
    assert a != b


# ---------------------------------------------------------------------------
# M1 — declared primary_key is auto-promoted to KEY type
# ---------------------------------------------------------------------------


def test_apply_overrides_promotes_declared_primary_key_to_key() -> None:
    df = pl.DataFrame(
        {
            "order_no": [1001, 1002, 1003, 1004, 1005],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    inferred = infer_table("orders", df)
    # `order_no` would normally be classified as NUMERIC unless promoted by the schema.
    schema = SchemaToml(
        table=TableMeta(name="orders", primary_key="order_no"),
        columns={"amount": ColumnSpec(type=ColumnType.NUMERIC, nullable=False)},
    )
    merged = schema_toml_mod.apply_overrides(inferred, schema)
    pk_col = next(c for c in merged.columns if c.name == "order_no")
    assert pk_col.type is ColumnType.KEY
    assert merged.primary_key == "order_no"


def test_apply_overrides_rejects_unknown_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    inferred = infer_table("t", df)
    schema = SchemaToml(
        table=TableMeta(name="t"),
        columns={"does_not_exist": ColumnSpec(type=ColumnType.NUMERIC)},
    )
    with pytest.raises(ValueError, match="not in data"):
        schema_toml_mod.apply_overrides(inferred, schema)


def test_apply_overrides_rejects_unknown_primary_key() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    inferred = infer_table("t", df)
    schema = SchemaToml(table=TableMeta(name="t", primary_key="missing_id"))

    with pytest.raises(ValueError, match=r"primary_key .* not present"):
        schema_toml_mod.apply_overrides(inferred, schema)


def test_apply_overrides_rejects_non_unique_declared_primary_key() -> None:
    df = pl.DataFrame({"order_no": [1, 1, 2], "amount": [10.0, 20.0, 30.0]})
    inferred = infer_table("orders", df)
    schema = SchemaToml(table=TableMeta(name="orders", primary_key="order_no"))

    with pytest.raises(ValueError, match=r"primary_key .* must be unique"):
        schema_toml_mod.apply_overrides(inferred, schema)


def test_promoted_pk_yields_unique_synth_values() -> None:
    df = pl.DataFrame(
        {
            "order_no": [1001, 1002, 1003, 1004, 1005],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    inferred = infer_table("orders", df)
    schema = SchemaToml(table=TableMeta(name="orders", primary_key="order_no"))
    merged = schema_toml_mod.apply_overrides(inferred, schema)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(merged), Rng.from_seed(0))
    out = synth.sample(50, Rng.from_seed(0)).only().data
    assert out is not None
    assert out["order_no"].n_unique() == 50


def test_promoted_string_pk_yields_string_synth_values() -> None:
    df = pl.DataFrame(
        {
            "order_code": ["A001", "A002", "A003", "A004", "A005"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    inferred = infer_table("orders", df)
    schema = SchemaToml(table=TableMeta(name="orders", primary_key="order_code"))
    merged = schema_toml_mod.apply_overrides(inferred, schema)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(merged), Rng.from_seed(0))
    out = synth.sample(50, Rng.from_seed(0)).only().data
    assert out is not None
    assert out["order_code"].dtype == pl.String
    assert out["order_code"].n_unique() == 50


# ---------------------------------------------------------------------------
# M5 — PII confidence is clamped to [0, 1]
# ---------------------------------------------------------------------------


pii_available = pytest.importorskip("presidio_analyzer", reason="pii extra not installed")


def test_pii_confidence_never_exceeds_one() -> None:
    from doppel.pii.detect import detect

    # Cells that contain MULTIPLE emails would naively over-count.
    df = pl.DataFrame(
        {"contact": [f"primary {i}@example.com or backup {i}backup@example.org" for i in range(30)]}
    )
    table = infer_table("t", df)
    found = detect(df, table.columns, sample_size=20)
    for d in found:
        assert 0.0 <= d.confidence <= 1.0, f"confidence out of range: {d.confidence}"


_ = pii_available  # silence "unused" without disabling the importorskip side-effect.
