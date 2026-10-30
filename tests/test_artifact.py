"""Artifact module: save/load round-trip, version validation, error surface, no pickle."""

from __future__ import annotations

import ast
import gzip
import importlib
import io
import json
import pickle
import tarfile
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.artifact import ARTIFACT_VERSION, ArtifactError, Manifest, load, save
from doppel.cli import app
from doppel.dataset import Dataset
from doppel.schema.infer import infer_table
from doppel.synth.cart import CartSynthesizer
from doppel.synth.cart_state import StoredCart, take_values
from doppel.synth.seed import Rng

SRC = Path(__file__).resolve().parents[1] / "src" / "doppel"
load_module = importlib.import_module("doppel.artifact.load")


def _fit_synth(df: pl.DataFrame, seed: int = 42) -> tuple[CartSynthesizer, int]:
    table = infer_table("mixed", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(seed))
    return synth, df.height


def test_save_load_round_trip_preserves_sampling(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)

    loaded, manifest, schema_toml = load(artifact)
    assert manifest.version == ARTIFACT_VERSION
    assert manifest.synthesizer_class == "cart"
    assert manifest.training_row_count == n
    assert manifest.training_column_count == len(mixed_df.columns)
    assert schema_toml is None  # no schema was embedded

    out_a = synth.sample(50, Rng.from_seed(7)).only().data
    out_b = loaded.sample(50, Rng.from_seed(7)).only().data
    assert out_a is not None and out_b is not None
    assert out_a.equals(out_b), "round-tripped synthesizer must sample identically given same seed"


def test_save_refuses_unfitted_synthesizer(tmp_path: Path) -> None:
    synth = CartSynthesizer()
    with pytest.raises(ValueError, match="must be fitted"):
        save(synth, tmp_path / "out.doppel", training_row_count=0)


def test_load_rejects_unknown_version(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    # Tamper with the manifest version.
    _rewrite_manifest(artifact, lambda m: {**m, "version": "doppel-artifact-v999"})
    with pytest.raises(ArtifactError, match="unsupported artifact version"):
        load(artifact)


def test_load_rejects_corrupted_archive(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.doppel"
    bad.write_bytes(b"definitely not a tarball")
    with pytest.raises(ArtifactError, match="not a valid doppel artifact"):
        load(bad)


def test_load_rejects_oversized_manifest(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    _replace_member(artifact, "manifest.json", b"{" + (b" " * (1024 * 1024)) + b"}")

    with pytest.raises(ArtifactError, match="too large"):
        load(artifact)


def test_load_rejects_unknown_synthesizer_class(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    _rewrite_manifest(
        artifact,
        lambda m: {**m, "synthesizer_class": "secret-llm-magic"},
    )
    with pytest.raises(ArtifactError, match="unknown synthesizer_class"):
        load(artifact)


def test_artifact_contains_inspectable_schema_json(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    with tarfile.open(artifact, "r:gz") as tar:
        handle = tar.extractfile("schema.json")
        assert handle is not None
        schema = json.loads(handle.read())
    assert schema["table"] == "mixed"
    column_names = {c["name"] for c in schema["columns"]}
    assert column_names == set(mixed_df.columns)


def _rewrite_manifest(path: Path, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Open the tar.gz, replace manifest.json with a mutated copy, rewrite atomically."""
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member)
            assert handle is not None
            payloads[member.name] = handle.read()
    original = json.loads(payloads["manifest.json"])
    payloads["manifest.json"] = json.dumps(mutate(original)).encode()
    with tarfile.open(path, "w:gz") as tar:
        for name, blob in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    # gzip compatibility check — ensure the file is still a valid gzip blob.
    with gzip.open(path, "rb") as gz:
        gz.read(1)


def _replace_member(path: Path, target: str, payload: bytes) -> None:
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member)
            assert handle is not None
            payloads[member.name] = handle.read()
    payloads[target] = payload
    with tarfile.open(path, "w:gz") as tar:
        for name, blob in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))


def test_manifest_roundtrips_via_pydantic() -> None:
    m = Manifest(
        synthesizer_class="cart",
        doppel_version="0.0.0",
        table_name="t",
        training_row_count=10,
        training_column_count=3,
    )
    parsed = Manifest.model_validate_json(m.model_dump_json())
    assert parsed == m


def _wide_df() -> pl.DataFrame:
    """Every dtype the artifact stores, with nulls, unicode, and repair-rule columns."""
    gen = np.random.default_rng(3)
    n = 240
    base = datetime(2024, 3, 1, 8)
    created = [base + timedelta(minutes=int(m)) for m in gen.integers(0, 200_000, n)]
    closed = [
        c + timedelta(hours=int(h)) if keep else None
        for c, h, keep in zip(created, gen.integers(1, 500, n), gen.random(n) > 0.2, strict=True)
    ]
    notes = ["", "naïve café", "日本語", "plain"]
    seats = gen.integers(1, 50, n).tolist()
    mrr = [float(v) if v > 20 else None for v in gen.normal(100, 40, n)]
    return pl.DataFrame(
        {
            "uuid": [f"src-{i}" for i in range(n)],
            "plan_version": ["v1"] * n,
            "num_seats": seats,
            "num_viewer_seats": [int(gen.integers(0, s + 1)) for s in seats],
            "region": pl.Series(
                gen.choice(["NA", "EU", "APAC"], n).tolist(), dtype=pl.Enum(["NA", "EU", "APAC"])
            ),
            "tier": pl.Series(gen.choice(["free", "pro"], n).tolist(), dtype=pl.Categorical),
            "seats": pl.Series(gen.integers(1, 50, n).tolist(), dtype=pl.UInt64),
            "viewer_seats": pl.Series(gen.integers(0, 10, n).tolist(), dtype=pl.Int16),
            "ratio": pl.Series(gen.random(n).tolist(), dtype=pl.Float32),
            "mrr": mrr,
            "mrr_missing": [int(v is None) for v in mrr],
            "country": [
                str(v) if v != "XX" else None for v in gen.choice(["SE", "NO", "DK", "XX"], n)
            ],
            "is_active": (gen.random(n) > 0.3).tolist(),
            "has_trial": gen.integers(0, 2, n).tolist(),
            "signup_day": [
                date(2023, 1, 1) + timedelta(days=int(d)) for d in gen.integers(0, 600, n)
            ],
            "created_at": pl.Series(created).dt.replace_time_zone("Europe/Stockholm"),
            "closed_at": pl.Series(closed).dt.replace_time_zone("Europe/Stockholm"),
            "note": [
                notes[i % 4] + f" {gen.integers(0, 1000)}" if i % 9 else None for i in range(n)
            ],
        }
    )


def _fit_wide(seed: int) -> CartSynthesizer:
    synth = CartSynthesizer()
    synth.fit(Dataset.single(infer_table("wide", _wide_df())), Rng.from_seed(seed))
    return synth


def _csv(synth: CartSynthesizer, rows: int, seed: int) -> tuple[str, pl.Schema]:
    out = synth.sample(rows, Rng.from_seed(seed)).only().data
    assert out is not None
    return out.write_csv(), out.schema


@pytest.mark.parametrize("seed", [0, 11])
def test_round_trip_sample_is_byte_identical(tmp_path: Path, seed: int) -> None:
    synth = _fit_wide(seed)
    artifact = tmp_path / "wide.doppel"
    save(synth, artifact, training_row_count=240)
    loaded, _, _ = load(artifact)

    for sample_seed in (seed, seed + 1):
        assert _csv(loaded, 400, sample_seed) == _csv(synth, 400, sample_seed)
    assert loaded.explain_columns() == synth.explain_columns()


def test_round_trip_state_is_exact(tmp_path: Path) -> None:
    synth = _fit_wide(5)
    artifact = tmp_path / "wide.doppel"
    save(synth, artifact, training_row_count=240)
    loaded, _, _ = load(artifact)

    state, arrays = synth.to_state()
    state_again, arrays_again = loaded.to_state()
    assert state_again == state
    assert arrays_again.keys() == arrays.keys()
    for key, array in arrays.items():
        assert arrays_again[key].dtype == array.dtype, key
        assert arrays_again[key].tobytes() == array.tobytes(), key


def test_same_seed_fits_the_same_stored_model() -> None:
    state_a, arrays_a = _fit_wide(9).to_state()
    state_b, arrays_b = _fit_wide(9).to_state()
    assert state_a == state_b
    assert all(arrays_a[k].tobytes() == arrays_b[k].tobytes() for k in arrays_a)


def test_artifact_members_are_json_and_npz(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    with tarfile.open(artifact, "r:gz") as tar:
        names = tar.getnames()
    assert sorted(names) == ["arrays.npz", "manifest.json", "model.json", "schema.json"]


def test_load_never_calls_pickle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    synth = _fit_wide(1)
    artifact = tmp_path / "wide.doppel"
    save(synth, artifact, training_row_count=240)

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the artifact loader called pickle")

    for name in ("load", "loads", "Unpickler"):
        monkeypatch.setattr(pickle, name, refuse)
    loaded, _, _ = load(artifact)
    assert _csv(loaded, 50, 2) == _csv(synth, 50, 2)


def test_no_source_module_imports_a_code_executing_loader() -> None:
    banned = {"pickle", "marshal", "joblib", "dill", "cloudpickle", "shelve"}
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] in banned]
    assert offenders == []


class _TouchMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return Path.touch, (self.marker,)


def _write_members(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, blob in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))


def _old_pickle_artifact(path: Path, marker: Path) -> None:
    manifest = {
        "version": "doppel-artifact-v1",
        "synthesizer_class": "cart",
        "doppel_version": "0.1.0",
        "table_name": "t",
        "training_row_count": 10,
        "training_column_count": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    _write_members(
        path,
        {
            "manifest.json": json.dumps(manifest).encode(),
            "synth.pickle": pickle.dumps(_TouchMarker(marker)),
        },
    )


def test_load_refuses_the_old_pickle_format(tmp_path: Path) -> None:
    artifact = tmp_path / "old.doppel"
    marker = tmp_path / "pwned"
    _old_pickle_artifact(artifact, marker)
    with pytest.raises(ArtifactError, match="pickled synthesizer"):
        load(artifact)
    assert not marker.exists()


def test_sample_cli_refuses_the_old_pickle_format(tmp_path: Path) -> None:
    artifact = tmp_path / "old.doppel"
    marker = tmp_path / "pwned"
    _old_pickle_artifact(artifact, marker)
    result = CliRunner().invoke(
        app, ["sample", str(artifact), "--output", str(tmp_path / "out.csv"), "--rows", "5"]
    )
    assert result.exit_code == 2
    assert "no longer loads pickles" in result.output
    assert not isinstance(result.exception, ArtifactError)
    assert not marker.exists()


def test_load_refuses_an_npz_with_an_object_array(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    marker = tmp_path / "pwned"
    planted = io.BytesIO()
    np.savez(
        planted, allow_pickle=True, **{"c0.pool": np.array([_TouchMarker(marker)], dtype=object)}
    )
    _replace_member(artifact, "arrays.npz", planted.getvalue())

    with pytest.raises(ArtifactError, match="cannot read"):
        load(artifact)
    assert not marker.exists()


def test_load_refuses_a_bare_pickle_in_place_of_the_npz(
    mixed_df: pl.DataFrame, tmp_path: Path
) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    marker = tmp_path / "pwned"
    _replace_member(artifact, "arrays.npz", pickle.dumps(_TouchMarker(marker)))

    with pytest.raises(ArtifactError, match="not a plain npz archive"):
        load(artifact)
    assert not marker.exists()


def test_load_refuses_unexpected_and_duplicate_members(
    mixed_df: pl.DataFrame, tmp_path: Path
) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    members = _members(artifact)

    _write_members(artifact, {**members, "run.sh": b"echo hi"})
    with pytest.raises(ArtifactError, match=r"unexpected member 'run\.sh'"):
        load(artifact)

    with tarfile.open(artifact, "w:gz") as tar:
        for name, blob in [*members.items(), ("model.json", members["model.json"])]:
            info = tarfile.TarInfo(name=name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    with pytest.raises(ArtifactError, match="more than once"):
        load(artifact)


def test_load_refuses_an_artifact_past_the_total_cap(
    mixed_df: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    total = sum(len(blob) for blob in _members(artifact).values())
    monkeypatch.setattr(load_module, "_MAX_TOTAL_BYTES", total - 1)
    with pytest.raises(ArtifactError, match="across its members"):
        load(artifact)


def test_load_refuses_a_tampered_tree(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    synth, n = _fit_synth(mixed_df)
    artifact = tmp_path / "model.doppel"
    save(synth, artifact, training_row_count=n)
    with np.load(io.BytesIO(_members(artifact)["arrays.npz"]), allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    key = next(k for k in arrays if k.endswith("value_tree.left_child"))
    arrays[key] = np.zeros_like(arrays[key])
    planted = io.BytesIO()
    np.savez(planted, allow_pickle=False, **arrays)
    _replace_member(artifact, "arrays.npz", planted.getvalue())

    with pytest.raises(ArtifactError, match="inconsistent"):
        load(artifact)


def test_save_refuses_a_dtype_it_cannot_store(tmp_path: Path) -> None:
    from decimal import Decimal

    df = pl.DataFrame({"a": list(range(20)), "d": [Decimal("1.5") + i for i in range(20)]})
    synth = CartSynthesizer()
    synth.fit(Dataset.single(infer_table("t", df)), Rng.from_seed(0))
    target = tmp_path / "out.doppel"
    with pytest.raises(ArtifactError, match="cannot store a column of dtype Decimal"):
        save(synth, target, training_row_count=20)
    assert not target.exists()


def _members(path: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member)
            assert handle is not None
            out[member.name] = handle.read()
    return out


def test_no_stored_array_keeps_source_row_order(tmp_path: Path) -> None:
    n = 60
    df = pl.DataFrame(
        {
            "x": list(range(n)),
            "y": [i * 1.5 for i in range(n)],
            "g": [["a", "b", "c"][(i * 7) % 3] if i % 5 else "a" for i in range(n)],
            "note": [f"note {i}" if i % 7 else None for i in range(n)],
            "flag": [i % 3 == 0 for i in range(n)],
        }
    )
    synth = CartSynthesizer()
    synth.fit(Dataset.single(infer_table("t", df)), Rng.from_seed(0))
    artifact = tmp_path / "t.doppel"
    save(synth, artifact, training_row_count=n)
    members = _members(artifact)
    state = StoredCart.model_validate_json(members["model.json"])
    with np.load(io.BytesIO(members["arrays.npz"]), allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}

    stored: dict[str, list[object]] = {}
    for i, cs in enumerate(state.column_synths):
        stored[f"c{i}.pool"] = take_values(arrays, f"c{i}.pool", cs.pool)
        stored[f"c{i}.encoder"] = take_values(arrays, f"c{i}.encoder", cs.encoder_categories)
    stored |= {key: array.tolist() for key, array in arrays.items()}

    source_orders = [
        order
        for name in df.columns
        for order in (df[name].to_list(), df[name].drop_nulls().to_list())
    ]
    source_orders += [[float(v) for v in df[name].drop_nulls()] for name in ("x", "y")]
    leaks = [key for key, values in stored.items() if values in source_orders]
    assert leaks == []
    assert len(stored["c0.pool"]) == n
