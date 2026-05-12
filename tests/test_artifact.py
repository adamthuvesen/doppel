"""Artifact module: save/load round-trip, version validation, error surface."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from doppel.artifact import ARTIFACT_VERSION, ArtifactError, Manifest, load, save
from doppel.dataset import Dataset
from doppel.schema.infer import infer_table
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng


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
