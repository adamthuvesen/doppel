"""Load a `.doppel` artifact: validate manifest, then unpickle the synthesizer.

The pickle blob is deserialised through `safe_pickle.safe_loads`, which refuses any
class outside an explicit allowlist (sklearn / numpy / polars / doppel / scipy + safe
builtins). This blocks the standard pickle-RCE vector: a crafted `__reduce__` payload
calling `os.system` or similar will raise `UnsafeArtifactError` before any code runs.
"""

from __future__ import annotations

import tarfile
import tomllib
from pathlib import Path

from doppel.artifact.manifest import ARTIFACT_VERSION, Manifest
from doppel.artifact.safe_pickle import UnsafeArtifactError, safe_loads
from doppel.schema.toml import SchemaToml
from doppel.synth.cart import CartSynthesizer

_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_SCHEMA_BYTES = 5 * 1024 * 1024
_MAX_PICKLE_BYTES = 512 * 1024 * 1024


class ArtifactError(ValueError):
    """Raised when a `.doppel` artifact is malformed, mis-versioned, or untrusted."""


def load(path: Path) -> tuple[CartSynthesizer, Manifest, SchemaToml | None]:
    try:
        with tarfile.open(path, "r:gz") as tar:
            manifest = _read_manifest(tar)
            _validate(manifest)
            synth_blob = _read_member(tar, "synth.pickle", max_bytes=_MAX_PICKLE_BYTES)
            schema_blob = _read_optional(tar, "schema.toml", max_bytes=_MAX_SCHEMA_BYTES)
    except tarfile.TarError as exc:
        raise ArtifactError(f"{path} is not a valid doppel artifact: {exc}") from exc

    try:
        synth = safe_loads(synth_blob)
    except UnsafeArtifactError as exc:
        raise ArtifactError(f"{path} contains a disallowed class: {exc}") from exc
    if not isinstance(synth, CartSynthesizer):
        raise ArtifactError(f"artifact contains {type(synth).__name__}, expected CartSynthesizer")

    schema_toml: SchemaToml | None = None
    if schema_blob is not None:
        schema_toml = SchemaToml.model_validate(tomllib.loads(schema_blob.decode("utf-8")))
    return synth, manifest, schema_toml


def _read_manifest(tar: tarfile.TarFile) -> Manifest:
    payload = _read_member(tar, "manifest.json", max_bytes=_MAX_MANIFEST_BYTES)
    try:
        return Manifest.model_validate_json(payload)
    except ValueError as exc:
        raise ArtifactError(f"manifest is malformed: {exc}") from exc


def _validate(manifest: Manifest) -> None:
    if manifest.version != ARTIFACT_VERSION:
        raise ArtifactError(
            f"unsupported artifact version: {manifest.version!r} "
            f"(this build of doppel reads {ARTIFACT_VERSION!r})"
        )
    if manifest.synthesizer_class != "cart":
        raise ArtifactError(
            f"unknown synthesizer_class {manifest.synthesizer_class!r} "
            "(this build only loads 'cart')"
        )


def _read_member(tar: tarfile.TarFile, name: str, *, max_bytes: int | None = None) -> bytes:
    try:
        member = tar.getmember(name)
    except KeyError as exc:
        raise ArtifactError(f"artifact is missing required member {name!r}") from exc
    if max_bytes is not None and member.size > max_bytes:
        raise ArtifactError(
            f"artifact member {name!r} is too large: {member.size} bytes (limit {max_bytes} bytes)"
        )
    handle = tar.extractfile(member)
    if handle is None:
        raise ArtifactError(f"could not read member {name!r}")
    if max_bytes is None:
        return handle.read()
    payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ArtifactError(f"artifact member {name!r} is too large: more than {max_bytes} bytes")
    return payload


def _read_optional(tar: tarfile.TarFile, name: str, *, max_bytes: int) -> bytes | None:
    try:
        tar.getmember(name)
    except KeyError:
        return None
    return _read_member(tar, name, max_bytes=max_bytes)
