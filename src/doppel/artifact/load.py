"""Load a `.doppel` artifact: validate the container, then rebuild the synthesizer from data.

The artifact holds JSON and one `.npz` of plain arrays. Strict Pydantic models parse the
JSON, and `numpy.load(..., allow_pickle=False)` reads the arrays, so loading never
deserialises code. A member that is unknown, duplicated, too large, or a pickle from the
old format is refused before anything is parsed.

`inspect_artifact()` reads only the manifest and the embedded schema. `doppel artifact info`
uses it to show metadata without rebuilding the model.
"""

from __future__ import annotations

import io
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile
from pydantic import ValidationError

from doppel.artifact.manifest import (
    ARRAYS_MEMBER,
    ARTIFACT_VERSION,
    MANIFEST_MEMBER,
    MODEL_MEMBER,
    SCHEMA_JSON_MEMBER,
    SCHEMA_TOML_MEMBER,
    Manifest,
)
from doppel.schema.toml import SchemaToml
from doppel.synth.cart import CartSynthesizer
from doppel.synth.cart_state import StoredCart

_MIB = 1024 * 1024
_MEMBER_LIMITS: dict[str, int] = {
    MANIFEST_MEMBER: _MIB,
    MODEL_MEMBER: 64 * _MIB,
    ARRAYS_MEMBER: 512 * _MIB,
    SCHEMA_JSON_MEMBER: 5 * _MIB,
    SCHEMA_TOML_MEMBER: 5 * _MIB,
}
_REQUIRED_MEMBERS = (MANIFEST_MEMBER, MODEL_MEMBER, ARRAYS_MEMBER)
_MAX_TOTAL_BYTES = 512 * _MIB
_MAX_ARRAYS = 65_536
_OLD_PICKLE_MEMBER = "synth.pickle"
_NPZ_ERRORS = (ValueError, OSError, EOFError, zipfile.BadZipFile)


class ArtifactError(ValueError):
    """Raised when a `.doppel` artifact is malformed, mis-versioned, or untrusted."""


@dataclass(frozen=True)
class ArtifactInfo:
    """Metadata-only view of a `.doppel` artifact. Does not rebuild the synthesizer."""

    path: Path
    file_size: int
    manifest: Manifest
    schema_toml: SchemaToml | None
    model_size: int
    arrays_size: int


def inspect_artifact(path: Path) -> ArtifactInfo:
    """Read the manifest, the embedded schema, and the member sizes. The model stays unread."""
    payloads, sizes = _read_members(path, read={MANIFEST_MEMBER, SCHEMA_TOML_MEMBER})
    return ArtifactInfo(
        path=path,
        file_size=path.stat().st_size,
        manifest=_parse_manifest(payloads[MANIFEST_MEMBER]),
        schema_toml=_parse_schema(payloads.get(SCHEMA_TOML_MEMBER)),
        model_size=sizes[MODEL_MEMBER],
        arrays_size=sizes[ARRAYS_MEMBER],
    )


def load(path: Path) -> tuple[CartSynthesizer, Manifest, SchemaToml | None]:
    payloads, _ = _read_members(path, read=set(_MEMBER_LIMITS))
    manifest = _parse_manifest(payloads[MANIFEST_MEMBER])
    try:
        state = StoredCart.model_validate_json(payloads[MODEL_MEMBER])
    except ValidationError as exc:
        raise ArtifactError(f"{path}: {MODEL_MEMBER} is malformed: {exc}") from exc
    arrays = _read_arrays(payloads[ARRAYS_MEMBER])
    try:
        synth = CartSynthesizer.from_state(state, arrays)
    except ValueError as exc:
        raise ArtifactError(f"{path}: the stored synthesizer is inconsistent: {exc}") from exc
    return synth, manifest, _parse_schema(payloads.get(SCHEMA_TOML_MEMBER))


def _read_members(path: Path, *, read: set[str]) -> tuple[dict[str, bytes], dict[str, int]]:
    payloads: dict[str, bytes] = {}
    sizes: dict[str, int] = {}
    try:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                _check_member(member, sizes)
                sizes[member.name] = member.size
                if member.name in read:
                    payloads[member.name] = _read_member(tar, member)
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ArtifactError(f"{path} is not a valid doppel artifact: {exc}") from exc
    for name in _REQUIRED_MEMBERS:
        if name not in sizes:
            raise ArtifactError(f"artifact is missing required member {name!r}")
    return payloads, sizes


def _check_member(member: tarfile.TarInfo, seen: dict[str, int]) -> None:
    name = member.name
    if name == _OLD_PICKLE_MEMBER:
        raise ArtifactError(
            "artifact holds a pickled synthesizer from an older doppel. doppel no longer "
            "loads pickles; refit the model with `doppel fit`."
        )
    limit = _MEMBER_LIMITS.get(name)
    if limit is None:
        raise ArtifactError(f"artifact has unexpected member {name!r}")
    if name in seen:
        raise ArtifactError(f"artifact has member {name!r} more than once")
    if not member.isfile():
        raise ArtifactError(f"artifact member {name!r} is not a regular file")
    if member.size > limit:
        raise ArtifactError(
            f"artifact member {name!r} is too large: {member.size} bytes (limit {limit} bytes)"
        )
    if sum(seen.values()) + member.size > _MAX_TOTAL_BYTES:
        raise ArtifactError(
            f"artifact is too large: more than {_MAX_TOTAL_BYTES} bytes across its members"
        )


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise ArtifactError(f"could not read member {member.name!r}")
    limit = _MEMBER_LIMITS[member.name]
    payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise ArtifactError(
            f"artifact member {member.name!r} is too large: more than {limit} bytes"
        )
    return payload


def _parse_manifest(payload: bytes) -> Manifest:
    try:
        manifest = Manifest.model_validate_json(payload)
    except ValidationError as exc:
        raise ArtifactError(f"manifest is malformed: {exc}") from exc
    if manifest.version != ARTIFACT_VERSION:
        raise ArtifactError(
            f"unsupported artifact version: {manifest.version!r} "
            f"(this build of doppel reads {ARTIFACT_VERSION!r}); refit the model with `doppel fit`"
        )
    if manifest.synthesizer_class != "cart":
        raise ArtifactError(
            f"unknown synthesizer_class {manifest.synthesizer_class!r} "
            "(this build only loads 'cart')"
        )
    return manifest


def _parse_schema(payload: bytes | None) -> SchemaToml | None:
    if payload is None:
        return None
    try:
        return SchemaToml.model_validate(tomllib.loads(payload.decode("utf-8")))
    except ValueError as exc:
        raise ArtifactError(f"{SCHEMA_TOML_MEMBER} is malformed: {exc}") from exc


def _read_arrays(payload: bytes) -> dict[str, np.ndarray]:
    try:
        npz = np.load(io.BytesIO(payload), allow_pickle=False)
    except _NPZ_ERRORS as exc:
        raise ArtifactError(f"{ARRAYS_MEMBER} is not a plain npz archive: {exc}") from exc
    if not isinstance(npz, NpzFile):
        raise ArtifactError(f"{ARRAYS_MEMBER} is not an npz archive")
    with npz:
        if npz.zip is None:
            raise ArtifactError(f"{ARRAYS_MEMBER} is closed")
        entries = npz.zip.infolist()
        if len(entries) > _MAX_ARRAYS:
            raise ArtifactError(f"{ARRAYS_MEMBER} holds more than {_MAX_ARRAYS} arrays")
        if sum(entry.file_size for entry in entries) > _MEMBER_LIMITS[ARRAYS_MEMBER]:
            raise ArtifactError(f"{ARRAYS_MEMBER} expands past its size limit")
        try:
            return {name: npz[name] for name in npz.files}
        except _NPZ_ERRORS as exc:
            raise ArtifactError(
                f"{ARRAYS_MEMBER} holds an array doppel cannot read: {exc}"
            ) from exc
