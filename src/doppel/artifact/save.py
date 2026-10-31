"""Save a fitted synthesizer to a `.doppel` artifact: a gzipped tar of JSON and one `.npz`.

Members: `manifest.json`, `model.json` (the synthesizer structure), `arrays.npz` (its
values and trees), `schema.json` (column metadata for human inspection), and, when a
`SchemaToml` is supplied, `schema.toml`, so `doppel sample` can apply declared
constraints without the user passing them again.

Write is atomic: we go through a sibling tempfile and `os.replace` only on success, so a
failed save can never leave a truncated `.doppel` at the destination path.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tomli_w

from doppel import __version__
from doppel.artifact.manifest import (
    ARRAYS_MEMBER,
    MANIFEST_MEMBER,
    MODEL_MEMBER,
    SCHEMA_JSON_MEMBER,
    SCHEMA_TOML_MEMBER,
    Manifest,
)
from doppel.errors import ArtifactError
from doppel.schema.toml import SchemaToml, to_payload
from doppel.synth.cart import CartSynthesizer


def save(
    synth: CartSynthesizer,
    path: Path,
    *,
    training_row_count: int,
    schema_toml: SchemaToml | None = None,
) -> None:
    if not synth.is_fitted:
        raise ValueError("synthesizer must be fitted before saving")
    try:
        state, arrays = synth.to_state()
    except ValueError as exc:
        raise ArtifactError(f"cannot save this synthesizer: {exc}") from exc
    arrays_buffer = io.BytesIO()
    np.savez(arrays_buffer, allow_pickle=False, **arrays)

    manifest = Manifest(
        synthesizer_class="cart",
        doppel_version=__version__,
        table_name=synth.table_name,
        training_row_count=training_row_count,
        training_column_count=len(synth.original_columns),
    )

    schema_payload = {
        "table": synth.table_name,
        "primary_key": synth.primary_key,
        "columns": [asdict(c) for c in synth.original_columns],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".doppel-", suffix=".tmp", dir=path.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            _add_text(tar, MANIFEST_MEMBER, manifest.model_dump_json(indent=2))
            _add_text(tar, MODEL_MEMBER, state.model_dump_json())
            _add_bytes(tar, ARRAYS_MEMBER, arrays_buffer.getvalue())
            _add_text(tar, SCHEMA_JSON_MEMBER, json.dumps(schema_payload, indent=2, default=str))
            if schema_toml is not None:
                _add_text(tar, SCHEMA_TOML_MEMBER, tomli_w.dumps(to_payload(schema_toml)))
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    _add_bytes(tar, name, text.encode("utf-8"))


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))
