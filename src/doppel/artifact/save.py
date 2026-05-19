"""Save a fitted synthesizer to a `.doppel` artifact (gzipped tar with manifest + schema + pickle).

If a `SchemaToml` is supplied, it is embedded as `schema.toml` so `doppel sample` can
honour declared constraints without the user re-passing them. `schema.json` always
captures the inferred-or-merged column metadata for human inspection.

Write is atomic: we go through a sibling tempfile and `os.replace` only on success, so a
failed save can never leave a truncated `.doppel` at the destination path.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path

import tomli_w

from doppel import __version__
from doppel.artifact.manifest import Manifest
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
            _add_text(tar, "manifest.json", manifest.model_dump_json(indent=2))
            _add_text(tar, "schema.json", json.dumps(schema_payload, indent=2, default=str))
            if schema_toml is not None:
                _add_text(tar, "schema.toml", tomli_w.dumps(to_payload(schema_toml)))
            _add_bytes(
                tar,
                "synth.pickle",
                pickle.dumps(synth, protocol=pickle.HIGHEST_PROTOCOL),
            )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    _add_bytes(tar, name, text.encode("utf-8"))


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))
