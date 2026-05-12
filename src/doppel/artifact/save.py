"""Save a fitted synthesizer to a `.doppel` artifact (gzipped tar with manifest + schema + pickle).

If a `SchemaToml` is supplied, it is embedded as `schema.toml` so `doppel sample` can
honour declared constraints without the user re-passing them. `schema.json` always
captures the inferred-or-merged column metadata for human inspection.
"""

from __future__ import annotations

import io
import json
import pickle
import tarfile
from dataclasses import asdict
from pathlib import Path

import tomli_w

from doppel import __version__
from doppel.artifact.manifest import Manifest
from doppel.schema.toml import SchemaToml
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
    with tarfile.open(path, "w:gz") as tar:
        _add_text(tar, "manifest.json", manifest.model_dump_json(indent=2))
        _add_text(tar, "schema.json", json.dumps(schema_payload, indent=2, default=str))
        if schema_toml is not None:
            _add_text(tar, "schema.toml", tomli_w.dumps(_schema_dict(schema_toml)))
        _add_bytes(
            tar,
            "synth.pickle",
            pickle.dumps(synth, protocol=pickle.HIGHEST_PROTOCOL),
        )


def _schema_dict(schema: SchemaToml) -> dict[str, object]:
    out: dict[str, object] = {
        "table": {k: v for k, v in schema.table.model_dump().items() if v is not None},
    }
    if schema.columns:
        out["columns"] = {
            name: {k: v for k, v in spec.model_dump().items() if v is not None}
            for name, spec in schema.columns.items()
        }
    if schema.constraints:
        out["constraints"] = [
            {k: v for k, v in c.model_dump().items() if v is not None} for c in schema.constraints
        ]
    return out


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    _add_bytes(tar, name, text.encode("utf-8"))


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))
