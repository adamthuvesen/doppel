"""Lock the documented `doppel.pipeline` programmatic API (README "Python API")."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from doppel.pipeline import SingleTableGenerateConfig, generate_single_table
from doppel.sources.spec import FilePath
from doppel.text_policy import TextPolicy


def test_generate_single_table_matches_readme_example(tmp_path: Path) -> None:
    source = tmp_path / "sales.parquet"
    pl.DataFrame(
        {
            "region": ["north", "south", "east", "west"] * 25,
            "amount": [float(i) for i in range(100)],
        }
    ).write_parquet(source)

    result = generate_single_table(
        SingleTableGenerateConfig(
            source_spec=FilePath(path=source),
            rows=1000,
            seed=1,
            text_policy=TextPolicy.SAMPLE,
        ),
        sample_fit=lambda df, n: df if n is None else df.head(n),
    )

    assert result.out_df.height == 1000
    assert set(result.out_df.columns) == {"region", "amount"}
