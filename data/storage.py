from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(frame: pd.DataFrame, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=True)
    if metadata:
        encoded = {str(key).encode(): json.dumps(value, sort_keys=True, default=str).encode() for key, value in metadata.items()}
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **encoded})
    pq.write_table(table, destination, compression="zstd")
    return destination


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path), engine="pyarrow")
