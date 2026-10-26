from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .errors import ValidationFailed

csv.field_size_limit(16 * 1024 * 1024)

TABULAR = {".csv": ",", ".tsv": "\t"}
LABELED = {".txt"}


def detect_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in TABULAR:
        return suffix[1:]
    if suffix in LABELED:
        return "labeled-text"
    return "binary"


def dataset_format(paths: list[str]) -> str:
    kinds = {detect_format(p) for p in paths}
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed" if kinds else "empty"


def iter_table(data: bytes, path: str) -> Iterator[dict[str, str]]:
    delimiter = TABULAR[Path(path).suffix.lower()]
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    if reader.fieldnames is None:
        raise ValidationFailed(f"{path} has no header row")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValidationFailed(f"{path} has duplicate column names")
    for row in reader:
        yield {k: (v if v is not None else "") for k, v in row.items() if k is not None}


def table_columns(data: bytes, path: str) -> list[str]:
    delimiter = TABULAR[Path(path).suffix.lower()]
    first = io.StringIO(data.decode("utf-8-sig"), newline="")
    header = next(csv.reader(first, delimiter=delimiter), [])
    return header


def infer_schema(data: bytes, path: str) -> dict[str, Any]:
    kind = detect_format(path)
    if kind in ("csv", "tsv"):
        return {"columns": table_columns(data, path)}
    return {}
