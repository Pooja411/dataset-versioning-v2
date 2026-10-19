from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .datasets import detect_format, iter_table
from .errors import ValidationFailed
from .ml.preprocessing import split_label
from .ml.text_metrics import text_stats

MAX_ANALYSIS_BYTES = 64 * 1024 * 1024
CATEGORY_LIMIT = 20


def is_missing(value: str) -> bool:
    return value.strip() == ""


def load_table(path: str, data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    kind = detect_format(path)
    if kind in ("csv", "tsv"):
        rows = list(iter_table(data, path))
        columns = list(rows[0]) if rows else []
        if not rows:
            header = data.decode("utf-8-sig").splitlines()[:1]
            sep = "," if kind == "csv" else "\t"
            columns = header[0].split(sep) if header else []
        return columns, rows
    if kind == "labeled-text":
        lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
        rows = []
        for line in lines:
            text, label = split_label(line)
            rows.append({"text": text, "label": label or ""})
        return ["text", "label"], rows
    raise ValidationFailed(f"{path}: unsupported format for tabular analysis")


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def profile_column(values: list[str]) -> dict[str, Any]:
    present = [v for v in values if not is_missing(v)]
    out: dict[str, Any] = {
        "count": len(values),
        "missing": len(values) - len(present),
        "distinct": len(set(present)),
    }
    numbers = [_number(v) for v in present]
    if present and all(n is not None for n in numbers):
        nums = [n for n in numbers if n is not None]
        out["numeric"] = {
            "min": min(nums),
            "max": max(nums),
            "mean": round(sum(nums) / len(nums), 6),
        }
    elif 0 < out["distinct"] <= CATEGORY_LIMIT:
        counts = Counter(present)
        out["categories"] = {k: counts[k] for k in sorted(counts)}
    return out


def profile_file(path: str, data: bytes) -> dict[str, Any]:
    if len(data) > MAX_ANALYSIS_BYTES:
        return {"skipped": f"larger than {MAX_ANALYSIS_BYTES} bytes"}
    kind = detect_format(path)
    if kind == "binary":
        return {"format": kind, "bytes": len(data)}
    columns, rows = load_table(path, data)
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = tuple(row.get(c, "") for c in columns)
        duplicates += key in seen
        seen.add(key)
    profile: dict[str, Any] = {
        "format": kind,
        "rows": len(rows),
        "columns": columns,
        "duplicate_rows": duplicates,
        "column_stats": {c: profile_column([r.get(c, "") for r in rows]) for c in columns},
    }
    if kind == "labeled-text":
        profile["text"] = text_stats([r["text"] for r in rows])
    return profile


def total_variation(a: dict[str, int], b: dict[str, int]) -> float:
    ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
    keys = set(a) | set(b)
    return round(0.5 * sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb) for k in keys), 6)
