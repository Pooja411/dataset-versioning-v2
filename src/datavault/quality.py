from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

from .datasets import detect_format
from .errors import ValidationFailed
from .profile import MAX_ANALYSIS_BYTES, _number, is_missing, load_table

SEVERITIES = ("error", "warning")
TYPE_CHECKS = {
    "int": lambda v: re.fullmatch(r"[+-]?\d+", v.strip()) is not None,
    "float": lambda v: _number(v) is not None,
    "bool": lambda v: v.strip().lower() in ("true", "false"),
    "str": lambda v: True,
}
CHECKS = {
    "required_columns", "column_type", "max_missing", "no_duplicate_rows",
    "unique_key", "allowed_values", "range", "min_rows", "max_rows",
}


@dataclass
class Issue:
    rule: str
    severity: str
    path: str
    message: str
    column: str | None = None
    rows: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule, "severity": self.severity, "path": self.path,
            "column": self.column, "message": self.message, "rows": self.rows,
        }


@dataclass
class QualityReport:
    issues: list[Issue] = field(default_factory=list)
    rules_run: int = 0

    @property
    def error_count(self) -> int:
        return sum(i.severity == "error" for i in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(i.severity == "warning" for i in self.issues)

    def render_failures(self) -> list[str]:
        return [f"  [{i.severity}] {i.path}: {i.message}" for i in self.issues]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules_run": self.rules_run,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }


def validate_config(config: Any) -> list[dict[str, Any]]:
    if not isinstance(config, dict) or not isinstance(config.get("rules"), list):
        raise ValidationFailed("quality config must be an object with a 'rules' list")
    for index, rule in enumerate(config["rules"]):
        if not isinstance(rule, dict) or rule.get("check") not in CHECKS:
            raise ValidationFailed(f"rule {index}: 'check' must be one of {sorted(CHECKS)}")
        if rule.get("severity", "error") not in SEVERITIES:
            raise ValidationFailed(f"rule {index}: severity must be error or warning")
    return config["rules"]


def _check(rule: dict[str, Any], path: str, columns: list[str], rows: list[dict[str, str]]) -> list[Issue]:
    kind, severity = rule["check"], rule.get("severity", "error")
    column = rule.get("column")

    def issue(message: str, col: str | None = None, hits: list[int] | None = None) -> Issue:
        return Issue(kind, severity, path, message, col, (hits or [])[:5])

    if kind == "required_columns":
        absent = [c for c in rule.get("columns", []) if c not in columns]
        return [issue(f"missing required columns: {absent}")] if absent else []
    if kind == "min_rows":
        return [issue(f"{len(rows)} rows, expected at least {rule['n']}")] if len(rows) < rule["n"] else []
    if kind == "max_rows":
        return [issue(f"{len(rows)} rows, expected at most {rule['n']}")] if len(rows) > rule["n"] else []
    if kind == "no_duplicate_rows":
        seen: set[tuple[str, ...]] = set()
        dupes = []
        for i, row in enumerate(rows, start=1):
            key = tuple(row.get(c, "") for c in columns)
            if key in seen:
                dupes.append(i)
            seen.add(key)
        return [issue(f"{len(dupes)} duplicate rows", hits=dupes)] if dupes else []
    if kind == "unique_key":
        keys = rule.get("columns", [])
        if any(k not in columns for k in keys):
            return [issue(f"key columns not present: {keys}")]
        seen_keys: set[tuple[str, ...]] = set()
        dupes = []
        for i, row in enumerate(rows, start=1):
            key = tuple(row[k] for k in keys)
            if key in seen_keys:
                dupes.append(i)
            seen_keys.add(key)
        return [issue(f"{len(dupes)} rows repeat key {keys}", hits=dupes)] if dupes else []

    targets = columns if column == "*" else [column]
    out: list[Issue] = []
    for col in targets:
        if col not in columns:
            out.append(issue(f"column {col!r} not present", col))
            continue
        values = [r.get(col, "") for r in rows]
        if kind == "max_missing":
            frac = sum(map(is_missing, values)) / len(values) if values else 0.0
            if frac > rule["fraction"]:
                out.append(issue(f"{frac:.1%} missing exceeds {rule['fraction']:.1%}", col))
        elif kind == "column_type":
            test = TYPE_CHECKS[rule["type"]]
            bad = [i for i, v in enumerate(values, 1) if not is_missing(v) and not test(v)]
            if bad:
                out.append(issue(f"{len(bad)} values are not {rule['type']}", col, bad))
        elif kind == "allowed_values":
            allowed = set(rule["values"])
            bad = [i for i, v in enumerate(values, 1) if not is_missing(v) and v not in allowed]
            if bad:
                out.append(issue(f"{len(bad)} values outside the allowed set", col, bad))
        elif kind == "range":
            bad = []
            for i, v in enumerate(values, 1):
                number = _number(v) if not is_missing(v) else None
                if number is None:
                    continue
                if ("min" in rule and number < rule["min"]) or ("max" in rule and number > rule["max"]):
                    bad.append(i)
            if bad:
                out.append(issue(f"{len(bad)} values outside [{rule.get('min')}, {rule.get('max')}]", col, bad))
    return out


def run_checks(config: dict[str, Any], files: dict[str, bytes]) -> QualityReport:
    rules = validate_config(config)
    report = QualityReport()
    cache: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for rule in rules:
        pattern = rule.get("path", "*")
        for path, data in sorted(files.items()):
            if not fnmatch.fnmatch(path, pattern) or detect_format(path) == "binary":
                continue
            if len(data) > MAX_ANALYSIS_BYTES:
                report.issues.append(Issue(rule["check"], "warning", path, "skipped: file too large"))
                continue
            if path not in cache:
                cache[path] = load_table(path, data)
            columns, rows = cache[path]
            report.rules_run += 1
            report.issues.extend(_check(rule, path, columns, rows))
    return report
