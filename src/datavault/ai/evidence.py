from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_REDACTIONS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"\b(?:sk|pk|ghp|gho|xox[bap])[-_][A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
    re.compile(r"\b(?:\+?\d[\d\s().-]{8,}\d)\b"),
]
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def clean(text: Any, limit: int = 160) -> str:
    value = _CONTROL.sub("", str(text))
    for pattern in _REDACTIONS:
        value = pattern.sub("[REDACTED]", value)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def numbers_in(text: str) -> set[float]:
    return {float(m) for m in NUMBER.findall(text)}


@dataclass(frozen=True)
class Evidence:
    id: str
    fact: str
    source: str

    @property
    def numbers(self) -> set[float]:
        return numbers_in(self.fact)


@dataclass
class EvidenceSet:
    items: list[Evidence] = field(default_factory=list)
    max_items: int = 90

    def add(self, fact: str, source: str) -> str | None:
        if len(self.items) >= self.max_items:
            return None
        ident = f"E{len(self.items) + 1}"
        self.items.append(Evidence(ident, clean(fact, 300), source))
        return ident

    def by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.items}

    def render(self) -> str:
        return "\n".join(json.dumps({"id": e.id, "fact": e.fact}, ensure_ascii=True) for e in self.items)

    def to_list(self) -> list[dict[str, str]]:
        return [{"id": e.id, "fact": e.fact, "source": e.source} for e in self.items]


def _pct(x: float) -> str:
    return f"{x:.1%}"


def dataset_evidence(diff: dict[str, Any], redact_values: bool = False) -> EvidenceSet:
    ev = EvidenceSet()
    a, b = diff["a"]["id"][:12], diff["b"]["id"][:12]
    ev.add(f"Comparing version {a} to version {b}.", "diff")
    ev.add(
        "Dataset content is identical." if diff["identical_content"] else "Dataset content differs.",
        "diff.identical_content",
    )
    files = diff["files"]
    for item in files["added"]:
        ev.add(f"File added: {clean(item['path'], 80)} ({item['size']} bytes).", "diff.files.added")
    for item in files["removed"]:
        ev.add(f"File removed: {clean(item['path'], 80)} ({item['size']} bytes).", "diff.files.removed")
    for item in files["modified"]:
        ev.add(
            f"File modified: {clean(item['path'], 80)} size {item['size_a']} -> {item['size_b']} bytes.",
            "diff.files.modified",
        )
    ev.add(f"{len(files['unchanged'])} files unchanged.", "diff.files.unchanged")
    for step, change in diff["preprocessing"].items():
        ev.add(f"Preprocessing step {clean(step, 40)} changed {change['a']} -> {change['b']}.", "diff.preprocessing")
    for path, changes in diff["stats"].items():
        p = clean(path, 80)
        src = f"diff.stats[{p}]"
        if "new_file" in changes:
            ev.add(f"{p} is new with {changes['new_file']['rows']} rows.", src)
        if "rows" in changes:
            r = changes["rows"]
            ev.add(f"{p} row count {r['a']} -> {r['b']} (delta {r['delta']}).", src + ".rows")
        if "columns_added" in changes:
            ev.add(f"{p} columns added: {[clean(c, 40) for c in changes['columns_added']]}; "
                   f"removed: {[clean(c, 40) for c in changes['columns_removed']]}.", src + ".columns")
        if "duplicate_rows" in changes:
            d = changes["duplicate_rows"]
            ev.add(f"{p} duplicate rows {d['a']} -> {d['b']}.", src + ".duplicate_rows")
        for col, c in changes.get("columns", {}).items():
            cname = clean(col, 40)
            if "missing_fraction" in c:
                m = c["missing_fraction"]
                ev.add(f"{p} column {cname} missing fraction {_pct(m['a'])} -> {_pct(m['b'])}.", src + f".{cname}.missing")
            if "distinct" in c:
                ev.add(f"{p} column {cname} distinct values {c['distinct']['a']} -> {c['distinct']['b']}.", src + f".{cname}.distinct")
            if "numeric_type_changed" in c:
                ev.add(f"{p} column {cname} numeric parsing changed {c['numeric_type_changed']['a']} -> {c['numeric_type_changed']['b']}.", src + f".{cname}.type")
            for stat, m in c.get("numeric", {}).items():
                if m["a"] != m["b"]:
                    ev.add(f"{p} column {cname} {stat} {m['a']} -> {m['b']}.", src + f".{cname}.{stat}")
            if "categories" in c:
                cat = c["categories"]
                ev.add(f"{p} column {cname} category distribution total variation distance {cat['total_variation']}.", src + f".{cname}.categories")
                if not redact_values:
                    for value, m in cat["counts"].items():
                        ev.add(f"{p} column {cname} value {clean(value, 40)!r} count {m['a']} -> {m['b']}.", src + f".{cname}.categories")
        if "text" in changes:
            for key, m in changes["text"].items():
                ev.add(f"{p} text {key} {m['a']} -> {m['b']}.", src + ".text")
    quality = diff.get("quality")
    if quality:
        for side, label in (("a", a), ("b", b)):
            q = quality[side]
            ev.add(f"Quality checks on {label}: {q['errors']} errors, {q['warnings']} warnings.", f"quality.{side}")
            for issue in q["issues"][:8]:
                ev.add(f"Quality issue on {label} ({issue['severity']}): {clean(issue['message'], 120)} in {clean(issue['path'], 60)}.", f"quality.{side}")
    return ev


def experiment_evidence(
    a: dict[str, Any], b: dict[str, Any], comparison: dict[str, Any], dataset_diff: dict[str, Any] | None
) -> EvidenceSet:
    ev = EvidenceSet()
    ev.add(f"Experiment A {a['id'][:12]} ({a['status']}) on dataset {a['dataset']['commit'][:12]}.", "records.a")
    ev.add(f"Experiment B {b['id'][:12]} ({b['status']}) on dataset {b['dataset']['commit'][:12]}.", "records.b")
    ev.add(
        "Both runs used identical dataset content." if comparison["same_dataset_content"]
        else "The runs used different dataset content.",
        "comparison.same_dataset_content",
    )
    if not comparison["config_changes"]:
        ev.add("No experiment configuration settings differ.", "comparison.config_changes")
    for key, change in comparison["config_changes"].items():
        ev.add(f"Setting {clean(key, 60)} changed {clean(change['a'], 60)} -> {clean(change['b'], 60)}.", "comparison.config_changes")
    for name, m in comparison["metrics"].items():
        sign = "" if m.get("delta", 0) <= 0 else "+"
        ev.add(f"Metric {clean(name, 40)}: {m['a']} -> {m['b']} (delta {sign}{m.get('delta')}).", "comparison.metrics")
    for note in comparison["notes"]:
        ev.add(note, "comparison.notes")
    for key, change in comparison["environment_changes"].items():
        ev.add(f"Environment {clean(key, 40)} differs: {clean(change['a'], 60)} vs {clean(change['b'], 60)}.", "comparison.environment")
    for label, rec in (("A", a), ("B", b)):
        if rec["error"]:
            ev.add(f"Run {label} failed: {clean(rec['error'], 200)}", f"records.{label.lower()}.error")
        for line in rec["logs"][-4:]:
            ev.add(f"Run {label} log: {clean(line, 160)}", f"records.{label.lower()}.logs")
        if rec["metrics"]:
            ds = rec["metrics"]["dataset"]
            ev.add(f"Run {label} split: train {ds['train_size']}, test {ds['test_size']}, vocabulary {ds['vocabulary_size']}.", f"records.{label.lower()}.dataset")
    if dataset_diff:
        sub = dataset_evidence(dataset_diff)
        for item in sub.items[2:30]:
            ev.add("Dataset diff: " + item.fact, item.source)
    return ev


def summary_evidence(rec: dict[str, Any], parent: dict[str, Any] | None, comparison: dict[str, Any] | None) -> EvidenceSet:
    ev = EvidenceSet()
    ev.add(f"Experiment {rec['id'][:12]} status {rec['status']} on dataset {rec['dataset']['commit'][:12]}.", "record")
    for key, value in sorted(rec["config"].items()):
        ev.add(f"Setting {clean(key, 40)} = {clean(json.dumps(value, sort_keys=True), 120)}.", "record.config")
    if rec["metrics"]:
        t = rec["metrics"]["test"]
        ev.add(f"Test accuracy {t['accuracy']}, macro F1 {t['macro_f1']} on {t['n']} test rows.", "record.metrics.test")
        for cls, m in t["per_class"].items():
            ev.add(f"Class {clean(cls, 30)}: precision {m['precision']}, recall {m['recall']}, F1 {m['f1']}, support {m['support']}.", "record.metrics.test")
        ev.add(f"Train accuracy {rec['metrics']['train']['accuracy']}.", "record.metrics.train")
        ev.add(f"One test prediction changes accuracy by {round(1 / t['n'], 4)}.", "derived")
    if rec["error"]:
        ev.add(f"Run failed: {clean(rec['error'], 200)}", "record.error")
    for line in rec["logs"][-4:]:
        ev.add(f"Log: {clean(line, 160)}", "record.logs")
    if comparison and parent:
        ev.add(f"Parent experiment is {parent['id'][:12]}.", "parent")
        for name, m in comparison["metrics"].items():
            ev.add(f"Versus parent, {clean(name, 40)}: {m['a']} -> {m['b']} (delta {m.get('delta')}).", "comparison.metrics")
        for key, change in comparison["config_changes"].items():
            ev.add(f"Versus parent, setting {clean(key, 60)} changed {clean(change['a'], 60)} -> {clean(change['b'], 60)}.", "comparison.config_changes")
    return ev
