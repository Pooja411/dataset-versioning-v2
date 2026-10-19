from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..errors import InvalidModelOutput
from .evidence import EvidenceSet, numbers_in

FINDINGS, STRINGS = "findings", "strings"
SPECS: dict[str, dict[str, str]] = {
    "dataset_change": {
        "summary": FINDINGS,
        "preprocessing_implications": FINDINGS,
        "training_implications": FINDINGS,
        "quality_concerns": FINDINGS,
        "suggested_checks": STRINGS,
        "uncertainties": STRINGS,
    },
    "experiment_comparison": {
        "changed_settings": FINDINGS,
        "metric_changes": FINDINGS,
        "supported_by_evidence": FINDINGS,
        "hypotheses_to_test": FINDINGS,
        "follow_up_experiments": STRINGS,
        "uncertainties": STRINGS,
    },
    "experiment_summary": {
        "what_was_evaluated": FINDINGS,
        "data_and_configuration": FINDINGS,
        "metric_changes": FINDINGS,
        "supported_by_evidence": FINDINGS,
        "uncertainties": STRINGS,
        "next_steps": STRINGS,
    },
}
CAUSAL = re.compile(r"\b(because|caused|causes|due to|led to|leads to|result of|resulted in|responsible for)\b", re.I)
EVIDENCE_REF = re.compile(r"\bE\d+\b")
MAX_ITEMS = 12
MAX_TEXT = 600


@dataclass
class Item:
    text: str
    kind: str
    evidence: list[str]
    status: str
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Analysis:
    task: str
    sections: dict[str, list[Item]]

    def items(self) -> list[Item]:
        return [i for items in self.sections.values() for i in items]

    def stats(self) -> dict[str, Any]:
        items = [i for i in self.items() if i.kind in ("verified", "hypothesis")]
        counts = {s: sum(i.status == s for i in items) for s in ("grounded", "hypothesis", "unsupported")}
        total = len(items)
        return {**counts, "total": total, "unsupported_rate": round(counts["unsupported"] / total, 4) if total else 0.0}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "sections": {k: [i.to_dict() for i in v] for k, v in self.sections.items()},
            "stats": self.stats(),
        }


def extract_json(text: str) -> dict[str, Any]:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise InvalidModelOutput("response did not contain a JSON object")
    try:
        value = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InvalidModelOutput(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise InvalidModelOutput("response JSON must be an object")
    return value


# percent and fraction forms are accepted so 0.125 can support "12.5%"
def _number_supported(number: float, allowed: set[float]) -> bool:
    for candidate in allowed:
        for scaled in (candidate, candidate * 100, candidate / 100):
            if abs(scaled - number) <= max(1e-6, abs(number) * 1e-3) or abs(round(scaled, 2) - number) < 1e-9:
                return True
    return False


def _check_finding(raw: Any, evidence: EvidenceSet) -> Item:
    if not isinstance(raw, dict):
        raise InvalidModelOutput("finding must be an object")
    text, kind, refs = raw.get("statement"), raw.get("kind"), raw.get("evidence", [])
    if not isinstance(text, str) or not text.strip():
        raise InvalidModelOutput("finding is missing a statement")
    if kind not in ("verified", "hypothesis"):
        raise InvalidModelOutput("finding kind must be 'verified' or 'hypothesis'")
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise InvalidModelOutput("finding evidence must be a list of ids")
    text = text.strip()[:MAX_TEXT]
    known = evidence.by_id()
    problems: list[str] = []
    unknown = [r for r in refs if r not in known]
    if unknown:
        problems.append(f"cites unknown evidence: {unknown}")
    if kind == "verified":
        if not refs:
            problems.append("verified statement cites no evidence")
        allowed = set().union(*(known[r].numbers for r in refs if r in known)) if refs else set()
        stripped = EVIDENCE_REF.sub("", text)
        unsupported = sorted(n for n in numbers_in(stripped) if not _number_supported(n, allowed))
        if unsupported:
            problems.append(f"numbers not present in cited evidence: {unsupported}")
        if CAUSAL.search(text):
            problems.append("causal language in a statement presented as verified")
    status = "unsupported" if problems else ("grounded" if kind == "verified" else "hypothesis")
    return Item(text, kind, refs, status, problems)


def validate(task: str, raw_text: str, evidence: EvidenceSet) -> Analysis:
    spec = SPECS[task]
    data = extract_json(raw_text)
    missing = [k for k in spec if k not in data]
    if missing:
        raise InvalidModelOutput(f"response is missing sections: {missing}")
    sections: dict[str, list[Item]] = {}
    for name, shape in spec.items():
        value = data[name]
        if not isinstance(value, list):
            raise InvalidModelOutput(f"section {name} must be a list")
        items: list[Item] = []
        for raw in value[:MAX_ITEMS]:
            if shape == FINDINGS:
                items.append(_check_finding(raw, evidence))
            else:
                if not isinstance(raw, str) or not raw.strip():
                    raise InvalidModelOutput(f"section {name} must contain non-empty strings")
                items.append(Item(raw.strip()[:MAX_TEXT], "note", [], "note"))
        sections[name] = items
    return Analysis(task, sections)


SHAPES = {
    FINDINGS: '[{"statement": str, "kind": "verified" | "hypothesis", "evidence": ["E1", ...]}]',
    STRINGS: "[str]",
}


def describe(task: str) -> str:
    return json.dumps({k: SHAPES[v] for k, v in SPECS[task].items()}, indent=2).replace('\\"', '"')
