from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..repo import Repository
from . import analysts, testgen
from .provider import LLMRequest, MockProvider, Provider
from .schemas import SPECS, validate

Responder = Callable[[LLMRequest], str]

CSV_BASE = "id,age,label,note\n1,30,a,x\n2,40,b,y\n3,50,a,z\n4,25,b,w\n"


@dataclass
class DatasetCase:
    name: str
    old: str
    new: str
    expected: list[str]


DATASET_CASES = [
    DatasetCase("schema_change", CSV_BASE, "id,age,label\n1,30,a\n2,40,b\n3,50,a\n4,25,b\n", [r"removed: \['note'\]"]),
    DatasetCase("row_count", CSV_BASE, CSV_BASE + "5,31,a,q\n6,41,b,r\n7,51,a,s\n8,26,b,t\n", [r"row count 4 -> 8"]),
    DatasetCase("missing_regression", CSV_BASE, "id,age,label,note\n1,30,a,x\n2,,b,y\n3,50,a,z\n4,25,b,w\n",
                [r"column age missing fraction 0\.0% -> 25\.0%"]),
    DatasetCase("category_shift", CSV_BASE, "id,age,label,note\n1,30,a,x\n2,40,a,y\n3,50,a,z\n4,25,a,w\n",
                [r"column label value 'b' count 2 -> 0"]),
    DatasetCase("no_change", CSV_BASE, CSV_BASE, [r"content is identical"]),
]


def _facts(request: LLMRequest) -> list[dict[str, str]]:
    block = request.user.split("<evidence>")[1].split("</evidence>")[0].strip()
    return [json.loads(line) for line in block.splitlines()]


def _empty(task: str) -> dict[str, list[Any]]:
    return {name: [] for name in SPECS[task]}


def reference_responder(request: LLMRequest) -> str:
    out = _empty("dataset_change")
    for fact in _facts(request):
        text = fact["fact"]
        if re.search(r"row count|missing fraction|columns added|value '|content is identical", text):
            out["summary"].append({"statement": text, "kind": "verified", "evidence": [fact["id"]]})
    out["uncertainties"].append("Only aggregate statistics were available.")
    return json.dumps(out)


def hallucinating_responder(request: LLMRequest) -> str:
    out = _empty("dataset_change")
    facts = _facts(request)
    if facts:
        out["summary"].append({"statement": "Row count rose by 73 percent to 412 rows.", "kind": "verified", "evidence": [facts[0]["id"]]})
        out["summary"].append({"statement": "The change was caused by a new labelling vendor.", "kind": "verified", "evidence": ["E99"]})
    return json.dumps(out)


def score_dataset_case(case: DatasetCase, provider: Provider) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repository.init(Path(tmp))
        (Path(tmp) / "t.csv").write_text(case.old)
        repo.add([Path(tmp) / "t.csv"])
        a = repo.commit("old")
        (Path(tmp) / "t.csv").write_text(case.new)
        repo.add([Path(tmp) / "t.csv"])
        b = repo.commit("new", allow_empty=True)
        try:
            result = analysts.analyze_dataset_change(repo, a.id, b.id, provider)
        except Exception as exc:
            return {"case": case.name, "structured_valid": False, "error": f"{type(exc).__name__}: {exc}"}
        stats = result.analysis.stats()
        known = {e.id for e in result.evidence.items}
        cited = [r for i in result.analysis.items() for r in i.evidence]
        grounded_text = " ".join(
            result.evidence.by_id()[r].fact for i in result.analysis.items() if i.status == "grounded"
            for r in i.evidence if r in known
        )
        covered = sum(bool(re.search(p, grounded_text)) for p in case.expected)
        return {
            "case": case.name,
            "structured_valid": True,
            "evidence_consistency": round(sum(r in known for r in cited) / len(cited), 4) if cited else 1.0,
            "unsupported_rate": stats["unsupported_rate"],
            "key_fact_coverage": round(covered / len(case.expected), 4),
            "latency_s": result.usage["latency_s"],
            "input_tokens": result.usage["input_tokens"],
            "output_tokens": result.usage["output_tokens"],
        }


def run_dataset_evals(provider: Provider | None = None, responder: Responder = reference_responder) -> list[dict[str, Any]]:
    return [score_dataset_case(c, provider or MockProvider(responder)) for c in DATASET_CASES]


MALFORMED_RESPONSES = {
    "empty_object": "{}",
    "prose": "The dataset changed a lot.",
    "wrong_section_type": json.dumps({**_empty("dataset_change"), "summary": "text"}),
    "bad_kind": json.dumps({**_empty("dataset_change"), "summary": [{"statement": "x", "kind": "fact", "evidence": []}]}),
    "contradictory_metadata": json.dumps({**_empty("dataset_change"), "summary": [{"statement": "x", "kind": "verified", "evidence": "E1"}]}),
}


def run_malformed_evals() -> dict[str, bool]:
    from ..errors import InvalidModelOutput
    from .evidence import EvidenceSet

    ev = EvidenceSet()
    ev.add("Row count 1 -> 2.", "t")
    rejected = {}
    for name, raw in MALFORMED_RESPONSES.items():
        try:
            validate("dataset_change", raw, ev)
            rejected[name] = False
        except InvalidModelOutput:
            rejected[name] = True
    return rejected


CLAMP_GOOD = '''def clamp(value, low, high):
    """Return value limited to the inclusive range [low, high]. Raises ValueError if low > high."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(value, high))
'''
CLAMP_BUGGY = CLAMP_GOOD.replace("max(low, min(value, high))", "max(low, min(value, high - 1))")
CANNED_TESTS = '''import unittest
from mathutil import clamp


class ClampTests(unittest.TestCase):
    def test_inside(self):
        self.assertEqual(clamp(5, 0, 10), 5)

    def test_below(self):
        self.assertEqual(clamp(-3, 0, 10), 0)

    def test_above_upper_bound_is_inclusive(self):
        self.assertEqual(clamp(99, 0, 10), 10)

    def test_invalid_range(self):
        with self.assertRaises(ValueError):
            clamp(1, 5, 2)
'''


def run_testgen_eval() -> dict[str, Any]:
    canned = json.dumps({"tests": [{"name": "clamp", "description": "bounds"}], "code": CANNED_TESTS})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = Repository.init(root)
        module = root / "mathutil.py"
        module.write_text(CLAMP_GOOD)
        proposal = testgen.propose_tests(repo, MockProvider([canned]), "mathutil.py::clamp")
        result: dict[str, Any] = {"syntactically_valid": not proposal["review_problems"], "sandbox": testgen.sandbox_available()}
        if not proposal["review_problems"] and testgen.sandbox_available():
            sha = proposal["code_sha256"][:12]
            result["passes_on_correct_code"] = testgen.run_proposal(repo, proposal["path"], sha)["passed"]
            module.write_text(CLAMP_BUGGY)
            result["detects_known_defect"] = not testgen.run_proposal(repo, proposal["path"], sha)["passed"]
        return result


def main() -> None:
    report = {
        "dataset_reference_responder": run_dataset_evals(),
        "dataset_hallucinating_responder": run_dataset_evals(responder=hallucinating_responder),
        "malformed_output_rejection": run_malformed_evals(),
        "test_generation_fixture": run_testgen_eval(),
        "note": "Responders are deterministic stand-ins that exercise the checks; they say nothing about any real model.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
