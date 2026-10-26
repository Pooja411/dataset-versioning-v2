import json
import os
import unittest
from unittest import mock

from helpers import RepoTestCase

from datavault.ai import evals, testgen
from datavault.ai.provider import MockProvider
from datavault.errors import InvalidModelOutput, UnsafePath, ValidationFailed

GOOD_TESTS = '''import unittest
from calc import add


class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
'''


def reply(code, tests=None):
    return json.dumps({"tests": tests or [{"name": "t", "description": "d"}], "code": code})


class ReviewTests(unittest.TestCase):
    def reject(self, code):
        self.assertTrue(testgen.review_code(code, "calc"), code)

    def test_accepts_plain_unittest(self):
        self.assertEqual(testgen.review_code(GOOD_TESTS, "calc"), [])

    def test_rejects_dangerous_imports_and_calls(self):
        self.reject("import os\nclass T:\n    def test_a(self): os.system('x')")
        self.reject("import subprocess\ndef test_a(): pass")
        self.reject("from calc import add\ndef test_a():\n    eval('1')")
        self.reject("from calc import add\ndef test_a():\n    open('/etc/passwd')")
        self.reject("from calc import add\ndef test_a():\n    __import__('os')")
        self.reject("from calc import add\ndef test_a():\n    add.__globals__")
        self.reject("import socket\ndef test_a(): pass")
        self.reject("from . import x\ndef test_a(): pass")

    def test_rejects_syntax_errors_and_empty_suites(self):
        self.reject("def broken(:")
        self.reject("x = 1")

    def test_rejects_oversized_code(self):
        self.reject("# " + "x" * 13000)


class ProposalTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.module = self.write("calc.py", 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n')

    def propose(self, code=GOOD_TESTS, target="calc.py::add", **kw):
        return testgen.propose_tests(self.repo, MockProvider([reply(code)]), target, **kw)

    def test_proposal_is_written_under_datavault_never_to_tests(self):
        p = self.propose()
        self.assertTrue(p["path"].startswith(".datavault/ai/proposed_tests/"))
        self.assertEqual(p["status"], "pending_review")
        self.assertFalse((self.root / "tests").exists())

    def test_rejected_proposal_is_marked(self):
        p = self.propose("import os\nclass T:\n    def test_a(self): pass")
        self.assertEqual(p["status"], "rejected")

    def test_only_target_function_and_requirements_are_sent(self):
        self.write("calc.py", 'SECRET_CONSTANT = "do-not-send"\n\ndef add(a, b):\n    return a + b\n')
        self.write("reqs.md", "add must be commutative")
        provider = MockProvider([reply(GOOD_TESTS)])
        testgen.propose_tests(self.repo, provider, "calc.py::add", requirements_file="reqs.md", defects=["overflow"])
        sent = provider.requests[0].user
        self.assertIn("def add", sent)
        self.assertNotIn("do-not-send", sent)
        self.assertIn("commutative", sent)
        self.assertIn("overflow", sent)

    def test_unsafe_targets_refused(self):
        self.write(".env", "KEY=1")
        self.write("secrets.py", "def f(): pass")
        for spec in ["../x.py::f", "/etc/passwd::f", "secrets.py::f", "calc.py", "calc.py::missing", "calc.txt::f"]:
            with self.assertRaises((UnsafePath, ValidationFailed), msg=spec):
                testgen.resolve_target(self.root, spec)

    def test_requirements_path_cannot_escape(self):
        with self.assertRaises(UnsafePath):
            self.propose(requirements_file="../../etc/passwd")

    def test_malformed_response_rejected(self):
        with self.assertRaises(InvalidModelOutput):
            testgen.propose_tests(self.repo, MockProvider(["no json"]), "calc.py::add")
        with self.assertRaises(InvalidModelOutput):
            testgen.propose_tests(self.repo, MockProvider([json.dumps({"tests": []})]), "calc.py::add")


@unittest.skipUnless(testgen.sandbox_available(), "POSIX only")
class SandboxTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.write("calc.py", "def add(a, b):\n    return a + b\n")

    def proposal(self, code):
        return testgen.propose_tests(self.repo, MockProvider([reply(code)]), "calc.py::add")

    def test_requires_matching_approval(self):
        p = self.proposal(GOOD_TESTS)
        with self.assertRaises(ValidationFailed):
            testgen.run_proposal(self.repo, p["path"], "deadbeefdeadbeef")
        with self.assertRaises(ValidationFailed):
            testgen.run_proposal(self.repo, p["path"], "")

    def test_edited_file_needs_fresh_approval(self):
        p = self.proposal(GOOD_TESTS)
        path = self.root / p["path"]
        path.write_text(GOOD_TESTS + "\n# edited\n")
        with self.assertRaises(ValidationFailed):
            testgen.run_proposal(self.repo, p["path"], p["code_sha256"][:12])

    def test_edit_that_introduces_unsafe_code_is_blocked_even_when_approved(self):
        p = self.proposal(GOOD_TESTS)
        path = self.root / p["path"]
        path.write_text("import os\n" + GOOD_TESTS)
        from datavault.hashing import sha256_bytes
        with self.assertRaises(ValidationFailed):
            testgen.run_proposal(self.repo, p["path"], sha256_bytes(path.read_bytes())[:12])

    def test_only_proposed_tests_directory_is_executable(self):
        outside = self.write("evil.py", GOOD_TESTS)
        from datavault.hashing import sha256_bytes
        with self.assertRaises(UnsafePath):
            testgen.run_proposal(self.repo, "evil.py", sha256_bytes(outside.read_bytes())[:12])

    def test_passing_and_failing_suites(self):
        p = self.proposal(GOOD_TESTS)
        ok = testgen.run_proposal(self.repo, p["path"], p["code_sha256"][:12])
        self.assertTrue(ok["passed"], ok["output"])
        self.assertEqual(ok["tests_run"], 1)
        bad = self.proposal(GOOD_TESTS.replace("3)", "4)"))
        res = testgen.run_proposal(self.repo, bad["path"], bad["code_sha256"][:12])
        self.assertFalse(res["passed"])
        self.assertEqual(res["failures"], 1)

    def test_runaway_code_is_killed(self):
        code = GOOD_TESTS.replace("self.assertEqual(add(1, 2), 3)", "while True:\n            pass")
        p = self.proposal(code)
        limits = {**testgen.LIMITS, "wall_seconds": 4, "cpu_seconds": 2}
        with mock.patch.object(testgen, "LIMITS", limits):
            res = testgen.run_proposal(self.repo, p["path"], p["code_sha256"][:12])
        self.assertFalse(res["passed"])

    def test_child_does_not_inherit_secrets(self):
        os.environ["DATAVAULT_TEST_SECRET"] = "leak-me"
        self.addCleanup(os.environ.pop, "DATAVAULT_TEST_SECRET", None)
        calls = {}
        real = testgen.subprocess.Popen

        def spy(cmd, **kw):
            if "env" in kw:
                calls["env"] = kw["env"]
            return real(cmd, **kw)

        p = self.proposal(GOOD_TESTS)
        with mock.patch.object(testgen.subprocess, "Popen", spy):
            testgen.run_proposal(self.repo, p["path"], p["code_sha256"][:12])
        self.assertNotIn("DATAVAULT_TEST_SECRET", calls["env"])
        self.assertNotIn("ANTHROPIC_API_KEY", calls["env"])


class EvalHarnessTests(unittest.TestCase):
    def test_reference_responder_scores_perfectly(self):
        for row in evals.run_dataset_evals():
            self.assertTrue(row["structured_valid"], row)
            self.assertEqual(row["unsupported_rate"], 0.0, row)
            self.assertEqual(row["evidence_consistency"], 1.0, row)
            self.assertEqual(row["key_fact_coverage"], 1.0, row)

    def test_hallucinating_responder_is_caught(self):
        for row in evals.run_dataset_evals(responder=evals.hallucinating_responder):
            self.assertEqual(row["unsupported_rate"], 1.0, row)
            self.assertEqual(row["key_fact_coverage"], 0.0, row)

    def test_malformed_outputs_are_all_rejected(self):
        self.assertTrue(all(evals.run_malformed_evals().values()))

    @unittest.skipUnless(testgen.sandbox_available(), "POSIX only")
    def test_generated_tests_detect_a_seeded_defect(self):
        result = evals.run_testgen_eval()
        self.assertTrue(result["syntactically_valid"])
        self.assertTrue(result["passes_on_correct_code"])
        self.assertTrue(result["detects_known_defect"])


if __name__ == "__main__":
    unittest.main()
