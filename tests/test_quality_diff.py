import unittest

from helpers import RepoTestCase

from datavault.diff import diff_commits
from datavault.errors import ValidationFailed
from datavault.quality import run_checks, validate_config

CSV_A = "id,age,label,note\n1,30,a,x\n2,40,b,y\n3,50,a,z\n4,25,b,w\n"
CSV_B = "id,age,label,note\n1,30,a,x\n2,,b,\n3,500,a,\n4,25,c,w\n4,25,c,w\n"


def run(rules, data=CSV_B, path="t.csv"):
    return run_checks({"rules": rules}, {path: data.encode()})


class QualityTests(unittest.TestCase):
    def test_required_columns(self):
        r = run([{"check": "required_columns", "columns": ["id", "missing_col"]}])
        self.assertEqual(r.error_count, 1)

    def test_missing_threshold_and_severity(self):
        r = run([{"check": "max_missing", "column": "note", "fraction": 0.2, "severity": "warning"}])
        self.assertEqual((r.error_count, r.warning_count), (0, 1))

    def test_wildcard_column_missing_check(self):
        r = run([{"check": "max_missing", "column": "*", "fraction": 0.0}])
        self.assertEqual({i.column for i in r.issues}, {"age", "note"})

    def test_types_ranges_allowed_values(self):
        r = run([
            {"check": "range", "column": "age", "min": 0, "max": 120},
            {"check": "allowed_values", "column": "label", "values": ["a", "b"]},
            {"check": "column_type", "column": "id", "type": "int"},
        ])
        self.assertEqual({i.rule for i in r.issues}, {"range", "allowed_values"})

    def test_duplicates_and_unique_keys(self):
        r = run([{"check": "no_duplicate_rows"}, {"check": "unique_key", "columns": ["id"]}])
        self.assertEqual(r.error_count, 2)
        self.assertEqual(r.issues[0].rows, [5])

    def test_row_bounds(self):
        self.assertEqual(run([{"check": "min_rows", "n": 10}]).error_count, 1)
        self.assertEqual(run([{"check": "max_rows", "n": 2}]).error_count, 1)
        self.assertEqual(run([{"check": "min_rows", "n": 2}]).error_count, 0)

    def test_reports_never_echo_cell_values(self):
        r = run([{"check": "allowed_values", "column": "label", "values": ["a"]}])
        self.assertTrue(r.issues)
        for issue in r.issues:
            self.assertNotIn(" c ", f" {issue.message} ")

    def test_invalid_rule_config_rejected(self):
        for bad in [{}, {"rules": [{"check": "nonsense"}]}, {"rules": [{"check": "min_rows", "n": 1, "severity": "fatal"}]}]:
            with self.assertRaises(ValidationFailed):
                validate_config(bad)

    def test_path_patterns_select_files(self):
        r = run_checks(
            {"rules": [{"check": "min_rows", "n": 99, "path": "a/*.csv"}]},
            {"a/x.csv": b"c\n1\n", "b/y.csv": b"c\n1\n"},
        )
        self.assertEqual(r.rules_run, 1)

    def test_labeled_text_is_a_table_with_text_and_label(self):
        r = run_checks({"rules": [{"check": "allowed_values", "column": "label", "values": ["p"]}]}, {"d.txt": b"a|p\nb|n\n"})
        self.assertEqual(r.error_count, 1)


class DiffTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.a = self.commit_file("t.csv", CSV_A, "a")
        self.b = self.commit_file("t.csv", CSV_B, "b")
        self.diff = diff_commits(self.repo, self.a.id, self.b.id)

    def test_file_level_changes(self):
        self.assertEqual([m["path"] for m in self.diff["files"]["modified"]], ["t.csv"])
        self.assertFalse(self.diff["identical_content"])

    def test_row_count_and_duplicates(self):
        stats = self.diff["stats"]["t.csv"]
        self.assertEqual(stats["rows"], {"a": 4, "b": 5, "delta": 1})
        self.assertEqual(stats["duplicate_rows"]["b"], 1)

    def test_missing_value_regression(self):
        cols = self.diff["stats"]["t.csv"]["columns"]
        self.assertEqual(cols["age"]["missing_fraction"]["a"], 0.0)
        self.assertEqual(cols["age"]["missing_fraction"]["b"], 0.2)
        self.assertEqual(cols["note"]["missing_fraction"]["b"], 0.4)

    def test_numeric_and_category_shift(self):
        cols = self.diff["stats"]["t.csv"]["columns"]
        self.assertEqual(cols["age"]["numeric"]["max"], {"a": 50.0, "b": 500.0, "delta": 450.0})
        cat = cols["label"]["categories"]
        self.assertEqual(cat["new_values"], ["c"])
        self.assertGreater(cat["total_variation"], 0)

    def test_schema_change(self):
        c = self.commit_file("t.csv", "id,age,label\n1,30,a\n", "drop note")
        d = diff_commits(self.repo, self.b.id, c.id)
        self.assertEqual(d["stats"]["t.csv"]["columns_removed"], ["note"])

    def test_identical_content_diff(self):
        self.repo.create_tag("same", "HEAD")
        d = diff_commits(self.repo, "HEAD", "same")
        self.assertTrue(d["identical_content"])
        self.assertEqual(d["stats"], {})

    def test_quality_reports_for_both_sides(self):
        rules = {"rules": [{"check": "max_missing", "column": "age", "fraction": 0.1}]}
        self.write("t.csv", CSV_B)
        self.repo.add([self.root / "t.csv"])
        self.repo.create_branch("keep", "HEAD")
        c = self.commit_file("t.csv", CSV_B + "9,1,a,q\n", "c", quality_config=rules, allow_failures=True)
        d = diff_commits(self.repo, self.a.id, c.id)
        self.assertEqual(d["quality"]["a"]["errors"], 0)
        self.assertEqual(d["quality"]["b"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
