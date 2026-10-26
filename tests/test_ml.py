import json
import unittest

from helpers import LABELED, RepoTestCase

from datavault.errors import ValidationFailed
from datavault.experiments import (
    compare_experiments,
    get_experiment,
    list_experiments,
    reproduce,
    run_experiment,
)
from datavault.ml.model import NaiveBayes, TFIDFVectorizer, evaluate, stratified_split
from datavault.ml.preprocessing import preprocess, preprocess_labeled_text, split_label
from datavault.verify import verify


class PreprocessingTests(unittest.TestCase):
    def test_remove_duplicates_works_on_labeled_rows(self):
        raw = b"good movie|positive\ngood movie|positive\nbad film|negative\nbad film|negative\nfine plot|positive\nweak cast|negative\n"
        out = preprocess_labeled_text(raw, {"remove_duplicates": True}).decode().splitlines()
        self.assertEqual(len(out), 4)

    def test_duplicates_with_different_labels_are_kept(self):
        rows = [("same", "a"), ("same", "b")]
        self.assertEqual(len(preprocess(rows, {"remove_duplicates": True})), 2)

    def test_duplicates_are_detected_after_normalisation(self):
        rows = [("Hello!", "a"), ("hello", "a")]
        cfg = {"lowercase": True, "remove_punctuation": True, "remove_duplicates": True}
        self.assertEqual(preprocess(rows, cfg), [("hello", "a")])

    def test_stopword_removal_drops_emptied_rows(self):
        rows = [("the and of", "a"), ("good thing", "a")]
        self.assertEqual(preprocess(rows, {"remove_stopwords": True}), [("good thing", "a")])

    def test_unknown_or_non_boolean_steps_rejected(self):
        with self.assertRaises(ValidationFailed):
            preprocess([], {"lemmatize": True})
        with self.assertRaises(ValidationFailed):
            preprocess([], {"lowercase": "yes"})

    def test_comment_keys_are_ignored(self):
        self.assertEqual(preprocess([("A", "x")], {"_comment": "hi", "lowercase": True}), [("a", "x")])

    def test_label_splits_on_last_pipe(self):
        self.assertEqual(split_label("a | b|pos"), ("a | b", "pos"))
        self.assertEqual(split_label("no label"), ("no label", None))
        self.assertEqual(split_label("text|"), ("text|", None))


class ModelTests(unittest.TestCase):
    def test_tfidf_idf_matches_formula(self):
        vec = TFIDFVectorizer().fit(["a b", "a c"])
        import math
        self.assertAlmostEqual(vec.idf["a"], math.log(3 / 3) + 1)
        self.assertAlmostEqual(vec.idf["b"], math.log(3 / 2) + 1)

    def test_vectorizer_ignores_unseen_tokens(self):
        vec = TFIDFVectorizer().fit(["a b"])
        self.assertEqual(vec.transform(["zzz"]), [{}])

    def test_evaluate_against_hand_computed_values(self):
        y_true = ["a", "a", "a", "b", "b", "b"]
        y_pred = ["a", "a", "b", "b", "b", "a"]
        r = evaluate(y_true, y_pred, ["a", "b"])
        self.assertEqual(r["confusion_matrix"], {"a": {"a": 2, "b": 1}, "b": {"a": 1, "b": 2}})
        self.assertAlmostEqual(r["accuracy"], 0.6667, places=4)
        self.assertAlmostEqual(r["per_class"]["a"]["precision"], 0.6667, places=4)
        self.assertAlmostEqual(r["macro_f1"], 0.6667, places=4)

    def test_classifier_learns_separable_data(self):
        texts = ["good great", "great fine", "bad awful", "awful poor"]
        labels = ["p", "p", "n", "n"]
        vec = TFIDFVectorizer()
        clf = NaiveBayes().fit(vec.fit_transform(texts), labels, vec.vocab_size)
        self.assertEqual(clf.predict(vec.transform(["great", "awful"])), ["p", "n"])

    def test_split_is_deterministic_stratified_and_disjoint(self):
        texts = [f"t{i}" for i in range(20)]
        labels = ["a"] * 10 + ["b"] * 10
        first = stratified_split(texts, labels, 0.3, 1)
        self.assertEqual(first, stratified_split(texts, labels, 0.3, 1))
        self.assertNotEqual(first, stratified_split(texts, labels, 0.3, 2))
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(sorted(first[3]), ["a"] * 3 + ["b"] * 3)

    def test_split_rejects_singleton_classes(self):
        with self.assertRaises(ValidationFailed):
            stratified_split(["a", "b", "c"], ["x", "x", "y"], 0.3, 1)


class ExperimentTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.c1 = self.commit_file("data.txt", LABELED, "labeled")

    def test_record_contains_lineage_environment_and_artifacts(self):
        rec = run_experiment(self.repo, "HEAD")
        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["dataset"], {"commit": self.c1.id, "content_id": self.c1.content_id})
        self.assertIn("python", rec["environment"])
        self.assertEqual(set(rec["artifacts"]), {"model", "metrics"})
        self.assertEqual(get_experiment(self.repo, rec["id"][:8])["id"], rec["id"])
        self.assertTrue(verify(self.repo, deep=True).ok)

    def test_reproduction_matches_metrics_and_model_artifact(self):
        rec = run_experiment(self.repo, "HEAD", {"seed": 7})
        result = reproduce(self.repo, rec["id"])
        self.assertTrue(result["same_dataset_content"] and result["same_config"])
        self.assertTrue(result["same_metrics"])
        self.assertTrue(result["same_model_artifact"])

    def test_different_seed_changes_the_split(self):
        a = run_experiment(self.repo, "HEAD", {"seed": 1})
        b = run_experiment(self.repo, "HEAD", {"seed": 2})
        self.assertNotEqual(a["config_hash"], b["config_hash"])
        self.assertEqual(compare_experiments(a, b)["config_changes"], {"seed": {"a": 1, "b": 2}})

    def test_reproduction_refuses_corrupted_dataset(self):
        rec = run_experiment(self.repo, "HEAD")
        path = self.repo.objects.path(self.c1.files["data.txt"].sha256)
        path.chmod(0o644)
        path.write_bytes(path.read_bytes()[:-1] + b"X")
        from datavault.errors import CorruptRepository
        with self.assertRaises(CorruptRepository):
            reproduce(self.repo, rec["id"])

    def test_failed_run_is_recorded_with_reason(self):
        self.commit_file("data.txt", "a|x\nb|y\n", "tiny")
        rec = run_experiment(self.repo, "HEAD")
        self.assertEqual(rec["status"], "failed")
        self.assertIn("at least", rec["error"])
        self.assertIsNone(rec["metrics"])
        self.assertEqual(len(list_experiments(self.repo)), 1)

    def test_unknown_setting_is_rejected_before_recording(self):
        with self.assertRaises(ValidationFailed):
            run_experiment(self.repo, "HEAD", {"learning_rate": 1})
        self.assertEqual(list_experiments(self.repo), [])

    def test_leakage_is_reported(self):
        self.commit_file("data.txt", LABELED + LABELED, "duplicated")
        rec = run_experiment(self.repo, "HEAD")
        self.assertGreater(rec["metrics"]["dataset"]["duplicate_test_texts_in_train"], 0)
        self.assertTrue(any("also appear" in line for line in rec["logs"]))

    def test_comparison_reports_deltas_and_split_caveats(self):
        a = run_experiment(self.repo, "HEAD", {"alpha": 1.0})
        b = run_experiment(self.repo, "HEAD", {"alpha": 0.1})
        cmp = compare_experiments(a, b)
        self.assertTrue(cmp["same_dataset_content"])
        self.assertEqual(cmp["config_changes"], {"alpha": {"a": 1.0, "b": 0.1}})
        self.assertAlmostEqual(cmp["metrics"]["accuracy"]["delta"], cmp["metrics"]["accuracy"]["b"] - cmp["metrics"]["accuracy"]["a"], places=4)
        self.assertTrue(any("one prediction changes accuracy" in n for n in cmp["notes"]))

    def test_tampered_experiment_record_is_detected(self):
        rec = run_experiment(self.repo, "HEAD")
        path = self.repo.experiments_dir / f"{rec['id']}.json"
        data = json.loads(path.read_text())
        data["metrics"]["test"]["accuracy"] = 1.0
        path.write_text(json.dumps(data))
        codes = {f.code for f in verify(self.repo).errors}
        self.assertIn("experiment_id_mismatch", codes)

    def test_csv_dataset_with_configured_columns(self):
        csv_rows = ["id,body,tag"] + [f"{i},{'good great' if i % 2 else 'bad awful'} {i % 3},{'p' if i % 2 else 'n'}" for i in range(24)]
        self.commit_file("r.csv", "\n".join(csv_rows) + "\n", "csv")
        rec = run_experiment(self.repo, "HEAD", {"file": "r.csv", "text_column": "body", "label_column": "tag"})
        self.assertEqual(rec["status"], "completed")
        self.assertGreaterEqual(rec["metrics"]["test"]["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
