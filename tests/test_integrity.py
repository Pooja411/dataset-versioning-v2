import json
import os
import threading
import time
import unittest
from unittest import mock

from helpers import LABELED, RepoTestCase

from datavault.errors import ValidationFailed
from datavault.gc import collect
from datavault.repo import Repository
from datavault.verify import verify


class VerifyTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.c1 = self.commit_file("d.txt", "one", "one")
        self.c2 = self.commit_file("d.txt", "two", "two")

    def codes(self, **kw):
        return {f.code for f in verify(self.repo, **kw).errors}

    def test_clean_repository_passes(self):
        report = verify(self.repo, deep=True)
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.commits_checked, 2)

    def test_truncation_is_caught_by_the_cheap_check(self):
        path = self.repo.objects.path(self.c2.files["d.txt"].sha256)
        path.chmod(0o644)
        path.write_bytes(b"xy")
        self.assertEqual(self.codes(), {"size_mismatch"})
        self.assertIn("hash_mismatch", self.codes(version="HEAD"))

    def test_same_size_corruption_needs_deep_check(self):
        path = self.repo.objects.path(self.c2.files["d.txt"].sha256)
        path.chmod(0o644)
        path.write_bytes(b"abc")
        self.assertEqual(self.codes(), set())
        self.assertIn("hash_mismatch", self.codes(deep=True))

    def test_detects_missing_object(self):
        path = self.repo.objects.path(self.c1.files["d.txt"].sha256)
        path.chmod(0o644)
        path.unlink()
        self.assertIn("missing_object", self.codes())

    def test_detects_invalid_manifest(self):
        (self.repo.commits_dir / f"{self.c2.id}.json").write_text("{not json")
        self.assertIn("invalid_manifest", self.codes())

    def test_detects_edited_manifest(self):
        path = self.repo.commits_dir / f"{self.c2.id}.json"
        data = json.loads(path.read_text())
        data["message"] = "rewritten history"
        path.write_text(json.dumps(data))
        self.assertIn("commit_id_mismatch", self.codes())

    def test_detects_broken_parent(self):
        (self.repo.commits_dir / f"{self.c1.id}.json").unlink()
        self.assertIn("broken_parent", self.codes())

    def test_detects_invalid_refs(self):
        (self.repo.heads_dir / "main").write_text("f" * 64 + "\n")
        (self.repo.tags_dir / "bad").write_text("not-a-hash\n")
        report = verify(self.repo)
        subjects = {f.subject for f in report.errors if f.code == "invalid_ref"}
        self.assertIn("branch main", subjects)
        self.assertIn("tag bad", subjects)

    def test_verify_never_rewrites_corrupt_content(self):
        path = self.repo.objects.path(self.c2.files["d.txt"].sha256)
        path.chmod(0o644)
        path.write_bytes(b"abc")
        verify(self.repo, deep=True)
        self.assertEqual(path.read_bytes(), b"abc")

    def test_unknown_version_reported(self):
        self.assertIn("unknown_version", self.codes(version="nope"))


class FailureInjectionTests(RepoTestCase):
    def test_failure_before_ref_update_publishes_nothing(self):
        before = self.commit_file("d.txt", "one", "one")
        self.write("d.txt", "two")
        self.repo.add([self.root / "d.txt"])
        with mock.patch.object(Repository, "_write_ref", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.repo.commit("two")
        self.assertEqual(self.repo.head().commit, before.id)
        self.assertEqual([c for c, _ in self.repo.history()], [before.id])
        self.assertTrue(verify(self.repo, deep=True).ok)
        retried = self.repo.commit("two")
        self.assertEqual(self.repo.head().commit, retried.id)

    def test_failed_object_write_leaves_no_temp_files(self):
        path = self.write("d.txt", "payload")
        with mock.patch("datavault.objects.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.repo.objects.put_file(path)
        self.assertEqual(list(self.repo.tmp.iterdir()), [])

    def test_quality_failure_blocks_commit(self):
        self.write("t.csv", "id,label\n1,a\n1,b\n")
        self.repo.add([self.root / "t.csv"])
        rules = {"rules": [{"check": "unique_key", "columns": ["id"]}]}
        with self.assertRaises(ValidationFailed):
            self.repo.commit("dup keys", quality_config=rules)
        self.assertIsNone(self.repo.head().commit)
        commit = self.repo.commit("dup keys", quality_config=rules, allow_failures=True)
        self.assertEqual(commit.quality["errors"], 1)


class ConcurrencyTests(RepoTestCase):
    def test_parallel_commits_form_a_linear_history(self):
        self.commit_file("seed.txt", "seed", "seed")
        errors, committed = [], []

        def worker(n: int):
            repo = Repository(self.root)
            try:
                path = self.write(f"f{n}.txt", f"content {n}")
                repo.add([path])
                committed.append(repo.commit(f"commit {n}").id)
            except ValidationFailed:
                pass
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        history = self.repo.history()
        self.assertEqual(len(history), 1 + len(committed))
        self.assertTrue(all(len(c.parents) <= 1 for _, c in history))
        self.assertEqual(self.repo.commit_ids().__len__(), len(history))
        self.assertTrue(verify(self.repo, deep=True).ok)


class GarbageCollectionTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.c1 = self.commit_file("d.txt", "one", "one")
        self.c2 = self.commit_file("d.txt", "two", "two")
        self.stray = self.repo.objects.put_bytes(b"orphan data")
        old = time.time() - 7200
        os.utime(self.repo.objects.path(self.stray.sha256), (old, old))

    def test_dry_run_deletes_nothing(self):
        result = collect(self.repo)
        self.assertEqual(result.unreferenced_objects, [self.stray.sha256])
        self.assertTrue(self.repo.objects.exists(self.stray.sha256))

    def test_apply_removes_only_unreferenced_old_objects(self):
        result = collect(self.repo, apply=True)
        self.assertFalse(self.repo.objects.exists(self.stray.sha256))
        self.assertTrue(verify(self.repo, deep=True).ok)
        self.assertEqual(result.unreachable_commits, [])

    def test_recent_objects_survive_grace_period(self):
        fresh = self.repo.objects.put_bytes(b"fresh orphan")
        result = collect(self.repo, apply=True)
        self.assertIn(fresh.sha256, result.kept_by_grace)
        self.assertTrue(self.repo.objects.exists(fresh.sha256))

    def test_commits_used_by_experiments_are_protected(self):
        from datavault.experiments import run_experiment

        trained = self.commit_file("t.txt", LABELED, "labeled")
        run_experiment(self.repo, trained.id)
        self.repo._write_ref(self.repo.heads_dir, "main", self.c2.id)
        result = collect(self.repo, apply=True, grace_seconds=0)
        self.assertNotIn(trained.id, result.unreachable_commits)
        self.assertTrue(verify(self.repo, deep=True).ok)

    def test_unreferenced_commits_are_reported(self):
        trained = self.commit_file("t.txt", LABELED, "labeled")
        self.repo._write_ref(self.repo.heads_dir, "main", self.c2.id)
        self.assertEqual(collect(self.repo).unreachable_commits, [trained.id])
        collect(self.repo, apply=True, grace_seconds=0)
        self.assertFalse((self.repo.commits_dir / f"{trained.id}.json").exists())
        self.assertTrue(verify(self.repo, deep=True).ok)

    def test_staged_objects_are_protected(self):
        self.write("new.txt", "staged only")
        self.repo.add([self.root / "new.txt"])
        collect(self.repo, apply=True, grace_seconds=0)
        self.assertTrue(verify(self.repo, deep=True).ok)
        self.repo.commit("staged")
        self.assertEqual(self.repo.read_file("HEAD", "new.txt"), b"staged only")


if __name__ == "__main__":
    unittest.main()
