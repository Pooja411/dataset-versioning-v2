import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from helpers import RepoTestCase

from datavault.errors import (
    ConflictError,
    CorruptRepository,
    InvalidReference,
    RepositoryExists,
    UnsafePath,
    ValidationFailed,
)
from datavault.fsutil import clean_relpath, validate_ref_name
from datavault.hashing import sha256_bytes, sha256_file
from datavault.models import Commit, FileEntry
from datavault.repo import Repository


class ObjectStoreTests(RepoTestCase):
    def test_streaming_hash_matches_reference(self):
        payload = os.urandom(3 * 1024 * 1024 + 17)
        path = self.write("big.bin", payload)
        entry = self.repo.objects.put_file(path)
        self.assertEqual(entry.sha256, sha256_bytes(payload))
        self.assertEqual(entry.size, len(payload))
        self.assertEqual(sha256_file(path)[0], entry.sha256)

    def test_identical_content_is_stored_once(self):
        a = self.write("a.txt", "same")
        b = self.write("b.txt", "same")
        self.repo.add([a, b])
        self.assertEqual(len(list(self.repo.objects.iter_ids())), 1)

    def test_read_bytes_detects_tampering(self):
        entry = self.repo.objects.put_bytes(b"payload")
        path = self.repo.objects.path(entry.sha256)
        path.chmod(0o644)
        path.write_bytes(b"tampered")
        with self.assertRaises(CorruptRepository):
            self.repo.objects.read_bytes(entry.sha256)

    def test_size_limit_enforced_while_streaming(self):
        path = self.write("big.bin", b"x" * 2048)
        with self.assertRaises(UnsafePath):
            self.repo.objects.put_file(path, max_bytes=1024)
        self.assertEqual(list(self.repo.tmp.iterdir()), [])

    def test_symlink_is_refused(self):
        target = self.write("real.txt", "data")
        link = self.root / "link.txt"
        link.symlink_to(target)
        with self.assertRaises(UnsafePath):
            self.repo.add([link])


class CommitTests(RepoTestCase):
    def test_roundtrip_and_status(self):
        commit = self.commit_file("data.txt", "hello\n", "first")
        self.assertEqual(self.repo.head().commit, commit.id)
        self.assertEqual(self.repo.read_file("HEAD", "data.txt"), b"hello\n")
        self.assertTrue(self.repo.status().clean)

    def test_content_id_ignores_time_author_and_message(self):
        c1 = self.commit_file("d.txt", "x", "one", clock=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc))
        c2 = Commit(parents=(), message="two", author="zed", created_at="2030-01-01T00:00:00+00:00", files=c1.files)
        self.assertEqual(c1.content_id, c2.content_id)
        self.assertNotEqual(c1.id, c2.id)

    def test_unchanged_content_is_rejected(self):
        self.commit_file("d.txt", "x")
        with self.assertRaises(ValidationFailed):
            self.repo.commit("again")

    def test_status_reports_changes(self):
        self.commit_file("d.txt", "x")
        self.write("d.txt", "y")
        self.assertEqual(self.repo.status().modified, ["d.txt"])
        self.repo.add([self.root / "d.txt"])
        self.assertEqual(self.repo.status().staged_modified, ["d.txt"])

    def test_empty_message_and_empty_index_rejected(self):
        with self.assertRaises(ValidationFailed):
            self.repo.commit("msg")
        self.write("a.txt", "a")
        self.repo.add([self.root / "a.txt"])
        with self.assertRaises(ValidationFailed):
            self.repo.commit("   ")

    def test_init_twice_fails(self):
        with self.assertRaises(RepositoryExists):
            Repository.init(self.root)

    def test_paths_outside_root_rejected(self):
        outside = Path(tempfile.mkdtemp()) / "x.txt"
        outside.write_text("x")
        with self.assertRaises(UnsafePath):
            self.repo.add([outside])

    def test_repository_directory_cannot_be_tracked(self):
        with self.assertRaises(UnsafePath):
            self.repo.add([self.root / ".datavault" / "config.json"])


class PathAndRefValidationTests(unittest.TestCase):
    def test_bad_relative_paths(self):
        for bad in ["../x", "/etc/passwd", "a/../../b", "a\\b", "", "a/./b", "x\x00"]:
            with self.assertRaises(UnsafePath, msg=bad):
                clean_relpath(bad)

    def test_good_relative_path(self):
        self.assertEqual(clean_relpath("dir/file.csv"), "dir/file.csv")

    def test_bad_ref_names(self):
        for bad in ["", "..", "a..b", "-x", "a/", ".hidden", "a/.b", "x y", "a.lock"]:
            with self.assertRaises(InvalidReference, msg=bad):
                validate_ref_name(bad)

    def test_manifest_with_traversal_path_is_rejected_on_load(self):
        digest = "a" * 64
        raw = {
            "schema_version": 1, "content_id": "x", "parents": [], "message": "m", "author": "a",
            "created_at": "t", "files": {"../evil": {"sha256": digest, "size": 1}}, "sources": {},
            "preprocessing": None, "schema": {}, "quality": None, "annotations": {},
        }
        with self.assertRaises(UnsafePath):
            Commit.from_dict(raw)

    def test_manifest_rejects_bad_hash_and_unknown_schema(self):
        with self.assertRaises(CorruptRepository):
            FileEntry.from_dict({"sha256": "zz", "size": 1})
        with self.assertRaises(CorruptRepository):
            Commit.from_dict({"schema_version": 99})


class RefTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.c1 = self.commit_file("d.txt", "1", "one")
        self.c2 = self.commit_file("d.txt", "2", "two")

    def test_resolve_forms(self):
        self.assertEqual(self.repo.resolve("HEAD"), self.c2.id)
        self.assertEqual(self.repo.resolve("HEAD~1"), self.c1.id)
        self.assertEqual(self.repo.resolve("main"), self.c2.id)
        self.assertEqual(self.repo.resolve(self.c1.id[:8]), self.c1.id)
        with self.assertRaises(InvalidReference):
            self.repo.resolve("HEAD~5")
        with self.assertRaises(InvalidReference):
            self.repo.resolve("nope")

    def test_tags_are_immutable(self):
        self.repo.create_tag("v1", "HEAD~1")
        with self.assertRaises(ConflictError):
            self.repo.create_tag("v1", "HEAD")
        self.assertEqual(self.repo.resolve("v1"), self.c1.id)

    def test_branch_lifecycle(self):
        self.repo.create_branch("exp", "HEAD~1")
        self.assertIn("exp", self.repo.branches())
        with self.assertRaises(ConflictError):
            self.repo.delete_branch("main")
        self.repo.delete_branch("exp")
        self.assertNotIn("exp", self.repo.branches())

    def test_history_order_is_topological(self):
        history = [cid for cid, _ in self.repo.history()]
        self.assertEqual(history, [self.c2.id, self.c1.id])


class CheckoutTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.c1 = self.commit_file("d.txt", "one", "one")
        self.c2 = self.commit_file("d.txt", "two", "two")

    def test_checkout_restores_content_and_detaches_on_tag(self):
        self.repo.create_tag("old", "HEAD~1")
        self.repo.checkout("old")
        self.assertEqual((self.root / "d.txt").read_text(), "one")
        self.assertIsNone(self.repo.head().branch)
        with self.assertRaises(ConflictError):
            self.repo.commit("on detached")
        self.repo.checkout("main")
        self.assertEqual((self.root / "d.txt").read_text(), "two")

    def test_checkout_refuses_to_discard_local_edits(self):
        self.write("d.txt", "my edit")
        with self.assertRaises(ConflictError):
            self.repo.checkout("HEAD~1")
        self.assertEqual((self.root / "d.txt").read_text(), "my edit")
        self.repo.checkout("HEAD~1", force=True)
        self.assertEqual((self.root / "d.txt").read_text(), "one")

    def test_export_refuses_non_empty_destination(self):
        dest = self.root / "out"
        dest.mkdir()
        (dest / "keep").write_text("x")
        with self.assertRaises(ConflictError):
            self.repo.export("HEAD", dest)
        written = self.repo.export("HEAD~1", self.root / "fresh")
        self.assertEqual(written, ["d.txt"])
        self.assertEqual((self.root / "fresh" / "d.txt").read_text(), "one")


class PreprocessingCommitTests(RepoTestCase):
    def test_sources_and_files_are_recorded(self):
        from helpers import LABELED

        commit = self.commit_file(
            "data.txt", LABELED + "Great Product Works Well!|positive\n",
            preprocess={"lowercase": True, "remove_punctuation": True, "remove_duplicates": True},
        )
        self.assertIn("data.txt", commit.sources)
        self.assertNotEqual(commit.sources["data.txt"], commit.files["data.txt"])
        processed = self.repo.read_file("HEAD", "data.txt").decode()
        self.assertEqual(processed.count("great product works well|positive"), 1)
        self.assertTrue(self.repo.status().clean)

    def test_unknown_preprocessing_step_rejected(self):
        self.write("data.txt", "a|b\n")
        self.repo.add([self.root / "data.txt"])
        with self.assertRaises(ValidationFailed):
            self.repo.commit("x", preprocess={"stem_everything": True})


if __name__ == "__main__":
    unittest.main()
