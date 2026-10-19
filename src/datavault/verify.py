from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import CorruptRepository, InvalidReference
from .hashing import is_hex_digest, sha256_file
from .models import Commit, FileEntry
from .repo import Repository

STALE_TMP_SECONDS = 3600


@dataclass
class Finding:
    code: str
    severity: str
    subject: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass
class VerifyReport:
    findings: list[Finding] = field(default_factory=list)
    commits_checked: int = 0
    objects_checked: int = 0
    experiments_checked: int = 0

    def add(self, code: str, severity: str, subject: str, message: str) -> None:
        self.findings.append(Finding(code, severity, subject, message))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "commits_checked": self.commits_checked,
            "objects_checked": self.objects_checked,
            "experiments_checked": self.experiments_checked,
            "findings": [f.to_dict() for f in self.findings],
        }


def _check_entry(repo: Repository, report: VerifyReport, subject: str, entry: FileEntry, deep: bool) -> None:
    report.objects_checked += 1
    path = repo.objects.path(entry.sha256)
    if not path.is_file():
        report.add("missing_object", "error", subject, f"object {entry.sha256[:12]} is missing")
        return
    if deep:
        problem = repo.objects.check(entry)
        if problem:
            report.add("hash_mismatch" if "hash" in problem else "size_mismatch", "error", subject,
                       f"object {entry.sha256[:12]}: {problem}")
    elif path.stat().st_size != entry.size:
        report.add("size_mismatch", "error", subject, f"object {entry.sha256[:12]} has the wrong size")


def _verify_commit(repo: Repository, report: VerifyReport, commit_id: str, deep: bool) -> Commit | None:
    path = repo.commits_dir / f"{commit_id}.json"
    subject = f"commit {commit_id[:12]}"
    try:
        commit = Commit.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (CorruptRepository, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        report.add("invalid_manifest", "error", subject, str(exc))
        return None
    except Exception as exc:
        report.add("invalid_manifest", "error", subject, f"unreadable manifest: {exc}")
        return None
    report.commits_checked += 1
    if commit.id != commit_id:
        report.add("commit_id_mismatch", "error", subject, "manifest hash does not match its filename")
    for parent in commit.parents:
        if not (repo.commits_dir / f"{parent}.json").is_file():
            report.add("broken_parent", "error", subject, f"parent {parent[:12]} does not exist")
    for rel, entry in commit.files.items():
        _check_entry(repo, report, f"{subject}:{rel}", entry, deep)
    for rel, entry in commit.sources.items():
        _check_entry(repo, report, f"{subject}:source:{rel}", entry, deep)
    if commit.quality:
        digest = commit.quality.get("config_sha256", "")
        if not is_hex_digest(digest) or not repo.objects.exists(digest):
            report.add("missing_object", "error", subject, "quality config object is missing")
    return commit


def _verify_refs(repo: Repository, report: VerifyReport) -> None:
    known = set(repo.commit_ids())
    for kind, base in (("branch", repo.heads_dir), ("tag", repo.tags_dir)):
        for name, value in repo._list_refs(base).items():
            if not is_hex_digest(value):
                report.add("invalid_ref", "error", f"{kind} {name}", "does not contain a commit id")
            elif value not in known:
                report.add("invalid_ref", "error", f"{kind} {name}", f"points to unknown commit {value[:12]}")
    try:
        head = repo.head()
        if head.commit and head.commit not in known:
            report.add("invalid_ref", "error", "HEAD", f"points to unknown commit {head.commit[:12]}")
        if head.branch and head.commit is None and repo.branches():
            report.add("unborn_head", "warning", "HEAD", f"branch {head.branch} has no commits")
    except CorruptRepository as exc:
        report.add("invalid_ref", "error", "HEAD", str(exc))


def _verify_experiments(repo: Repository, report: VerifyReport, only_commit: str | None) -> None:
    from .experiments import load_record_file, record_id

    known = set(repo.commit_ids())
    for path in sorted(repo.experiments_dir.glob("*.json")):
        subject = f"experiment {path.stem[:12]}"
        try:
            record = load_record_file(path)
        except CorruptRepository as exc:
            report.add("invalid_experiment", "error", subject, str(exc))
            continue
        if only_commit and record["dataset"]["commit"] != only_commit:
            continue
        report.experiments_checked += 1
        if record_id(record) != path.stem:
            report.add("experiment_id_mismatch", "error", subject, "record does not match its id")
        if record["dataset"]["commit"] not in known:
            report.add("broken_lineage", "error", subject,
                       f"dataset commit {record['dataset']['commit'][:12]} does not exist")
        parent = record.get("parent_experiment")
        if parent and not (repo.experiments_dir / f"{parent}.json").is_file():
            report.add("broken_lineage", "error", subject, f"parent experiment {parent[:12]} does not exist")
        for name, entry in record.get("artifacts", {}).items():
            _check_entry(repo, report, f"{subject}:{name}", FileEntry.from_dict(entry), deep=True)


def verify(repo: Repository, *, deep: bool = False, version: str | None = None) -> VerifyReport:
    report = VerifyReport()
    if version:
        try:
            commit_id = repo.resolve(version)
        except InvalidReference as exc:
            report.add("unknown_version", "error", version, str(exc))
            return report
        _verify_commit(repo, report, commit_id, deep=True)
        _verify_experiments(repo, report, commit_id)
        return report
    for commit_id in repo.commit_ids():
        _verify_commit(repo, report, commit_id, deep)
    _verify_refs(repo, report)
    _verify_experiments(repo, report, None)
    if deep:
        for digest in repo.objects.iter_ids():
            path = repo.objects.path(digest)
            if sha256_file(path)[0] != digest:
                report.add("hash_mismatch", "error", f"object {digest[:12]}", "content does not match its id")
    cutoff = time.time() - STALE_TMP_SECONDS
    for stale in repo.tmp.glob("*") if repo.tmp.exists() else []:
        if stale.stat().st_mtime < cutoff:
            report.add("stale_temp", "warning", stale.name, "leftover temporary file; 'gc' removes it")
    return report
