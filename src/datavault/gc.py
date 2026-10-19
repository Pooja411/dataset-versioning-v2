from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .fsutil import file_lock
from .repo import Repository


@dataclass
class GCResult:
    applied: bool
    unreachable_commits: list[str] = field(default_factory=list)
    unreferenced_objects: list[str] = field(default_factory=list)
    kept_by_grace: list[str] = field(default_factory=list)
    stale_temp: list[str] = field(default_factory=list)
    bytes_reclaimable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


def _roots(repo: Repository) -> set[str]:
    roots = set(repo.branches().values()) | set(repo.tags().values())
    head = repo.head().commit
    if head:
        roots.add(head)
    for path in repo.experiments_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        roots.add(record["dataset"]["commit"])
    return roots


def collect(repo: Repository, *, apply: bool = False, grace_seconds: int = 3600) -> GCResult:
    result = GCResult(applied=apply)
    with file_lock(repo.lock_path):
        reachable: set[str] = set()
        stack = [c for c in _roots(repo) if (repo.commits_dir / f"{c}.json").is_file()]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(repo.load_commit(current).parents)

        live: set[str] = set()
        for cid in reachable:
            commit = repo.load_commit(cid)
            live.update(e.sha256 for e in commit.files.values())
            live.update(e.sha256 for e in commit.sources.values())
            if commit.quality:
                live.add(commit.quality["config_sha256"])
        live.update(e.sha256 for e in repo._index().values())
        for path in repo.experiments_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            live.update(a["sha256"] for a in record.get("artifacts", {}).values())

        result.unreachable_commits = sorted(set(repo.commit_ids()) - reachable)
        now = time.time()
        for digest in repo.objects.iter_ids():
            if digest in live:
                continue
            path = repo.objects.path(digest)
            if now - path.stat().st_mtime < grace_seconds:
                result.kept_by_grace.append(digest)
                continue
            result.unreferenced_objects.append(digest)
            result.bytes_reclaimable += path.stat().st_size
        for tmp in repo.tmp.glob("*") if repo.tmp.exists() else []:
            if now - tmp.stat().st_mtime >= grace_seconds:
                result.stale_temp.append(tmp.name)

        if apply:
            for cid in result.unreachable_commits:
                (repo.commits_dir / f"{cid}.json").unlink()
            for digest in result.unreferenced_objects:
                path = repo.objects.path(digest)
                path.chmod(0o644)
                path.unlink()
            for name in result.stale_temp:
                (repo.tmp / name).unlink(missing_ok=True)
    return result
