from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ValidationFailed
from .fsutil import atomic_write
from .ml.preprocessing import STEPS
from .models import Commit
from .repo import Repository

DATASET_FILE = "dataset.txt"


def _timestamp(meta: dict[str, Any]) -> str:
    if "created_ts" in meta:
        moment = datetime.fromtimestamp(float(meta["created_ts"]), timezone.utc)
    else:
        moment = datetime.strptime(meta.get("created_at", "1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
    return moment.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def import_legacy(repo: Repository, legacy_repo: Path) -> dict[str, Any]:
    versions_dir = legacy_repo / "versions"
    if not versions_dir.is_dir():
        raise ValidationFailed(f"{legacy_repo} is not a legacy dataset_repo (no versions directory)")
    entries = []
    for folder in sorted(p for p in versions_dir.iterdir() if p.is_dir()):
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            rows = (folder / DATASET_FILE).read_bytes()
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries.append((meta, rows, config, folder.name))
    entries.sort(key=lambda e: (e[0].get("created_ts", 0.0), e[0].get("creation_index", 0), e[3]))

    by_legacy_id: dict[str, str] = {}
    imported: list[str] = []
    skipped: list[str] = []
    tags: dict[str, str] = {}
    last = None
    for meta, rows, config, folder_name in entries:
        legacy_id = meta.get("version_id", folder_name)
        if legacy_id in by_legacy_id:
            skipped.append(f"{folder_name} (duplicate of {meta.get('name')})")
            continue
        entry = repo.objects.put_bytes(rows)
        parent = by_legacy_id.get(meta.get("parent_id") or "")
        commit = Commit(
            parents=(parent,) if parent else (),
            message=meta.get("message") or meta.get("name") or folder_name,
            author="legacy-import",
            created_at=_timestamp(meta),
            files={DATASET_FILE: entry},
            preprocessing={s: bool(config.get(s, False)) for s in STEPS},
            annotations={
                k: str(v)
                for k, v in {
                    "legacy_name": meta.get("name"),
                    "legacy_version_id": legacy_id,
                    "transformation_step": meta.get("transformation_step"),
                }.items()
                if v
            },
        )
        path = repo.commits_dir / f"{commit.id}.json"
        if not path.exists():
            atomic_write(path, (json.dumps(commit.to_dict(), sort_keys=True, indent=2) + "\n").encode(), repo.tmp)
        by_legacy_id[legacy_id] = commit.id
        imported.append(commit.id)
        last = commit.id
        name = meta.get("name", "")
        tag = name if name and name not in tags else f"{name}-{legacy_id[:8]}" if name else legacy_id[:12]
        tags[tag] = commit.id
    for tag, cid in tags.items():
        if not repo._read_ref(repo.tags_dir, tag):
            repo.create_tag(tag, cid)
    if last and repo._read_ref(repo.heads_dir, "main") is None:
        repo._write_ref(repo.heads_dir, "main", last)
    return {"imported": len(imported), "skipped": skipped, "tags": sorted(tags)}
