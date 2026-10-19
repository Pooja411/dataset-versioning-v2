from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .datasets import detect_format, infer_schema
from .errors import (
    ConflictError,
    CorruptRepository,
    InvalidReference,
    RepositoryExists,
    RepositoryNotFound,
    UnsafePath,
    ValidationFailed,
)
from .fsutil import atomic_write, clean_relpath, file_lock, validate_ref_name, within
from .hashing import canonical_json, is_hex_digest, sha256_file
from .models import Commit, FileEntry
from .objects import ObjectStore

DIRNAME = ".datavault"
FORMAT_VERSION = 1
DEFAULT_BRANCH = "main"
DEFAULT_CONFIG: dict[str, Any] = {
    "author": "",
    "max_file_bytes": 2 * 1024**3,
    "ai": {"allow_external": False},
}

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Head:
    branch: str | None
    commit: str | None


@dataclass
class Status:
    branch: str | None
    head: str | None
    staged_added: list[str]
    staged_modified: list[str]
    staged_removed: list[str]
    modified: list[str]
    missing: list[str]

    @property
    def clean(self) -> bool:
        return not (
            self.staged_added or self.staged_modified or self.staged_removed
            or self.modified or self.missing
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "clean": self.clean}


class Repository:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.dv = self.root / DIRNAME
        self.tmp = self.dv / "tmp"
        self.objects = ObjectStore(self.dv / "objects", self.tmp)
        self.commits_dir = self.dv / "commits"
        self.heads_dir = self.dv / "refs" / "heads"
        self.tags_dir = self.dv / "refs" / "tags"
        self.experiments_dir = self.dv / "experiments"
        self.ai_dir = self.dv / "ai"
        self.lock_path = self.dv / "lock"

    @classmethod
    def init(cls, root: Path) -> "Repository":
        repo = cls(root)
        if repo.dv.exists():
            raise RepositoryExists(f"repository already exists at {repo.dv}")
        for directory in (
            repo.tmp, repo.objects.root, repo.commits_dir, repo.heads_dir,
            repo.tags_dir, repo.experiments_dir, repo.ai_dir,
        ):
            directory.mkdir(parents=True)
        atomic_write(repo.dv / "format", f"{FORMAT_VERSION}\n".encode())
        repo._write_json(repo.dv / "config.json", DEFAULT_CONFIG)
        repo._write_json(repo.dv / "index.json", {"files": {}})
        atomic_write(repo.dv / "HEAD", f"ref: {DEFAULT_BRANCH}\n".encode())
        return repo

    @classmethod
    def find(cls, start: Path | None = None) -> "Repository":
        here = (start or Path.cwd()).resolve()
        for candidate in (here, *here.parents):
            if (candidate / DIRNAME / "format").is_file():
                return cls(candidate)
        raise RepositoryNotFound("not inside a datavault repository (run 'datavault init')")

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise CorruptRepository(f"missing file: {path.relative_to(self.root)}")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorruptRepository(f"{path.relative_to(self.root)} is not valid JSON: {exc}")

    def _write_json(self, path: Path, value: Any) -> None:
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        atomic_write(path, text.encode("utf-8"), self.tmp if self.tmp.exists() else None)

    def config(self) -> dict[str, Any]:
        stored = self._read_json(self.dv / "config.json")
        merged = {**DEFAULT_CONFIG, **stored}
        merged["ai"] = {**DEFAULT_CONFIG["ai"], **stored.get("ai", {})}
        return merged

    def set_config(self, key: str, value: Any) -> None:
        with file_lock(self.lock_path):
            stored = self._read_json(self.dv / "config.json")
            target = stored
            *parents, leaf = key.split(".")
            for part in parents:
                target = target.setdefault(part, {})
                if not isinstance(target, dict):
                    raise ValidationFailed(f"{key} is not a configurable path")
            if leaf not in DEFAULT_CONFIG and parents == []:
                raise ValidationFailed(f"unknown config key: {key}")
            target[leaf] = value
            self._write_json(self.dv / "config.json", stored)

    def author(self) -> str:
        return os.environ.get("DATAVAULT_AUTHOR") or self.config()["author"] or "unknown"

    def head(self) -> Head:
        raw = (self.dv / "HEAD").read_text(encoding="utf-8").strip()
        if raw.startswith("ref: "):
            name = raw[5:]
            return Head(name, self._read_ref(self.heads_dir, name))
        if raw.startswith("commit: ") and is_hex_digest(raw[8:]):
            return Head(None, raw[8:])
        raise CorruptRepository(f"HEAD is malformed: {raw!r}")

    def _ref_file(self, base: Path, name: str) -> Path:
        validate_ref_name(name)
        path = base / name
        if not within(base, path):
            raise InvalidReference(f"invalid name: {name!r}")
        return path

    def _read_ref(self, base: Path, name: str) -> str | None:
        path = self._ref_file(base, name)
        if not path.is_file():
            return None
        value = path.read_text(encoding="utf-8").strip()
        if not is_hex_digest(value):
            raise CorruptRepository(f"ref {name} does not contain a commit id")
        return value

    def _write_ref(self, base: Path, name: str, commit_id: str) -> None:
        atomic_write(self._ref_file(base, name), f"{commit_id}\n".encode(), self.tmp)

    def _list_refs(self, base: Path) -> dict[str, str]:
        refs: dict[str, str] = {}
        if not base.exists():
            return refs
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.startswith(".tmp-"):
                name = path.relative_to(base).as_posix()
                value = path.read_text(encoding="utf-8").strip()
                refs[name] = value
        return refs

    def branches(self) -> dict[str, str]:
        return self._list_refs(self.heads_dir)

    def tags(self) -> dict[str, str]:
        return self._list_refs(self.tags_dir)

    def load_commit(self, commit_id: str) -> Commit:
        if not is_hex_digest(commit_id):
            raise InvalidReference(f"not a commit id: {commit_id!r}")
        path = self.commits_dir / f"{commit_id}.json"
        if not path.is_file():
            raise InvalidReference(f"unknown commit: {commit_id[:12]}")
        commit = Commit.from_dict(self._read_json(path))
        if commit.id != commit_id:
            raise CorruptRepository(f"commit {commit_id[:12]} does not match its content hash")
        return commit

    def commit_ids(self) -> list[str]:
        return sorted(p.stem for p in self.commits_dir.glob("*.json") if is_hex_digest(p.stem))

    def resolve(self, ref: str) -> str:
        base, _, steps = ref.partition("~")
        if steps and not steps.isdigit():
            raise InvalidReference(f"invalid ancestor suffix in {ref!r}")
        commit_id = self._resolve_base(base)
        for _ in range(int(steps or 0)):
            parents = self.load_commit(commit_id).parents
            if not parents:
                raise InvalidReference(f"{ref!r} goes past the root commit")
            commit_id = parents[0]
        return commit_id

    def _resolve_base(self, base: str) -> str:
        if base == "HEAD":
            head = self.head().commit
            if head is None:
                raise InvalidReference("HEAD does not point to a commit yet")
            return head
        if base.startswith("tags/"):
            found = self._read_ref(self.tags_dir, base[5:])
        elif base.startswith("heads/"):
            found = self._read_ref(self.heads_dir, base[6:])
        else:
            branch = self._safe_ref(self.heads_dir, base)
            tag = self._safe_ref(self.tags_dir, base)
            if branch and tag and branch != tag:
                raise InvalidReference(f"{base!r} is both a branch and a tag; use heads/ or tags/")
            found = branch or tag
        if found:
            return found
        if re.fullmatch(r"[0-9a-f]{4,64}", base):
            matches = [c for c in self.commit_ids() if c.startswith(base)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise InvalidReference(f"ambiguous commit prefix {base!r}")
        raise InvalidReference(f"unknown reference: {base!r}")

    def _safe_ref(self, base: Path, name: str) -> str | None:
        try:
            return self._read_ref(base, name)
        except InvalidReference:
            return None

    def _index(self) -> dict[str, FileEntry]:
        raw = self._read_json(self.dv / "index.json").get("files", {})
        return {clean_relpath(p): FileEntry.from_dict(e) for p, e in raw.items()}

    def _write_index(self, files: dict[str, FileEntry]) -> None:
        self._write_json(
            self.dv / "index.json",
            {"files": {p: e.to_dict() for p, e in sorted(files.items())}},
        )

    def _relpath(self, path: Path) -> str:
        absolute = path if path.is_absolute() else Path.cwd() / path
        if absolute.is_symlink():
            raise UnsafePath(f"refusing to track a symlink: {path}")
        resolved = absolute.resolve()
        try:
            rel = resolved.relative_to(self.root)
        except ValueError:
            raise UnsafePath(f"{path} is outside the repository root")
        if rel.parts and rel.parts[0] == DIRNAME:
            raise UnsafePath("cannot track files inside the repository directory")
        return clean_relpath(rel.as_posix())

    def _expand(self, paths: Iterable[Path]) -> list[Path]:
        files: list[Path] = []
        for path in paths:
            if path.is_symlink():
                raise UnsafePath(f"refusing to track a symlink: {path}")
            if path.is_dir():
                for current, dirs, names in os.walk(path):
                    dirs[:] = sorted(d for d in dirs if d != DIRNAME and not d.startswith("."))
                    for name in sorted(names):
                        candidate = Path(current) / name
                        if not candidate.is_symlink() and not name.startswith("."):
                            files.append(candidate)
            elif path.is_file():
                files.append(path)
            else:
                raise ValidationFailed(f"no such file: {path}")
        return files

    def add(self, paths: Iterable[Path]) -> list[str]:
        limit = self.config()["max_file_bytes"]
        staged: dict[str, FileEntry] = {}
        for file in self._expand(paths):
            rel = self._relpath(file)
            staged[rel] = self.objects.put_file(file, limit)
        with file_lock(self.lock_path):
            index = self._index()
            index.update(staged)
            self._write_index(index)
        return sorted(staged)

    def unstage(self, rel_paths: Iterable[str]) -> list[str]:
        removed = []
        with file_lock(self.lock_path):
            index = self._index()
            for raw in rel_paths:
                rel = clean_relpath(raw)
                if rel not in index:
                    raise ValidationFailed(f"not tracked: {rel}")
                del index[rel]
                removed.append(rel)
            self._write_index(index)
        return removed

    @staticmethod
    def raw_files(commit: Commit) -> dict[str, FileEntry]:
        return {**commit.files, **commit.sources}

    def status(self) -> Status:
        head = self.head()
        base = self.raw_files(self.load_commit(head.commit)) if head.commit else {}
        index = self._index()
        added = sorted(set(index) - set(base))
        removed = sorted(set(base) - set(index))
        changed = sorted(p for p in set(index) & set(base) if index[p] != base[p])
        modified, missing = [], []
        for rel, entry in sorted(index.items()):
            target = self.root / rel
            if not target.is_file():
                missing.append(rel)
            elif sha256_file(target)[0] != entry.sha256:
                modified.append(rel)
        return Status(head.branch, head.commit, added, changed, removed, modified, missing)

    def commit(
        self,
        message: str,
        *,
        preprocess: dict[str, Any] | None = None,
        quality_config: dict[str, Any] | None = None,
        allow_failures: bool = False,
        allow_empty: bool = False,
        annotations: dict[str, str] | None = None,
        clock: Clock = _utcnow,
    ) -> Commit:
        from .ml.preprocessing import normalize_config, preprocess_labeled_text
        from .quality import run_checks

        if not message.strip():
            raise ValidationFailed("commit message cannot be empty")
        cfg = normalize_config(preprocess) if preprocess is not None else None
        with file_lock(self.lock_path):
            head = self.head()
            if head.branch is None:
                raise ConflictError("HEAD is detached; create a branch before committing")
            index = self._index()
            if not index:
                raise ValidationFailed("nothing staged; use 'datavault add' first")
            parent = self.load_commit(head.commit) if head.commit else None

            files: dict[str, FileEntry] = {}
            sources: dict[str, FileEntry] = {}
            for rel, entry in index.items():
                problem = self.objects.check(entry)
                if problem:
                    raise CorruptRepository(f"staged object for {rel} is {problem}")
                if cfg is not None and detect_format(rel) == "labeled-text":
                    processed = preprocess_labeled_text(self.objects.read_bytes(entry.sha256), cfg)
                    files[rel] = self.objects.put_bytes(processed)
                    sources[rel] = entry
                else:
                    files[rel] = entry

            schema = {
                rel: s
                for rel in files
                if (s := infer_schema(self.objects.read_bytes(files[rel].sha256), rel))
            }
            quality = None
            if quality_config is not None:
                report = run_checks(quality_config, {r: self.objects.read_bytes(e.sha256) for r, e in files.items()})
                config_entry = self.objects.put_bytes(canonical_json(quality_config))
                quality = {
                    "config_sha256": config_entry.sha256,
                    "errors": report.error_count,
                    "warnings": report.warning_count,
                }
                if report.error_count and not allow_failures:
                    raise ValidationFailed(
                        "quality checks failed:\n" + "\n".join(report.render_failures())
                    )

            commit = Commit(
                parents=(parent_id,) if (parent_id := head.commit) else (),
                message=message.strip(),
                author=self.author(),
                created_at=clock().astimezone(timezone.utc).isoformat(timespec="seconds"),
                files=files,
                sources=sources,
                preprocessing=cfg,
                schema=schema,
                quality=quality,
                annotations=annotations or {},
            )
            if parent and parent.content_id == commit.content_id and not allow_empty:
                raise ValidationFailed("nothing to commit: dataset content is unchanged")

            # manifest first, branch ref last: the ref move is what publishes the commit
            atomic_write(
                self.commits_dir / f"{commit.id}.json",
                (json.dumps(commit.to_dict(), sort_keys=True, indent=2) + "\n").encode(),
                self.tmp,
            )
            self._write_ref(self.heads_dir, head.branch, commit.id)
            return commit

    def generation(self, commit_id: str, cache: dict[str, int] | None = None) -> int:
        cache = {} if cache is None else cache
        stack = [commit_id]
        while stack:
            current = stack[-1]
            if current in cache:
                stack.pop()
                continue
            parents = self.load_commit(current).parents
            pending = [p for p in parents if p not in cache]
            if pending:
                stack.extend(pending)
            else:
                cache[current] = 1 + max((cache[p] for p in parents), default=0)
                stack.pop()
        return cache[commit_id]

    def history(self, start: str | None = None, limit: int | None = None) -> list[tuple[str, Commit]]:
        tip = self.resolve(start or "HEAD")
        seen: dict[str, Commit] = {}
        stack = [tip]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            commit = self.load_commit(current)
            seen[current] = commit
            stack.extend(commit.parents)
        gens: dict[str, int] = {}
        for cid in seen:
            self.generation(cid, gens)
        ordered = sorted(seen.items(), key=lambda kv: (-gens[kv[0]], kv[1].created_at, kv[0]))
        return ordered[:limit] if limit else ordered

    def create_branch(self, name: str, start: str | None = None) -> str:
        target = self.resolve(start or "HEAD")
        with file_lock(self.lock_path):
            if self._read_ref(self.heads_dir, name):
                raise ConflictError(f"branch already exists: {name}")
            self._write_ref(self.heads_dir, name, target)
        return target

    def delete_branch(self, name: str) -> None:
        with file_lock(self.lock_path):
            if self.head().branch == name:
                raise ConflictError("cannot delete the checked-out branch")
            path = self._ref_file(self.heads_dir, name)
            if not path.is_file():
                raise InvalidReference(f"unknown branch: {name}")
            path.unlink()

    def create_tag(self, name: str, start: str | None = None) -> str:
        target = self.resolve(start or "HEAD")
        with file_lock(self.lock_path):
            if self._read_ref(self.tags_dir, name):
                raise ConflictError(f"tag already exists: {name} (tags are immutable)")
            self._write_ref(self.tags_dir, name, target)
        return target

    def delete_tag(self, name: str) -> None:
        with file_lock(self.lock_path):
            path = self._ref_file(self.tags_dir, name)
            if not path.is_file():
                raise InvalidReference(f"unknown tag: {name}")
            path.unlink()

    def checkout(self, ref: str, force: bool = False) -> Commit:
        with file_lock(self.lock_path):
            head = self.head()
            base_ref = ref[6:] if ref.startswith("heads/") else ref
            on_branch = (
                self._safe_ref(self.heads_dir, base_ref) is not None
                and not ref.startswith("tags/")
            )
            target_id = self.resolve(ref)
            target = self.load_commit(target_id)
            current_index = self._index()
            current_raw = self.raw_files(self.load_commit(head.commit)) if head.commit else {}

            if not force:
                for rel, entry in current_index.items():
                    path = self.root / rel
                    if path.is_file() and sha256_file(path)[0] != entry.sha256:
                        raise ConflictError(f"uncommitted changes in {rel}; commit them or use --force")
                if current_index != current_raw:
                    raise ConflictError("staged changes would be lost; commit them or use --force")

            wanted = self.raw_files(target)
            for rel in wanted:
                path = self.root / rel
                self._guard_target(path)
                if path.exists() and rel not in current_index and not force:
                    raise ConflictError(f"{rel} exists and is untracked; use --force to overwrite")
            for rel, entry in wanted.items():
                self.objects.copy_to(entry, self.root / rel)
            for rel in set(current_index) - set(wanted):
                path = self.root / rel
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            self._write_index(dict(wanted))
            pointer = f"ref: {base_ref}\n" if on_branch else f"commit: {target_id}\n"
            atomic_write(self.dv / "HEAD", pointer.encode(), self.tmp)
            return target

    def _guard_target(self, path: Path) -> None:
        if not within(self.root, path):
            raise UnsafePath(f"refusing to write outside the repository: {path}")
        for parent in path.parents:
            if parent == self.root:
                break
            if parent.is_symlink():
                raise UnsafePath(f"refusing to write through symlink {parent}")
        if path.is_symlink():
            raise UnsafePath(f"refusing to overwrite symlink {path}")

    def export(self, ref: str, destination: Path, force: bool = False) -> list[str]:
        commit = self.load_commit(self.resolve(ref))
        destination = destination.resolve()
        if destination.exists() and any(destination.iterdir()) and not force:
            raise ConflictError(f"{destination} is not empty; use --force")
        written = []
        for rel, entry in sorted(commit.files.items()):
            target = destination / rel
            if not within(destination, target):
                raise UnsafePath(f"manifest path escapes destination: {rel}")
            self.objects.copy_to(entry, target)
            written.append(rel)
        return written

    def read_file(self, ref: str, rel: str) -> bytes:
        commit = self.load_commit(self.resolve(ref))
        entry = commit.files.get(clean_relpath(rel))
        if entry is None:
            raise InvalidReference(f"{rel} is not part of {ref}")
        return self.objects.read_bytes(entry.sha256)
