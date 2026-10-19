from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import CorruptRepository
from .fsutil import clean_relpath
from .hashing import is_hex_digest, sha256_json

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FileEntry:
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size": self.size}

    @staticmethod
    def from_dict(raw: Any) -> "FileEntry":
        if not isinstance(raw, dict) or set(raw) != {"sha256", "size"}:
            raise CorruptRepository("file entry must contain exactly sha256 and size")
        digest, size = raw["sha256"], raw["size"]
        if not isinstance(digest, str) or not is_hex_digest(digest):
            raise CorruptRepository("file entry has an invalid sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CorruptRepository("file entry has an invalid size")
        return FileEntry(digest, size)


def _file_map(raw: Any, label: str) -> dict[str, FileEntry]:
    if not isinstance(raw, dict):
        raise CorruptRepository(f"{label} must be an object")
    return {clean_relpath(path): FileEntry.from_dict(entry) for path, entry in raw.items()}


@dataclass(frozen=True)
class Commit:
    parents: tuple[str, ...]
    message: str
    author: str
    created_at: str
    files: dict[str, FileEntry]
    sources: dict[str, FileEntry] = field(default_factory=dict)
    preprocessing: dict[str, Any] | None = None
    schema: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] | None = None
    annotations: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def content_id(self) -> str:
        return sha256_json(
            {
                "files": {p: e.to_dict() for p, e in sorted(self.files.items())},
                "sources": {p: e.to_dict() for p, e in sorted(self.sources.items())},
                "preprocessing": self.preprocessing,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "content_id": self.content_id,
            "parents": list(self.parents),
            "message": self.message,
            "author": self.author,
            "created_at": self.created_at,
            "files": {p: e.to_dict() for p, e in sorted(self.files.items())},
            "sources": {p: e.to_dict() for p, e in sorted(self.sources.items())},
            "preprocessing": self.preprocessing,
            "schema": self.schema,
            "quality": self.quality,
            "annotations": self.annotations,
        }

    @property
    def id(self) -> str:
        return sha256_json(self.to_dict())

    @staticmethod
    def from_dict(raw: Any) -> "Commit":
        if not isinstance(raw, dict):
            raise CorruptRepository("commit must be an object")
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CorruptRepository(f"unsupported commit schema_version: {version!r}")
        required = {
            "content_id", "parents", "message", "author", "created_at", "files",
            "sources", "preprocessing", "schema", "quality", "annotations",
        }
        missing = required - set(raw)
        if missing:
            raise CorruptRepository(f"commit is missing fields: {sorted(missing)}")
        parents = raw["parents"]
        if not isinstance(parents, list) or not all(
            isinstance(p, str) and is_hex_digest(p) for p in parents
        ):
            raise CorruptRepository("commit has invalid parents")
        for key in ("message", "author", "created_at"):
            if not isinstance(raw[key], str):
                raise CorruptRepository(f"commit field {key} must be a string")
        if raw["preprocessing"] is not None and not isinstance(raw["preprocessing"], dict):
            raise CorruptRepository("preprocessing must be an object or null")
        if not isinstance(raw["schema"], dict) or not isinstance(raw["annotations"], dict):
            raise CorruptRepository("schema and annotations must be objects")
        commit = Commit(
            parents=tuple(parents),
            message=raw["message"],
            author=raw["author"],
            created_at=raw["created_at"],
            files=_file_map(raw["files"], "files"),
            sources=_file_map(raw["sources"], "sources"),
            preprocessing=raw["preprocessing"],
            schema=raw["schema"],
            quality=raw["quality"],
            annotations={str(k): str(v) for k, v in raw["annotations"].items()},
        )
        if commit.content_id != raw["content_id"]:
            raise CorruptRepository("content_id does not match files")
        return commit
