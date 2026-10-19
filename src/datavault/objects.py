from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Iterator

from .errors import CorruptRepository, UnsafePath
from .fsutil import atomic_write, fsync_dir
from .hashing import CHUNK, is_hex_digest, sha256_bytes
from .models import FileEntry


class ObjectStore:
    def __init__(self, root: Path, tmp_dir: Path):
        self.root = root
        self.tmp_dir = tmp_dir

    def path(self, digest: str) -> Path:
        if not is_hex_digest(digest):
            raise CorruptRepository(f"invalid object id: {digest!r}")
        return self.root / digest[:2] / digest[2:]

    def exists(self, digest: str) -> bool:
        return self.path(digest).is_file()

    def put_bytes(self, data: bytes) -> FileEntry:
        digest = sha256_bytes(data)
        target = self.path(digest)
        if not target.exists():
            atomic_write(target, data, self.tmp_dir)
            os.chmod(target, 0o444)
        return FileEntry(digest, len(data))

    def put_file(self, source: Path, max_bytes: int | None = None) -> FileEntry:
        info = os.lstat(source)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise UnsafePath(f"not a regular file: {source}")
        if max_bytes is not None and info.st_size > max_bytes:
            raise UnsafePath(f"{source} exceeds the {max_bytes} byte limit")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # hash while copying so the stored bytes are exactly the hashed bytes
        digest = hashlib.sha256()
        size = 0
        fd, tmp_name = tempfile.mkstemp(dir=self.tmp_dir, prefix=".obj-")
        try:
            with os.fdopen(fd, "wb") as out, source.open("rb") as src:
                while chunk := src.read(CHUNK):
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise UnsafePath(f"{source} exceeds the {max_bytes} byte limit")
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            hexdigest = digest.hexdigest()
            target = self.path(hexdigest)
            if target.exists():
                os.unlink(tmp_name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.chmod(tmp_name, 0o444)
                os.replace(tmp_name, target)
                fsync_dir(target.parent)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        return FileEntry(hexdigest, size)

    def read_bytes(self, digest: str) -> bytes:
        data = self.path(digest).read_bytes()
        if sha256_bytes(data) != digest:
            raise CorruptRepository(f"object {digest[:12]} failed hash verification")
        return data

    def check(self, entry: FileEntry) -> str | None:
        path = self.path(entry.sha256)
        if not path.is_file():
            return "missing"
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != entry.sha256:
            return "hash mismatch"
        if size != entry.size:
            return "size mismatch"
        return None

    def iter_ids(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for bucket in sorted(self.root.iterdir()):
            if len(bucket.name) != 2 or not bucket.is_dir():
                continue
            for item in sorted(bucket.iterdir()):
                candidate = bucket.name + item.name
                if is_hex_digest(candidate):
                    yield candidate

    def copy_to(self, entry: FileEntry, target: Path) -> None:
        data = self.read_bytes(entry.sha256)
        if len(data) != entry.size:
            raise CorruptRepository(f"object {entry.sha256[:12]} has the wrong size")
        atomic_write(target, data)
