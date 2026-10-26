import os
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from .errors import InvalidReference, LockTimeout, UnsafePath

try:
    import fcntl
except ImportError:
    fcntl = None

_REF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


def fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes, tmp_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=tmp_dir or path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    fsync_dir(path.parent)


def validate_ref_name(name: str) -> str:
    if not _REF_NAME.match(name) or ".." in name or name.endswith(("/", ".lock", ".")):
        raise InvalidReference(f"invalid name: {name!r}")
    if any(part.startswith(".") or not part for part in name.split("/")):
        raise InvalidReference(f"invalid name: {name!r}")
    return name


def clean_relpath(raw: str) -> str:
    if not raw or "\\" in raw or "\x00" in raw:
        raise UnsafePath(f"unsafe path: {raw!r}")
    if raw.startswith("/") or any(part in ("", ".", "..") for part in raw.split("/")):
        raise UnsafePath(f"unsafe path: {raw!r}")
    return PurePosixPath(raw).as_posix()


def within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@contextmanager
def file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    if fcntl is not None:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise LockTimeout(f"could not acquire {path.name} within {timeout}s")
                    time.sleep(0.02)
            yield
        finally:
            os.close(fd)
        return
    # no fcntl (Windows): marker file, which can be left behind by a crash
    marker = path.with_suffix(".pid")
    while True:
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise LockTimeout(f"could not acquire {path.name}; remove {marker} if stale")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.unlink(marker)
