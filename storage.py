"""
storage.py
----------
Filesystem-based storage layer for dataset versions.

Repository layout
-----------------
dataset_repo/
    raw/                        <- user places source datasets here
    versions/
        <version_name>/         <- folder named by human-readable name (e.g. v1-raw)
            dataset.txt         <- preprocessed rows (one per line)
            config.json         <- preprocessing configuration used
            metrics.json        <- computed dataset metrics
            meta.json           <- name, hash, message, timestamp, parent, step
    registry.json               <- index: name -> name (kept for compatibility)
"""

import json
from datetime import datetime
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional


class StorageManager:
    """
    Manages reading and writing of version artefacts on the local filesystem.
    Folders are named by version name (e.g. v1-raw), not by hash.
    The SHA-256 hash is stored inside meta.json for integrity checking.
    """

    VERSIONS_DIR  = "versions"
    RAW_DIR       = "raw"
    DATASET_FILE  = "dataset.txt"
    CONFIG_FILE   = "config.json"
    METRICS_FILE  = "metrics.json"
    META_FILE     = "meta.json"
    REGISTRY_FILE = "registry.json"

    def __init__(self, repo_path: str = "dataset_repo"):
        self.repo_path     = Path(repo_path)
        self.versions_path = self.repo_path / self.VERSIONS_DIR
        self.raw_path      = self.repo_path / self.RAW_DIR
        self.registry_path = self.repo_path / self.REGISTRY_FILE

    # ------------------------------------------------------------------
    # Repository initialisation
    # ------------------------------------------------------------------

    def init(self) -> None:
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({})

    def is_initialised(self) -> bool:
        return self.repo_path.exists() and self.versions_path.exists()

    # ------------------------------------------------------------------
    # Registry  (name -> name, kept for API compatibility)
    # ------------------------------------------------------------------

    def _read_registry(self) -> Dict[str, str]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, registry: Dict[str, str]) -> None:
        self.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def name_exists(self, name: str) -> bool:
        return (self.versions_path / name).exists()

    def resolve_name(self, name: str) -> str:
        """
        Accept a version name and return the version_id (hash) from meta.json.
        Also accepts a raw 64-char hash for backwards compatibility.
        """
        # Direct name lookup — folder named by version name
        folder = self.versions_path / name
        if folder.exists():
            meta_path = folder / self.META_FILE
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return meta.get("version_id", name)
            return name

        # Fallback: scan all folders for matching hash in meta.json
        if len(name) == 64:
            for d in self.versions_path.iterdir():
                if not d.is_dir(): continue
                meta_path = d / self.META_FILE
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("version_id") == name:
                        return name

        raise KeyError(
            f"Version '{name}' not found. "
            f"Use 'python dsv.py list' to see available versions."
        )

    def register_name(self, name: str, version_id: str) -> None:
        registry = self._read_registry()
        registry[name] = version_id
        self._write_registry(registry)

    # ------------------------------------------------------------------
    # Version persistence
    # ------------------------------------------------------------------

    def version_exists(self, version_id: str) -> bool:
        """Check if a version with this hash already exists (scan meta.json files)."""
        for d in self.versions_path.iterdir():
            if not d.is_dir(): continue
            meta_path = d / self.META_FILE
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("version_id") == version_id:
                    return True
        return False

    def save_version(
        self,
        version_id: str,
        rows: List[str],
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        name: str,
        message: str,
        parent_name: Optional[str] = None,
        parent_id:   Optional[str] = None,
        transformation_step: Optional[str] = None,
    ) -> Path:
        """
        Persist a dataset version to disk.
        Folder is named by version name (e.g. v1-raw), not by hash.
        Hash is stored inside meta.json.
        """
        # Folder named by version name — human readable
        version_dir = self.versions_path / name
        version_dir.mkdir(parents=True, exist_ok=True)

        (version_dir / self.DATASET_FILE).write_text("\n".join(rows), encoding="utf-8")
        (version_dir / self.CONFIG_FILE).write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        (version_dir / self.METRICS_FILE).write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

        registry      = self._read_registry()
        creation_index = len(registry)

        meta: Dict[str, Any] = {
            "name":                name,
            "message":             message,
            "version_id":          version_id,   # SHA-256 hash stored here
            "created_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_ts":          _time.time(),
            "creation_index":      creation_index,
            "parent_name":         parent_name,
            "parent_id":           parent_id,
            "transformation_step": transformation_step,
        }
        (version_dir / self.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self.register_name(name, version_id)
        return version_dir

    # ------------------------------------------------------------------
    # Version retrieval
    # ------------------------------------------------------------------

    def load_version(self, version_id: str) -> Dict[str, Any]:
        """
        Load all artefacts for a version.
        Accepts either a version name or a SHA-256 hash.
        """
        # Try direct folder lookup by name first
        # Check all folders and match by hash or folder name
        target_dir = None

        # First try: folder named exactly like version_id (old hash-named folders)
        direct = self.versions_path / version_id
        if direct.exists():
            target_dir = direct
        else:
            # Second try: scan for folder whose meta.json has this version_id hash
            for d in self.versions_path.iterdir():
                if not d.is_dir(): continue
                meta_path = d / self.META_FILE
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("version_id") == version_id or d.name == version_id:
                        target_dir = d
                        break

        if target_dir is None:
            raise FileNotFoundError(f"Version '{version_id}' not found in repository.")

        rows    = (target_dir / self.DATASET_FILE).read_text(encoding="utf-8").splitlines()
        config  = json.loads((target_dir / self.CONFIG_FILE).read_text(encoding="utf-8"))
        metrics = json.loads((target_dir / self.METRICS_FILE).read_text(encoding="utf-8"))
        meta_path = target_dir / self.META_FILE
        meta    = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        return {"rows": rows, "config": config, "metrics": metrics, "meta": meta}

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all versions ordered by creation time (oldest first)."""
        if not self.versions_path.exists():
            return []

        entries = []
        for d in self.versions_path.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / self.META_FILE
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            entries.append({
                # version_id is now the hash stored inside meta.json
                "version_id":          meta.get("version_id", d.name),
                "name":                meta.get("name", d.name),
                "message":             meta.get("message", ""),
                "created_at":          meta.get("created_at", ""),
                "created_ts":          meta.get("created_ts", 0.0),
                "creation_index":      meta.get("creation_index", 0),
                "parent_name":         meta.get("parent_name"),
                "parent_id":           meta.get("parent_id"),
                "transformation_step": meta.get("transformation_step"),
            })

        entries.sort(key=lambda e: (
            e.get("created_ts", 0.0),
            e.get("creation_index", 0),
        ))
        return entries

    def get_latest_version(self) -> Optional[Dict[str, Any]]:
        versions = self.list_versions()
        return versions[-1] if versions else None

    # ------------------------------------------------------------------
    # Raw data helpers
    # ------------------------------------------------------------------

    def read_raw(self, data_path: str) -> str:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        return path.read_text(encoding="utf-8")
