from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .datasets import detect_format
from .errors import CorruptRepository, DatavaultError, ExperimentError, InvalidReference, ValidationFailed
from .hashing import canonical_json, is_hex_digest, sha256_bytes, sha256_file, sha256_json
from .ml.model import NaiveBayes, TFIDFVectorizer, evaluate, stratified_split, top_features
from .ml.preprocessing import normalize_config, preprocess
from .profile import load_table
from .repo import Repository

SCHEMA_VERSION = 1
LOCKFILES = ("uv.lock", "poetry.lock", "Pipfile.lock", "requirements.lock", "requirements.txt")
DEFAULTS: dict[str, Any] = {
    "file": None,
    "text_column": "text",
    "label_column": "label",
    "preprocessing": {},
    "alpha": 1.0,
    "test_ratio": 0.25,
    "seed": 42,
    "top_features": 8,
}
MIN_ROWS = 4

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ValidationFailed(f"unknown experiment settings: {unknown}")
    merged = {**DEFAULTS, **config}
    merged["preprocessing"] = normalize_config(merged["preprocessing"])
    if not isinstance(merged["seed"], int) or isinstance(merged["seed"], bool):
        raise ValidationFailed("seed must be an integer")
    for key in ("alpha", "test_ratio"):
        if isinstance(merged[key], bool) or not isinstance(merged[key], (int, float)):
            raise ValidationFailed(f"{key} must be a number")
    return merged


def record_id(record: dict[str, Any]) -> str:
    return sha256_json({k: v for k, v in record.items() if k != "id"})


def load_record_file(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorruptRepository(f"experiment record unreadable: {exc}")
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise CorruptRepository("unsupported experiment record")
    required = ("id", "status", "created_at", "dataset", "config", "config_hash", "environment", "artifacts")
    missing = [k for k in required if k not in record]
    if missing:
        raise CorruptRepository(f"experiment record missing fields: {missing}")
    dataset = record["dataset"]
    if not isinstance(dataset, dict) or not is_hex_digest(str(dataset.get("commit", ""))):
        raise CorruptRepository("experiment record has an invalid dataset commit")
    if record["status"] not in ("completed", "failed"):
        raise CorruptRepository("experiment record has an invalid status")
    return record


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def capture_environment(root: Path) -> dict[str, Any]:
    revision = _git(root, "rev-parse", "HEAD")
    dirty = None if revision is None else bool(_git(root, "status", "--porcelain"))
    lock = next((root / name for name in LOCKFILES if (root / name).is_file()), None)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "datavault": __version__,
        "code_revision": revision,
        "code_dirty": dirty,
        "lockfile": lock.name if lock else None,
        "lockfile_sha256": sha256_file(lock)[0] if lock else None,
    }


def _select_file(commit_files: dict[str, Any], wanted: str | None) -> str:
    if wanted:
        if wanted not in commit_files:
            raise ExperimentError(f"{wanted} is not part of this dataset version")
        return wanted
    usable = [p for p in commit_files if detect_format(p) in ("csv", "tsv", "labeled-text")]
    if len(usable) != 1:
        raise ExperimentError(
            f"expected exactly one trainable file, found {len(usable)}; set 'file' in the config"
        )
    return usable[0]


def _load_rows(repo: Repository, commit_id: str, config: dict[str, Any], logs: list[str]) -> list[tuple[str, str]]:
    commit = repo.load_commit(commit_id)
    path = _select_file(commit.files, config["file"])
    columns, table = load_table(path, repo.objects.read_bytes(commit.files[path].sha256))
    text_col = "text" if detect_format(path) == "labeled-text" else config["text_column"]
    label_col = "label" if detect_format(path) == "labeled-text" else config["label_column"]
    for col in (text_col, label_col):
        if col not in columns:
            raise ExperimentError(f"column {col!r} not found in {path}")
    rows = [(r[text_col], r[label_col]) for r in table if r[text_col].strip() and r[label_col].strip()]
    skipped = len(table) - len(rows)
    logs.append(f"loaded {len(rows)} labeled rows from {path}")
    if skipped:
        logs.append(f"skipped {skipped} rows with missing text or label")
    return rows


def train(rows: list[tuple[str, str]], config: dict[str, Any], logs: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    processed = [r for r in preprocess([(t, l) for t, l in rows], config["preprocessing"]) if r[1] is not None]
    if len(processed) < MIN_ROWS:
        raise ExperimentError(f"need at least {MIN_ROWS} labeled rows after preprocessing, got {len(processed)}")
    texts = [t for t, _ in processed]
    labels = [l for _, l in processed if l is not None]
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ExperimentError("need at least two classes")
    tr_x, te_x, tr_y, te_y = stratified_split(texts, labels, config["test_ratio"], config["seed"])
    leaked = len(set(te_x) & set(tr_x))
    if leaked:
        logs.append(f"warning: {leaked} distinct test texts also appear in the training split")
    vectorizer = TFIDFVectorizer()
    tr_vec = vectorizer.fit_transform(tr_x)
    te_vec = vectorizer.transform(te_x)
    clf = NaiveBayes(config["alpha"]).fit(tr_vec, tr_y, vectorizer.vocab_size)
    train_eval = evaluate(tr_y, clf.predict(tr_vec), classes)
    test_eval = evaluate(te_y, clf.predict(te_vec), classes)
    logs.append(f"train={len(tr_x)} test={len(te_x)} vocabulary={vectorizer.vocab_size}")
    metrics = {
        "test": test_eval,
        "train": train_eval,
        "dataset": {
            "rows": len(processed),
            "class_distribution": {c: labels.count(c) for c in classes},
            "train_size": len(tr_x),
            "test_size": len(te_x),
            "vocabulary_size": vectorizer.vocab_size,
            "duplicate_test_texts_in_train": leaked,
        },
        "top_features": {c: top_features(clf, vectorizer, c, config["top_features"]) for c in classes},
    }
    model = {"vectorizer": vectorizer.to_dict(), "classifier": clf.to_dict()}
    return metrics, model


def run_experiment(
    repo: Repository,
    version: str,
    config: dict[str, Any] | None = None,
    *,
    parent_experiment: str | None = None,
    clock: Clock = _utcnow,
) -> dict[str, Any]:
    commit_id = repo.resolve(version)
    commit = repo.load_commit(commit_id)
    cfg = normalize_experiment_config(config or {})
    if parent_experiment:
        parent_experiment = resolve_experiment(repo, parent_experiment)
    logs: list[str] = []
    artifacts: dict[str, Any] = {}
    metrics = None
    error = None
    started = time.monotonic()
    try:
        rows = _load_rows(repo, commit_id, cfg, logs)
        metrics, model = train(rows, cfg, logs)
        artifacts["model"] = repo.objects.put_bytes(canonical_json(model)).to_dict()
        artifacts["metrics"] = repo.objects.put_bytes(canonical_json(metrics)).to_dict()
    except DatavaultError as exc:
        error = str(exc)
        logs.append(f"failed: {error}")
        metrics = None
        artifacts = {}
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if error else "completed",
        "created_at": clock().astimezone(timezone.utc).isoformat(timespec="milliseconds"),
        "dataset": {"commit": commit_id, "content_id": commit.content_id},
        "parent_experiment": parent_experiment,
        "config": cfg,
        "config_hash": sha256_json(cfg),
        "environment": capture_environment(repo.root),
        "metrics": metrics,
        "artifacts": artifacts,
        "logs": logs,
        "error": error,
        "duration_seconds": round(time.monotonic() - started, 4),
    }
    record["id"] = record_id(record)
    from .fsutil import atomic_write

    atomic_write(
        repo.experiments_dir / f"{record['id']}.json",
        (json.dumps(record, sort_keys=True, indent=2) + "\n").encode(),
        repo.tmp,
    )
    return record


def list_experiments(repo: Repository) -> list[dict[str, Any]]:
    records = [load_record_file(p) for p in repo.experiments_dir.glob("*.json")]
    return sorted(records, key=lambda r: (r["created_at"], r["id"]))


def resolve_experiment(repo: Repository, ref: str) -> str:
    records = [p.stem for p in repo.experiments_dir.glob("*.json")]
    if ref == "latest":
        listed = list_experiments(repo)
        if not listed:
            raise InvalidReference("no experiments recorded")
        return listed[-1]["id"]
    matches = [r for r in records if r.startswith(ref)] if len(ref) >= 4 else []
    if len(matches) == 1:
        return matches[0]
    raise InvalidReference(f"unknown or ambiguous experiment: {ref!r}")


def get_experiment(repo: Repository, ref: str) -> dict[str, Any]:
    return load_record_file(repo.experiments_dir / f"{resolve_experiment(repo, ref)}.json")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            out.update(_flatten(inner, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


def compare_experiments(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    flat_a, flat_b = _flatten(a["config"]), _flatten(b["config"])
    config_changes = {
        k: {"a": flat_a.get(k), "b": flat_b.get(k)}
        for k in sorted(set(flat_a) | set(flat_b))
        if flat_a.get(k) != flat_b.get(k)
    }
    metrics: dict[str, Any] = {}
    notes: list[str] = []
    if a["metrics"] and b["metrics"]:
        for name in ("accuracy", "macro_f1"):
            va, vb = a["metrics"]["test"][name], b["metrics"]["test"][name]
            metrics[name] = {"a": va, "b": vb, "delta": round(vb - va, 4)}
        for cls in sorted(set(a["metrics"]["test"]["per_class"]) & set(b["metrics"]["test"]["per_class"])):
            va = a["metrics"]["test"]["per_class"][cls]["f1"]
            vb = b["metrics"]["test"]["per_class"][cls]["f1"]
            metrics[f"f1[{cls}]"] = {"a": va, "b": vb, "delta": round(vb - va, 4)}
        na, nb = a["metrics"]["test"]["n"], b["metrics"]["test"]["n"]
        metrics["test_size"] = {"a": na, "b": nb, "delta": nb - na}
        smallest = min(na, nb)
        notes.append(
            f"smallest test set has {smallest} rows; one prediction changes accuracy by {1 / smallest:.4f}"
        )
        if na != nb or a["config"]["seed"] != b["config"]["seed"] or a["config"]["test_ratio"] != b["config"]["test_ratio"]:
            notes.append("the two runs do not share an identical test split")
    return {
        "a": a["id"],
        "b": b["id"],
        "same_dataset_content": a["dataset"]["content_id"] == b["dataset"]["content_id"],
        "same_dataset_commit": a["dataset"]["commit"] == b["dataset"]["commit"],
        "config_changes": config_changes,
        "metrics": metrics,
        "environment_changes": {
            k: {"a": a["environment"].get(k), "b": b["environment"].get(k)}
            for k in sorted(set(a["environment"]) | set(b["environment"]))
            if a["environment"].get(k) != b["environment"].get(k)
        },
        "notes": notes,
    }


def reproduce(repo: Repository, ref: str, *, record: bool = False) -> dict[str, Any]:
    original = get_experiment(repo, ref)
    if original["status"] != "completed":
        raise ExperimentError("only completed experiments can be reproduced")
    commit = repo.load_commit(original["dataset"]["commit"])
    for rel, entry in commit.files.items():
        problem = repo.objects.check(entry)
        if problem:
            raise CorruptRepository(f"dataset object for {rel} is {problem}; cannot reproduce")
    logs: list[str] = []
    cfg = original["config"]
    rows = _load_rows(repo, original["dataset"]["commit"], cfg, logs)
    metrics, model = train(rows, cfg, logs)
    model_digest = sha256_bytes(canonical_json(model))
    env_now = capture_environment(repo.root)
    env_diff = {
        k: {"recorded": original["environment"].get(k), "current": env_now.get(k)}
        for k in ("python", "implementation", "platform", "lockfile_sha256")
        if original["environment"].get(k) != env_now.get(k)
    }
    result = {
        "experiment": original["id"],
        "same_dataset_content": commit.content_id == original["dataset"]["content_id"],
        "same_config": sha256_json(normalize_experiment_config(cfg)) == original["config_hash"],
        "same_metrics": metrics == original["metrics"],
        "same_model_artifact": model_digest == original["artifacts"]["model"]["sha256"],
        "environment_differences": env_diff,
    }
    if record:
        rerun = run_experiment(repo, original["dataset"]["commit"], cfg, parent_experiment=original["id"])
        result["recorded_as"] = rerun["id"]
    return result
