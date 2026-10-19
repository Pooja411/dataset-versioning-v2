from __future__ import annotations

import json
from typing import Any

from .datasets import detect_format
from .models import Commit
from .profile import profile_file, total_variation
from .quality import run_checks
from .repo import Repository


def _delta(a: Any, b: Any) -> dict[str, Any]:
    out = {"a": a, "b": b}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out["delta"] = round(b - a, 6)
    return out


def _column_changes(pa: dict[str, Any], pb: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    cols_a, cols_b = pa.get("columns", []), pb.get("columns", [])
    if set(cols_a) != set(cols_b):
        changes["columns_added"] = sorted(set(cols_b) - set(cols_a))
        changes["columns_removed"] = sorted(set(cols_a) - set(cols_b))
    if pa.get("rows") != pb.get("rows"):
        changes["rows"] = _delta(pa.get("rows"), pb.get("rows"))
    if pa.get("duplicate_rows") != pb.get("duplicate_rows"):
        changes["duplicate_rows"] = _delta(pa.get("duplicate_rows"), pb.get("duplicate_rows"))
    per_column: dict[str, Any] = {}
    for col in sorted(set(cols_a) & set(cols_b)):
        sa, sb = pa["column_stats"][col], pb["column_stats"][col]
        entry: dict[str, Any] = {}
        frac_a = sa["missing"] / sa["count"] if sa["count"] else 0.0
        frac_b = sb["missing"] / sb["count"] if sb["count"] else 0.0
        if round(frac_a, 6) != round(frac_b, 6):
            entry["missing_fraction"] = _delta(round(frac_a, 6), round(frac_b, 6))
        if sa["distinct"] != sb["distinct"]:
            entry["distinct"] = _delta(sa["distinct"], sb["distinct"])
        if ("numeric" in sa) != ("numeric" in sb):
            entry["numeric_type_changed"] = {"a": "numeric" in sa, "b": "numeric" in sb}
        elif "numeric" in sa and sa["numeric"] != sb["numeric"]:
            entry["numeric"] = {k: _delta(sa["numeric"][k], sb["numeric"][k]) for k in ("min", "max", "mean")}
        if "categories" in sa and "categories" in sb and sa["categories"] != sb["categories"]:
            ca, cb = sa["categories"], sb["categories"]
            entry["categories"] = {
                "total_variation": total_variation(ca, cb),
                "new_values": sorted(set(cb) - set(ca)),
                "dropped_values": sorted(set(ca) - set(cb)),
                "counts": {k: _delta(ca.get(k, 0), cb.get(k, 0)) for k in sorted(set(ca) | set(cb))},
            }
        if entry:
            per_column[col] = entry
    if per_column:
        changes["columns"] = per_column
    if "text" in pa and "text" in pb and pa["text"] != pb["text"]:
        changes["text"] = {k: _delta(pa["text"][k], pb["text"][k]) for k in pa["text"]}
    return changes


def _quality(repo: Repository, a: Commit, b: Commit) -> dict[str, Any] | None:
    ref = b.quality or a.quality
    if not ref:
        return None
    config = json.loads(repo.objects.read_bytes(ref["config_sha256"]))
    out = {}
    for label, commit in (("a", a), ("b", b)):
        files = {p: repo.objects.read_bytes(e.sha256) for p, e in commit.files.items()}
        out[label] = run_checks(config, files).to_dict()
    return out


def diff_commits(repo: Repository, ref_a: str, ref_b: str) -> dict[str, Any]:
    id_a, id_b = repo.resolve(ref_a), repo.resolve(ref_b)
    a, b = repo.load_commit(id_a), repo.load_commit(id_b)
    paths_a, paths_b = set(a.files), set(b.files)
    modified = sorted(p for p in paths_a & paths_b if a.files[p] != b.files[p])
    files = {
        "added": [{"path": p, "size": b.files[p].size} for p in sorted(paths_b - paths_a)],
        "removed": [{"path": p, "size": a.files[p].size} for p in sorted(paths_a - paths_b)],
        "modified": [
            {"path": p, "size_a": a.files[p].size, "size_b": b.files[p].size} for p in modified
        ],
        "unchanged": sorted(p for p in paths_a & paths_b if a.files[p] == b.files[p]),
    }
    pre_a, pre_b = a.preprocessing or {}, b.preprocessing or {}
    preprocessing = {
        k: {"a": pre_a.get(k), "b": pre_b.get(k)}
        for k in sorted(set(pre_a) | set(pre_b))
        if bool(pre_a.get(k)) != bool(pre_b.get(k))
    }
    stats: dict[str, Any] = {}
    for path in modified + sorted(paths_b - paths_a) :
        if detect_format(path) == "binary":
            continue
        pb = profile_file(path, repo.objects.read_bytes(b.files[path].sha256))
        if path in a.files:
            pa = profile_file(path, repo.objects.read_bytes(a.files[path].sha256))
            changes = _column_changes(pa, pb) if "columns" in pa and "columns" in pb else {}
        else:
            changes = {"new_file": {"rows": pb.get("rows"), "columns": pb.get("columns")}}
        if changes:
            stats[path] = changes
    return {
        "a": {"id": id_a, "message": a.message, "content_id": a.content_id},
        "b": {"id": id_b, "message": b.message, "content_id": b.content_id},
        "identical_content": a.content_id == b.content_id,
        "files": files,
        "preprocessing": preprocessing,
        "stats": stats,
        "quality": _quality(repo, a, b),
    }
