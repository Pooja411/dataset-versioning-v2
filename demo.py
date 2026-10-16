"""
demo.py
-------
One-click demo script. Works on Windows, macOS, and Linux.
No multi-line shell commands needed.

What it does
------------
1. Resets the repository (removes old versions)
2. Creates 5 dataset versions with a proper DAG tree
3. Trains a Naive Bayes classifier on every version
4. Opens the interactive dashboard in your browser

Usage
-----
    python demo.py              # full reset + rebuild + open dashboard
    python demo.py --dashboard  # just regenerate data + open dashboard (no reset)

Requirements
------------
    Python 3.8+  (no other dependencies)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# ── make sure we can import project modules regardless of cwd ────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from storage import StorageManager
from versioning import DatasetVersionManager
from preprocessing import PreprocessingPipeline
from train import train_and_evaluate


REPO      = ROOT / "dataset_repo"
RAW_FILE  = REPO / "raw" / "dataset.txt"


# ── Version definitions ──────────────────────────────────────────
# Each entry: (name, message, parent, step, config_overrides)
# config_overrides: keys to set True (all others False except lowercase+strip_whitespace always True)

VERSIONS = [
    (
        "v1-raw",
        "Root version: lowercase + strip only. No punctuation removal or stopwords.",
        None,   # no parent → root
        "initial ingestion",
        # enabled steps:
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": False, "remove_duplicates": False,
         "tokenize": False, "remove_stopwords": False},
    ),
    (
        "v2-dedup-punct",
        "Added deduplication and punctuation removal on top of v1-raw.",
        "v1-raw",
        "remove_duplicates + remove_punctuation",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": False, "remove_stopwords": False},
    ),
    (
        "v3-stopwords",
        "Added stopword removal on top of v2. Testing if removing function words helps.",
        "v2-dedup-punct",
        "remove_stopwords",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": False, "remove_stopwords": True},
    ),
    (
        "v3-tokenized",
        "Alternative to v3-stopwords: tokenize only, no stopword removal. Parallel experiment.",
        "v2-dedup-punct",
        "tokenize",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": True, "remove_stopwords": False},
    ),
    (
        "v4-full",
        "Full pipeline: all 6 steps. Best cleaning. Derived from v3-stopwords + tokenize.",
        "v3-stopwords",
        "tokenize (completing full pipeline)",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": True, "remove_stopwords": True},
    ),
]


def reset_repo():
    """Remove all existing versions and registry, keep raw data."""
    versions_dir = REPO / "versions"
    registry     = REPO / "registry.json"

    if versions_dir.exists():
        shutil.rmtree(versions_dir)
        print("[reset] Removed existing versions/")

    if registry.exists():
        registry.unlink()
        print("[reset] Removed registry.json")


def create_versions():
    """Create all demo versions programmatically (no interactive prompts)."""
    storage = StorageManager(repo_path=str(REPO))
    storage.init()
    manager = DatasetVersionManager(storage=storage)

    print("\n" + "═" * 60)
    print("  Creating dataset versions")
    print("═" * 60)

    for name, message, parent, step, config in VERSIONS:
        print(f"\n  ▶  {name}")
        print(f"     Parent : {parent or 'none (root)'}")
        print(f"     Step   : {step}")

        result = manager.create_version(
            str(RAW_FILE),
            config,
            name=name,
            message=message,
            parent_ref=parent,
            transformation_step=step,
        )

        if result["created"]:
            m = result["metrics"]
            print(f"     [ok] Created  hash={result['version_id'][:16]}…")
            print(f"          rows={m['num_rows']}  vocab={m['vocabulary_size']}  "
                  f"avg_len={m['avg_sentence_len']}")
        else:
            print(f"     [skip] Already exists")

    return storage, manager


def train_versions():
    """Train classifier on every version and save training reports."""
    storage = StorageManager(repo_path=str(REPO))
    manager = DatasetVersionManager(storage=storage)

    print("\n" + "═" * 60)
    print("  Training models on each version")
    print("═" * 60)

    for name, *_ in VERSIONS:
        print(f"\n  ▶  Training on {name} ...")
        try:
            report = train_and_evaluate(name, storage, manager, verbose=False)
            ta = report["test_metrics"]["accuracy"]
            f1 = report["test_metrics"]["macro_f1"]
            vs = report["dataset_stats"]["vectorizer_vocab_size"]
            print(f"     [ok] accuracy={ta:.4f}  macro_f1={f1:.4f}  vocab={vs}")
        except Exception as e:
            print(f"     [warn] Training failed: {e}")


def open_dashboard():
    """Regenerate dashboard_data.json and open the browser."""
    import webbrowser

    storage = StorageManager(repo_path=str(REPO))
    manager = DatasetVersionManager(storage=storage)

    dag          = manager.build_dag()
    data_file    = ROOT / "dashboard_data.json"
    html_file    = ROOT / "dashboard.html"

    data_file.write_text(json.dumps(dag, indent=2), encoding="utf-8")
    print(f"\n[ok] dashboard_data.json written  ({len(dag['nodes'])} nodes, {len(dag['edges'])} edges)")

    if not html_file.exists():
        print(f"[error] dashboard.html not found at {html_file}")
        sys.exit(1)

    html_path = html_file.resolve()

    if sys.platform == "win32":
        uri = "file:///" + str(html_path).replace("\\", "/")
    else:
        uri = html_path.as_uri()

    print(f"[ok] Opening → {uri}")
    print(f"\n     If the browser does not open automatically,")
    print(f"     manually open this file in your browser:")
    print(f"     {html_path}\n")

    opened = webbrowser.open(uri)
    if not opened and sys.platform == "win32":
        import os
        os.startfile(str(html_path))


def print_lineage():
    """Print the ASCII lineage tree in the terminal."""
    storage = StorageManager(repo_path=str(REPO))
    manager = DatasetVersionManager(storage=storage)
    versions = manager.list_versions()

    children = {}
    id_to_v  = {v["version_id"]: v for v in versions}
    roots    = []
    for v in versions:
        pid = v.get("parent_id")
        if pid and pid in id_to_v:
            children.setdefault(pid, []).append(v["version_id"])
        else:
            roots.append(v["version_id"])

    def tree(vid, prefix="", is_last=True):
        v    = id_to_v[vid]
        conn = "└─ " if is_last else "├─ "
        step = f"  [{v['transformation_step']}]" if v.get("transformation_step") else ""
        print(f"  {prefix}{conn}{v['name']}{step}")
        kids = children.get(vid, [])
        for i, kid in enumerate(kids):
            ext = "   " if is_last else "│  "
            tree(kid, prefix + ext, i == len(kids) - 1)

    print("\n  VERSION LINEAGE DAG")
    print("  " + "─" * 50)
    for i, root in enumerate(roots):
        tree(root, "", i == len(roots) - 1)
    print()


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DSV demo script — creates versions, trains models, opens dashboard.",
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Skip version creation and training; just regenerate data and open dashboard.",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Generate dashboard_data.json but do not open browser.",
    )
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   DSV — Dataset Versioning System  ·  Demo Runner      ║")
    print("╚" + "═"*58 + "╝")

    if not RAW_FILE.exists():
        print(f"\n[error] Raw dataset not found: {RAW_FILE}")
        print("        Make sure dataset_repo/raw/dataset.txt exists.")
        sys.exit(1)

    if not args.dashboard_only:
        reset_repo()
        create_versions()
        train_versions()
        print_lineage()
    else:
        print("\n[info] --dashboard-only: skipping version creation and training.")

    if not args.no_browser:
        open_dashboard()
    else:
        # Still write the data file
        storage = StorageManager(repo_path=str(REPO))
        manager = DatasetVersionManager(storage=storage)
        dag = manager.build_dag()
        data_file = ROOT / "dashboard_data.json"
        data_file.write_text(json.dumps(dag, indent=2), encoding="utf-8")
        print(f"\n[ok] dashboard_data.json written ({len(dag['nodes'])} nodes)")
        print(f"     Open dashboard.html in your browser to view.")

    print("\n[done] Demo complete!\n")


if __name__ == "__main__":
    main()
