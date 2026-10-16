# Dataset Versioning System (DSV)

A minimal, Git-inspired CLI tool for versioning text datasets in machine learning reproducibility workflows. Includes a full training pipeline, interactive web dashboard, DAG-based lineage tracking, and automatic file watching.

Built with Python standard library only. No scikit-learn, no databases — filesystem storage only.

---

## Project Structure

```
dataset_versioning_system/
├── dsv.py              ← Main CLI (init / create / list / show / compare / lineage / dashboard / watch)
├── train.py            ← Training CLI  (--version / --compare)
├── watch.py            ← File watcher (auto-creates versions on raw data change)
├── demo.py             ← One-click demo script (no interactive prompts)
├── preprocessing.py    ← Preprocessing pipeline (6 configurable steps)
├── versioning.py       ← Version creation, hashing, DAG, comparison logic
├── metrics.py          ← Dataset-level metrics (rows, vocab, avg length …)
├── model.py            ← TF-IDF vectoriser + Naive Bayes classifier (from scratch)
├── storage.py          ← Filesystem storage layer + name registry
├── dashboard.html      ← Interactive web dashboard (GitHub-dark theme)
├── dashboard_data.json ← Auto-generated data file consumed by dashboard
└── dataset_repo/
    ├── raw/
    │   └── dataset.txt          ← 40 labeled ML sentences (text|label)
    └── versions/
        └── <hash_id>/
            ├── dataset.txt      ← Preprocessed rows (text|label)
            ├── config.json      ← Preprocessing config for this version
            ├── metrics.json     ← Dataset-level metrics
            ├── meta.json        ← name, message, parent, created_at, creation_index
            └── training_report.json  ← Written by train.py
```

---

## Quick Start

```bash
# 1. One-click demo — creates 5 versions, trains all, opens dashboard
python demo.py

# 2. Or start fresh manually
python dsv.py init

python dsv.py create \
    --data dataset_repo/raw/dataset.txt \
    --name "v1-baseline" \
    --message "Baseline: lowercase + strip only" \
    --step "initial ingestion"

# 3. Train a classifier on a version
python train.py --version v1-baseline

# 4. Open the interactive dashboard
python dsv.py dashboard

# 5. Watch raw files and auto-create versions on change
python dsv.py watch
```

**Windows users** — use single-line commands (no backslash continuation):
```cmd
python dsv.py create --data dataset_repo/raw/dataset.txt --name "v1-baseline" --message "Baseline" --step "initial ingestion"
```

---

## CLI Reference

### `dsv.py` — Dataset Versioning

| Command | Description |
|---|---|
| `python dsv.py init` | Initialise a new repository |
| `python dsv.py create --data FILE --name NAME --message MSG [--parent VERSION] [--step DESC]` | Create a new version (config entered interactively) |
| `python dsv.py list` | List all versions with parent and metrics |
| `python dsv.py show VERSION` | Show full details of a version |
| `python dsv.py compare VERSION_A VERSION_B` | Side-by-side metric and config diff |
| `python dsv.py lineage [VERSION]` | Print full DAG tree or trace a version's ancestry |
| `python dsv.py dashboard [--port PORT]` | Serve dashboard at http://localhost:8765 |
| `python dsv.py watch [--data FILE] [--interval N] [--no-train] [--no-dashboard]` | Watch raw files and auto-version on change |

### `train.py` — Model Training

| Command | Description |
|---|---|
| `python train.py --version NAME` | Train and evaluate on one version |
| `python train.py --compare VERSION_A VERSION_B` | Train both and compare results |

### `demo.py` — One-Click Demo

| Command | Description |
|---|---|
| `python demo.py` | Full reset + create 5 versions + train all + open dashboard |
| `python demo.py --dashboard-only` | Just refresh data and open dashboard |
| `python demo.py --no-browser` | Run everything but don't open browser |

---

## System Design

### Modules

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Apply ordered, configurable text transformations |
| `metrics.py` | Compute descriptive statistics on processed rows |
| `storage.py` | Read/write artefacts; name → hash registry; precise ordering |
| `versioning.py` | Hashing, version creation, DAG builder, comparison |
| `model.py` | TF-IDF vectoriser + Naive Bayes classifier, from scratch |
| `train.py` | Training pipeline CLI; saves report; cross-version compare |
| `watch.py` | MD5-based file watcher; auto-creates versions on change |
| `dsv.py` | Unified CLI entry point for all commands |

### Data Flow

```
raw/dataset.txt  (text|label rows)
        │
        ├── Manual:  python dsv.py create  →  interactive config prompt
        └── Auto:    python dsv.py watch   →  inherits latest version's config
                │
                ▼
          versioning.py
                │  splits text from label (preserves | through preprocessing)
                │  runs PreprocessingPipeline on text only
                │  reattaches label
                │  computes SHA-256 hash
                │  computes dataset metrics
                └► saves to versions/<hash>/
                        │
                        ▼
                   train.py
                        │  loads version rows
                        │  parses text + labels
                        │  stratified 75/25 train-test split (seed=42)
                        │  fits TFIDFVectorizer on TRAIN only (no leakage)
                        │  trains NaiveBayesClassifier
                        │  evaluates on test set
                        └► saves training_report.json into version folder
                                │
                                ▼
                         dsv.py dashboard
                                │  builds DAG from all version metas
                                │  writes dashboard_data.json
                                └► serves at http://localhost:8765
```

---

## Hashing Strategy

Each version receives a **64-character SHA-256 hex digest** computed from:

```
SHA256( raw_file_content  +  "||"  +  canonical_json(config) )
```

- `raw_file_content` — exact UTF-8 bytes of the input file
- `canonical_json(config)` — JSON with **sorted keys**, no extra whitespace

| Scenario | Result |
|---|---|
| Same file + same config | Same hash — idempotent, no duplicate version created |
| Different config, same file | New version ID |
| Different file, same config | New version ID |
| Any byte change in either | New version ID |

This means the core reproducibility guarantee — **any change in raw data or preprocessing config always creates a new immutable version** — is enforced mathematically by the hash, not just behaviorally.

---

## Preprocessing Pipeline

Steps are always applied in this fixed canonical order, regardless of config key order:

| Order | Step | Effect |
|---|---|---|
| 1 | `strip_whitespace` | Collapse spaces, strip leading/trailing whitespace |
| 2 | `lowercase` | Lowercase all characters |
| 3 | `remove_punctuation` | Remove all `string.punctuation` characters |
| 4 | `remove_duplicates` | Remove repeated rows (preserve first occurrence) |
| 5 | `tokenize` | Normalise whitespace token boundaries |
| 6 | `remove_stopwords` | Remove ~80 predefined English stopwords |

**Label preservation:** When rows follow `text|label` format, the pipeline preprocesses only the text portion. The `|label` suffix is split off before preprocessing and reattached after — so `remove_punctuation` cannot destroy the separator.

**Config change → new version:** Because the config is part of the SHA-256 hash input, changing even one preprocessing step always produces a different hash and therefore a new immutable version. This is guaranteed at the cryptographic level.

---

## DAG — Version Lineage Tracking

Every version stores three lineage fields in its `meta.json`:

```json
{
  "parent_name": "v2-dedup-punct",
  "parent_id":   "ab02beb2...",
  "transformation_step": "remove_stopwords"
}
```

- **`parent_name` / `parent_id`** — links this version to its parent in the DAG
- **`transformation_step`** — human-readable label for what changed (appears as the edge label in the dashboard)

### Parent resolution

When creating a version with `python dsv.py create`:
- `--parent VERSION` — explicitly set the parent (use a version name or hash)
- No `--parent` flag — automatically uses the most recently created version as parent
- First version ever created — becomes the root node (no parent)

### Auto-versioning (watch) and DAG

When `python dsv.py watch` detects a file change:
- The new version is automatically linked as a child of the **most recently created version**
- The preprocessing config is inherited from that same parent version
- The transformation step is set to `"auto: file changed at <timestamp>"`
- The new node appears in the DAG immediately after running `python dsv.py dashboard`

### Version ordering

Versions are ordered by `(created_ts, creation_index)` — a float epoch timestamp plus an integer counter. This ensures correct ordering even when multiple versions are created in the same second (e.g. by `demo.py`), which previously caused the wrong version to be identified as "latest."

### CLI lineage commands

```bash
# Print full DAG tree in terminal
python dsv.py lineage

# Trace ancestry of one version
python dsv.py lineage v4-full
# Output: v1-raw → v2-dedup-punct → v3-stopwords → v4-full
```

---

## Model: Bag-of-Words + Naive Bayes

### Why this model?

Naive Bayes is ideal for demonstrating dataset version effects:
- **Interpretable** — every learned word probability can be inspected
- **Sensitive to preprocessing** — vocabulary changes directly change features
- **Instant training** — version comparisons run in milliseconds
- **Transparent math** — no black-box components

### TF-IDF Vectorisation (from scratch)

```
TF(t, d)      = count(t in d) / total_tokens(d)
IDF(t)        = log((1 + N) / (1 + df(t))) + 1   ← smoothed
TF-IDF(t, d)  = TF(t, d) * IDF(t)
```

Vocabulary is built from the **training set only** — no test-set leakage.

### Multinomial Naive Bayes (from scratch)

```
log P(c|d)  ∝  log P(c)  +  Σ_t  tfidf(t,d) * log P(t|c)
```

Laplace smoothing (α=1) prevents zero-probability for unseen tokens. Train/test split is stratified 75/25 with fixed seed 42.

---

## Dataset Format

Rows follow the format:
```
<text>|<label>
```
Example:
```
Deep learning models achieve state-of-the-art results.|positive
Overfitting occurs when a model memorizes training data.|negative
```

Labels can be any string. The pipeline auto-detects labeled rows and preserves the `|label` suffix through all preprocessing steps.

**Unlabeled datasets** are also supported — version creation works normally, but training is skipped with an informative message.

---

## Web Dashboard

Launched via `python dsv.py dashboard`, served at `http://localhost:8765` (not `file://` — this avoids browser CORS restrictions).

| Tab | Contents |
|---|---|
| **Overview** | Stat cards, recent versions, lineage tree, preprocessing step usage |
| **Lineage DAG** | Interactive D3 graph — zoom/pan, click node for detail panel |
| **Versions** | Full table with parent, transformation step, accuracy — click to expand |
| **Metrics** | Chart.js charts: accuracy/F1, vocabulary, tokens, avg length + analysis |
| **Compare** | Side-by-side metric diff, config diff, top features comparison |

---

## File Watcher

```bash
# Watch all files in dataset_repo/raw/ (default)
python dsv.py watch

# Watch a specific file
python dsv.py watch --data dataset_repo/raw/dataset.txt

# Adjust polling interval (default: 2 seconds)
python dsv.py watch --interval 3

# Skip auto-training
python dsv.py watch --no-train

# Skip dashboard data refresh
python dsv.py watch --no-dashboard
```

The watcher uses MD5 hashing to detect changes. It does **not** watch for config changes — config changes are intentional and handled through `python dsv.py create`, which guarantees a new version via the SHA-256 hash.

---

## Reproducibility Guarantees

1. **Deterministic hashing** — identical inputs always yield the same version ID
2. **Immutable storage** — version directories are written once, never modified
3. **Fixed preprocessing order** — `ORDERED_STEPS` is hardcoded, independent of config key order
4. **Fixed train/test seed** — `RANDOM_SEED = 42` in `train.py`, stratified split
5. **Train-only vocabulary** — TF-IDF fitted on training data only, no leakage
6. **Precise version ordering** — `(created_ts, creation_index)` ensures deterministic "latest" resolution even when versions share the same second timestamp
7. **Self-contained artefacts** — each version folder stores config, meta, and training report — fully reproducible from the folder alone

---

## Requirements

- Python 3.8+
- No external dependencies beyond the standard library
- Works on Windows, macOS, and Linux
- Compatible with Google Colab