# datavault

Version control for datasets, with reproducible experiment tracking and optional, evidence-grounded AI analysis.

datavault answers the questions that come up when a model's numbers change: which data was this trained on, what changed between those two versions, can I reproduce the run, and is the difference real or noise. Everything that touches your data is deterministic and works offline. The LLM features sit on top, read measured facts only, and cannot modify anything.

Python 3.10+, no runtime dependencies.

## Contents

- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Dataset integrity](#dataset-integrity)
- [Experiments and reproducibility](#experiments-and-reproducibility)
- [AI features](#ai-features)
- [Dashboard](#dashboard)
- [What has been tested](#what-has-been-tested)
- [Limitations](#limitations)
- [Origin of the project](#origin-of-the-project)

## Quickstart

```
git clone <this repo> && cd datavault
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

python examples/demo.py          # full workflow in a temp directory
```

Or by hand:

```
mkdir work && cd work && cp ../examples/{reviews.csv,quality_rules.json,experiment.json} .
datavault init
datavault add reviews.csv
datavault commit -m "initial reviews" --checks quality_rules.json
datavault tag v1
# edit reviews.csv, then:
datavault add reviews.csv
datavault commit -m "relabel batch and add sparse rows" --checks quality_rules.json
datavault diff v1 HEAD
datavault verify --all
datavault experiment run v1   --config experiment.json
datavault experiment run HEAD --config experiment.json
datavault experiment compare <id-a> <id-b>
datavault experiment reproduce <id-a>
datavault serve
```

Output from a real run of `examples/demo.py`:

```
$ datavault commit -m relabel batch and add sparse rows --checks quality_rules.json
[main 196ead5bd65b] relabel batch and add sparse rows
quality: 0 errors, 2 warnings

$ datavault diff v1 HEAD
  reviews.csv rows: 60 -> 66
  reviews.csv [label] categories: {"counts": {"negative": {"a": 30, "b": 45, ...}, "positive": {"a": 30, "b": 21, ...}}, "total_variation": 0.181818, ...}
  reviews.csv [rating] missing_fraction: {"a": 0.0, "b": 0.090909, "delta": 0.090909}

$ datavault experiment compare 2c5c1a06277f 4f7440dda27f
  accuracy: 1.0 -> 0.6429  (-0.3571)
  macro_f1: 1.0 -> 0.5906  (-0.4094)
  note: smallest test set has 14 rows; one prediction changes accuracy by 0.0714

$ datavault experiment reproduce 2c5c1a06277f
  "same_dataset_content": true, "same_config": true, "same_metrics": true, "same_model_artifact": true
```

The sample data is synthetic. See [docs/cli.md](docs/cli.md) for every command.

## How it works

```
working tree --add--> index --commit--> commits/<id>.json --> branch / tag
                         |                    |
                    objects/ab/cd...  <-------+   (file contents, named by SHA-256)
                                              |
                          experiments/<id>.json (dataset commit, config, metrics, model artifact)
```

A commit is an immutable manifest: file paths with SHA-256 and size, parent commits, optional preprocessing config and the raw sources it was derived from, schema (CSV columns), quality-rule summary, annotations. Branches and tags are files containing a commit id. Experiments reference a dataset commit, which gives you dataset-to-result lineage without any database. Layering and the module map are in [docs/architecture.md](docs/architecture.md).

## Dataset integrity

- Content addressing: SHA-256, streamed in 1 MiB chunks. Bytes are hashed while being copied into the store, so what is hashed is what is stored.
- Two identities: `content_id` covers files, sources and preprocessing only, so timestamps, authors and messages never change it. The commit id covers the full manifest including parents.
- Safe commits: temp file, fsync, atomic rename for every write. The branch ref moves last, under a file lock. A failure before that point publishes nothing; a test injects the failure and checks HEAD and history are unchanged.
- `datavault verify` finds missing objects, size and hash mismatches, invalid or edited manifests, broken parents, invalid refs, broken experiment lineage and tampered experiment records. Exit code 1 on errors. It never rewrites anything.
- `datavault gc` is a dry run unless `--apply`, protects everything reachable from branches, tags, HEAD, the index and experiments, and leaves recent objects alone for a grace period.
- Inputs are treated as hostile: path traversal, symlinks, oversized files and malformed manifests are rejected.

## Experiments and reproducibility

The model is the original from-scratch TF-IDF plus multinomial Naive Bayes, refactored and extended with stratified seeded splits, per-class precision, recall and F1, confusion matrices, and a warning when test texts also appear in the training split.

Each run records the dataset commit and content id, normalised configuration and its hash, seed, Python and platform, Git revision and dirty flag, lockfile hash, metrics, model and metrics artifacts, logs and status. Failures are recorded too.

`experiment reproduce` reruns from the recorded inputs and reports four separate results: same dataset content, same configuration, same metrics, same model artifact bytes, plus any environment differences. Byte-identical models are verified on the machine that ran the tests; nothing is claimed across platforms or Python versions.

`experiment compare` adds the context needed to read a delta honestly, for example how much one test prediction moves accuracy and whether the two runs share a split.

## AI features

Optional. Core commands never need a key or a network.

- `ai dataset A B`: explains a dataset change from measured diff and quality facts.
- `ai experiments A B`: explains the difference between two runs from recorded settings and metrics.
- `ai summarize ID`: short summary of one experiment.
- `ai suggest-tests path.py::func` and `ai run-tests`: proposes unit tests, which are statically reviewed, stored for human review, and only run after you pass the hash of the file you reviewed.

How claims stay grounded: the model sees numbered facts, not raw rows; it must cite fact ids; each `verified` claim is checked for valid citations, numbers that appear in the cited facts and absence of causal wording; failures are displayed as `UNSUPPORTED` with the reason; interpretation must be labelled `hypothesis`. The raw response and evidence table are saved in `.datavault/ai/` so you can audit them. This detects invented numbers and fake citations. It does not prove an interpretation is right.

Providers: Anthropic, any OpenAI-compatible endpoint (including local), and a mock for tests. External providers are refused until you opt in. Setup, data-flow and privacy details: [docs/ai.md](docs/ai.md).

## Dashboard

`datavault serve` starts a read-only local server (127.0.0.1) with a D3 lineage graph: commits with branch and tag labels, experiment nodes linked to the dataset version they used, search and branch/status filters, details for versions and experiments (including configuration, environment and Git revision), side-by-side version diffs and experiment comparisons, a metrics chart, quality reports, saved AI analyses with clickable evidence, a keyboard-accessible list view, and loading, empty and error states. It reads the same repository files as the CLI and has no demo data.

## What has been tested

156 automated tests, all passing in the development environment (run with `unittest`; the same files are collected by pytest in CI). They cover storage and hashing, path and ref validation, commits, checkout and export, corruption and failure injection, thread-level concurrency, garbage collection, preprocessing, model maths against hand-computed values, experiment recording and reproduction, quality rules, diffs, provider retries and error normalisation with a fake transport, AI output validation, prompt-injection placement, the test-generation sandbox, the evaluation harness, the CLI end to end in subprocesses, the dashboard API, and importing the original project's data.

Not done in the development environment: running `ruff`, `mypy` or `pytest` themselves (not installable offline there; CI runs them), opening the dashboard in a browser, any live LLM call, Windows, multi-process concurrency.

## Limitations

- Datasets are profiled in memory; files above 64 MiB are skipped for statistics and quality checks (hashing and storage stream).
- Trainable formats are `text|label` files and CSV/TSV with a text and a label column. The classifier is deliberately simple.
- Concurrency guarantees rely on `flock` on a local filesystem.
- The test runner is resource-limited, not a security boundary.
- No remote storage or sync. No `watch` mode (the original had one; it was not ported).
- AI evaluation fixtures use deterministic stand-in responders, so the repo has no measured quality numbers for any real model.
- D3 is loaded from a CDN.

More in [docs/security.md](docs/security.md).

## Origin of the project

This began as a flat set of scripts (`dsv.py`, `storage.py`, `versioning.py`, `train.py`, a standalone `dashboard.html`) with no tests or CI. Reviewing it turned up real problems, all fixed in the rewrite:

- `remove_duplicates` did nothing on labeled data, because the pipeline ran once per row. Reproduced: 6 rows in, 6 out, 4 expected. Now covered by a regression test.
- Writes were not atomic and an existing version name could be overwritten.
- The legacy repository held 16 version folders for 13 distinct version ids, with a registry listing only 5 names.
- The test split could leave a class with no training rows; the macro F1 was computed from rounded values; the model docstring claimed NumPy but used none.

`datavault import-legacy examples/legacy_dataset_repo` imports the original versions (13 unique, 3 duplicates skipped) with their preprocessing configs, parents and names as tags. The original commit history is preserved in Git.

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). License: MIT.
