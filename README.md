datavault

Git-style version control for datasets, with reproducible ML experiment tracking and evidence-checked AI analysis.

When a model's accuracy changes, the first questions are always the same: which data was it trained on, what changed in that data, can I reproduce the run, and is the difference real or just noise? datavault answers them from recorded facts, not from memory.

Pure Python 3.10+, zero runtime dependencies
Everything that touches your data is deterministic and works offline, with no API key
The LLM features are optional, read measured facts only, and cannot modify datasets, commits or experiment records
Highlights
Area	What it does
Dataset versioning	Content-addressed storage (SHA-256, streamed), immutable commit manifests, branches, immutable tags, safe checkout and export
Integrity	verify detects missing or corrupted objects, edited manifests, broken parents and refs, and tampered experiment records. It never rewrites anything
Crash safety	Every write is temp file, fsync, atomic rename. The branch pointer moves last, under a file lock, so a failed commit publishes nothing
Experiments	From-scratch TF-IDF and Naive Bayes with stratified splits, per-class metrics, confusion matrices and a train/test leakage warning. Each run records dataset, config, seed, environment and Git revision
Reproducibility	reproduce reruns from recorded inputs and reports four separate results: same data, same config, same metrics, same model bytes
Data quality	Configurable, versioned rules (types, ranges, missing values, unique keys, allowed values) with errors vs warnings, plus diffs of schema, missing values and category shifts
AI analysis	Explains dataset changes and experiment differences. Every claim must cite a measured fact and is checked automatically
Dashboard	Read-only D3 lineage graph linking experiments to the dataset versions they used, with diffs, metrics, quality reports and saved AI analyses
60-second demo
git clone <your-repo-url> && cd datavault
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python examples/demo.py

Real output from the demo, which commits a dataset, degrades it, and compares the two versions and the two models trained on them:

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

The sample data is synthetic. Run datavault serve inside the demo directory to open the dashboard.

How the AI part stays honest

Most "AI for data" features hand a model your data and trust the answer. This one does the opposite:

Deterministic code computes the diff, statistics and quality results.
Those become numbered facts (E1, E2, ...), sanitised and redacted. Raw rows are never sent.
The model must return structured JSON, with each claim marked verified or hypothesis and citing fact ids.
A validator rejects malformed output and flags each verified claim that cites unknown facts, contains numbers not in the cited facts, or uses causal wording. Flagged claims are shown as UNSUPPORTED with the reason.
The evidence table and raw response are saved for audit.

External providers are refused until you explicitly opt in. The AI test assistant never runs generated code on its own: proposals are statically reviewed, saved for human reading, and executed only after you approve the SHA-256 of the exact file, in a resource-limited subprocess.

This catches invented numbers and fake citations. It does not prove an interpretation is wise, and the docs say so.

Architecture
CLI
Repository: refs, staging,commit
Experiments
AI layer
Dashboard server
Content-addressed objectstore
TF-IDF + Naive Bayes
Diff and quality
Anthropic /OpenAI-compatible / mock

Dependencies point one way: the AI layer uses the core, never the reverse. Dataset identity is split in two: content_id covers files and preprocessing only, so timestamps and messages never change it, while the commit id covers the full manifest including history. Details in docs/architecture.md.