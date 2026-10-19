# CLI reference

All commands accept `-C DIR` to run in another directory. Commands marked **json** accept `--json`.

Exit codes: `0` success, `1` failure or verification errors, `2` usage error, `3` AI unavailable (not configured, provider failure, invalid model output).

| Command | Purpose |
| --- | --- |
| `init` | Create `.datavault/` in the current directory |
| `add PATH...` | Stage files or directories (symlinks are refused, dot-directories skipped) |
| `rm PATH...` | Unstage tracked files |
| `status` **json** | Staged and working-tree changes for tracked files |
| `commit -m MSG [--preprocess FILE] [--checks FILE] [--allow-failures] [--annotate K=V]` **json** | Snapshot staged files |
| `log [REF] [-n N]` **json** | History, topologically ordered |
| `show REF` **json** | One version's manifest |
| `diff A B` **json** | File, schema, preprocessing and statistical differences |
| `branch [NAME [START]] [-d NAME]` | List, create, delete branches |
| `tag [NAME [START]] [-d NAME]` | List, create, delete tags (tags are immutable) |
| `checkout REF [--force]` | Restore tracked files; refuses to overwrite local edits |
| `export REF DEST [--force]` | Write a version's files to an empty directory |
| `verify [--all] [--version REF]` **json** | Integrity check. Default is structural; `--all` rehashes every object; `--version` rehashes one version |
| `gc [--apply] [--grace SECONDS]` **json** | Dry run unless `--apply`. Keeps anything reachable from a branch, tag, HEAD, the index or an experiment |
| `config get` / `config set KEY VALUE` | `author`, `max_file_bytes`, `ai.allow_external` |
| `check REF --rules FILE` **json** | Run quality rules against a version |
| `experiment run REF [--config FILE] [--parent ID]` **json** | Train and record |
| `experiment list` / `show ID` / `compare A B` / `reproduce ID [--record]` | Inspect and reproduce |
| `ai dataset A B [--redact-values]` | Explain a dataset change |
| `ai experiments A B` | Explain an experiment difference |
| `ai summarize ID` | Summarise one experiment |
| `ai suggest-tests PATH::FUNC [--requirements FILE] [--defect TEXT]` | Propose tests (never executed automatically) |
| `ai run-tests PROPOSAL --approve SHA256` | Run a reviewed proposal in a resource-limited subprocess |
| `import-legacy PATH` | Import versions from the original `dataset_repo` layout |
| `serve [--host H] [--port P] [--no-browser]` | Read-only dashboard |

References accepted anywhere a `REF` is expected: `HEAD`, `HEAD~N`, branch, tag, `heads/NAME`, `tags/NAME`, or a unique commit-id prefix of four or more hex characters. Experiments accept a unique id prefix or `latest`.

## Preprocessing config

Booleans, applied in this order: `strip_whitespace`, `lowercase`, `remove_punctuation`, `remove_duplicates` (on text and label together), `tokenize`, `remove_stopwords`. Unknown steps are an error. Keys starting with `_` are ignored. Preprocessing applies to `.txt` files in `text|label` format.

## Quality rules

```json
{"rules": [
  {"check": "required_columns", "columns": ["id", "text"]},
  {"check": "column_type", "column": "rating", "type": "int"},
  {"check": "max_missing", "column": "*", "fraction": 0.05, "severity": "warning"}
]}
```

Checks: `required_columns`, `column_type` (`int`, `float`, `bool`, `str`), `max_missing`, `no_duplicate_rows`, `unique_key`, `allowed_values`, `range` (`min`, `max`), `min_rows`, `max_rows`. Each rule accepts `path` (a glob) and `severity` (`error` by default, or `warning`). Errors block a commit unless `--allow-failures` is given. Reports contain counts and row numbers, never cell values. The rule file is stored as an object and referenced from the commit.

## Experiment config

`file`, `text_column`, `label_column`, `preprocessing`, `alpha`, `test_ratio`, `seed`, `top_features`. Unknown keys are rejected. The split is stratified and seeded.
