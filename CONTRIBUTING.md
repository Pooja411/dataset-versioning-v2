# Contributing

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

Guidelines:

- Core versioning, verification and experiment tracking must keep working with no network and no credentials. AI code lives in `datavault/ai` and may only depend on the core, never the reverse.
- Anything that changes stored formats needs a `schema_version` bump and a migration note.
- New filesystem behaviour needs a test that uses a temporary directory and, where it concerns safety, a failure-injection or corruption test.
- Keep comments for the non-obvious. Prefer clearer names over explanatory comments.
- Tests that touch an LLM use `MockProvider` or a fake transport. Live-provider checks are never part of the default test run.
