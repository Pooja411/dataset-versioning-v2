# Testing and troubleshooting

```
pip install -e ".[dev]"
pytest
python -m datavault.ai.evals
python examples/demo.py
```

The tests are written with `unittest` so they also run without pytest: `PYTHONPATH=src:tests python -m unittest discover -s tests`.

Suites: `test_storage` (objects, commits, refs, checkout, path validation), `test_integrity` (verification, corruption, failure injection, concurrency, GC), `test_ml` (preprocessing, model maths, experiments, reproduction), `test_quality_diff`, `test_ai` (provider retries and errors, output validation, sanitisation, analysts), `test_testgen_and_evals` (static review, approval binding, sandbox runner, eval harness), `test_cli_web_legacy` (CLI as a subprocess, dashboard API, legacy import).

Live LLM calls are not part of the suite.

## Troubleshooting

- `not inside a datavault repository`: run `datavault init`, or use `-C DIR`.
- `could not acquire lock`: another process is committing. On Windows a crashed process can leave `.datavault/lock.pid`; delete it if no datavault process is running.
- `verify` reports `hash_mismatch`: the object was modified after storage. Restore it from a backup; datavault will not rewrite it.
- `AI analysis unavailable ... not configured`: see `docs/ai.md`. Exit code 3 means AI failed, not that your data is wrong.
- `this provider sends data to an external service`: opt in with `datavault config set ai.allow_external true`.
- `automatic execution is disabled on this platform`: review the proposal file and run it yourself.
- Dashboard graph is blank offline: D3 comes from cdnjs. Vendor `d3.min.js` into `web/static` and change the script tag and CSP if you need an offline install.
