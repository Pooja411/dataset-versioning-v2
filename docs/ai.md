# AI features

AI is optional. Every other command works without credentials or network access.

## Configuration

| Variable | Meaning |
| --- | --- |
| `DATAVAULT_AI_PROVIDER` | `anthropic`, `openai-compatible` (any `/chat/completions` server, including local ones), or `mock` (testing) |
| `DATAVAULT_AI_MODEL` | Required. No default is shipped, so a stale model name can never be baked in |
| `ANTHROPIC_API_KEY` or `DATAVAULT_AI_API_KEY` | Key for Anthropic. `OPENAI_API_KEY` or `DATAVAULT_AI_API_KEY` for openai-compatible, optional for local servers |
| `DATAVAULT_AI_BASE_URL` | Required for openai-compatible, e.g. `http://localhost:11434/v1` |
| `DATAVAULT_AI_TIMEOUT` | Seconds, default 60 |
| `DATAVAULT_AI_ALLOW_EXTERNAL=1` | Same as `datavault config set ai.allow_external true` |

Providers that send data off the machine are refused until you opt in. A provider counts as local only if its base URL host is `localhost`, `127.0.0.1` or `::1`.

```
export DATAVAULT_AI_PROVIDER=anthropic
export DATAVAULT_AI_MODEL=<model name from your provider's documentation>
export ANTHROPIC_API_KEY=...
datavault config set ai.allow_external true
datavault ai dataset v1 HEAD
```

The Anthropic adapter follows the documented Messages API shape (`/v1/messages`, `x-api-key`, `anthropic-version`). It has been exercised only against a mocked transport; check it against a live account before relying on it.

Retries: up to two retries with backoff for HTTP 429, 5xx, timeouts and connection errors, honouring `Retry-After` up to 30 seconds. Authentication errors are not retried. Prompts above 60,000 characters are refused before sending. Error messages never include keys or response bodies.

## What is sent

Aggregate, deterministic facts only: file names and sizes, column names, row counts, missing-value fractions, numeric min/max/mean, category counts for columns with at most 20 distinct values, preprocessing flags, quality-rule results, experiment settings and metrics, and the last few log lines. Raw rows are never sent. All text passes through a sanitiser that strips control characters, truncates and redacts emails, phone-like numbers, long hex strings and common API-key shapes. `--redact-values` also removes category value names. Column names and file names are still sent, so avoid sensitive names if that matters.

## How claims are grounded

The model must return JSON with fixed sections. Every finding has `kind` (`verified` or `hypothesis`) and `evidence` ids. The validator then checks:

- structure: all sections present, correct types, valid kinds, otherwise the response is rejected
- every cited id exists
- a `verified` claim cites at least one fact
- every number in a `verified` claim appears in the cited facts (percent and fraction forms are accepted)
- a `verified` claim does not use causal wording ("because", "caused", "due to", ...)

Failures are shown as `UNSUPPORTED` with the reason, not silently dropped. Hypotheses are labelled as such. This catches invented numbers and fabricated citations; it cannot tell whether a prose interpretation is wise. Treat `grounded` as "traceable to a measured fact", not "correct".

Each analysis is saved to `.datavault/ai/` with the evidence table, request hash, raw response, provider, model, latency and token counts when the provider reports them.

## Test assistant

`ai suggest-tests path.py::func` sends only that function's source (at most 8,000 characters), an optional requirements file inside the repository, and optional known-defect notes. Files that look like secrets are refused. The reply must be a `unittest` module; it is statically reviewed (import allow-list, no `eval`/`exec`/`open`/dunder escapes) and written to `.datavault/ai/proposed_tests/`. It is never written into your tests directory.

`ai run-tests` runs a proposal only when you pass the SHA-256 (prefix) of the exact bytes you reviewed. It runs in a separate interpreter (`-I`) with a scrubbed environment, a temp working directory, CPU, memory, file-size and descriptor limits, a wall-clock timeout, and a network namespace if `unshare` works on the host (the result says whether it did). It is disabled on non-POSIX systems. This limits accidents and runaway code; it is not a sandbox against a determined attacker, and the proposed test can still read files the user can read.

## Evaluation

`python -m datavault.ai.evals` runs fixtures for schema changes, row-count changes, missing-value regressions, category shift and no change, measures structured validity, evidence consistency, unsupported-claim rate and key-fact coverage, checks that malformed or contradictory responses are rejected, and checks that a generated test suite passes on correct code and fails on a seeded defect. The default responders are deterministic stand-ins: one that cites facts faithfully and one that hallucinates. They show that the checks work. They say nothing about how any real model performs. Run the same harness with a real provider before making claims about one.
