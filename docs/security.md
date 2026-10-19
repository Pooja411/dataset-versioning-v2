# Security and reliability notes

What the tests actually demonstrate, and what they do not.

| Area | Behaviour | Evidence |
| --- | --- | --- |
| Path traversal | Relative paths with `..`, `.`, absolute, backslash or NUL are rejected on add, on manifest load and on export | `tests/test_storage.py` |
| Symlinks | Tracking a symlink is refused; checkout will not write through or over one | `test_symlink_is_refused` |
| Partial writes | Objects, manifests, refs and the index are written by temp file, fsync, rename | failure-injection tests in `tests/test_integrity.py` |
| Failed commits | A failure before the ref update leaves HEAD and history unchanged | `test_failure_before_ref_update_publishes_nothing` |
| Corruption | Missing, truncated and bit-flipped objects, edited manifests, bad refs, tampered experiment records are detected; `verify` never repairs | `VerifyTests` |
| Concurrency | Commits serialise on `flock`; eight threads with separate `Repository` objects produce a linear history | `ConcurrencyTests` (threads in one process) |
| GC | Dry run by default; protects branch, tag, HEAD, index and experiment references; grace period for fresh objects | `GarbageCollectionTests` |
| Untrusted metadata | Manifests are validated structurally, ids are re-hashed | `test_manifest_*` |
| Dashboard | Binds to 127.0.0.1, GET only, validates `Host`, fixed file names for AI records, CSP and `nosniff` | `WebTests` |
| Prompt injection | Untrusted text is only placed in JSON-encoded evidence lines; the system prompt is constant; model output never selects paths or commands | `test_hostile_filename_stays_inside_the_evidence_block` |
| Secrets | Keys are read from the environment, kept out of `repr` and errors, never written to disk, and not passed to test subprocesses | `ProviderTests`, `test_child_does_not_inherit_secrets` |

## Known limits

- Concurrency is tested with threads, not with several processes, and not on network filesystems. `flock` is unreliable on NFS. On Windows the fallback lock is a marker file that can go stale after a crash. Windows is untested.
- Read-only permissions on objects are advisory; anyone with write access to `.datavault/` can alter it. `verify` detects that, it does not prevent it.
- Durability assumes a local POSIX filesystem where `rename` is atomic and `fsync` is honoured.
- Dataset analysis reads whole files into memory (files above 64 MiB are skipped for profiling and quality checks). Object storage streams.
- The AI test runner is resource-limited, not isolated from the file system.
- The dashboard loads D3 from cdnjs without a subresource-integrity hash.
- Redaction is pattern-based and will miss unusual secrets.
