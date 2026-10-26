from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .diff import diff_commits
from .errors import AIError, DatavaultError, ValidationFailed
from .repo import Repository

AI_UNAVAILABLE = 3


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _load_json(path: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationFailed(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        raise ValidationFailed(f"{path} must contain a JSON object")
    return value


def _short(commit_id: str) -> str:
    return commit_id[:12]


def cmd_init(args: argparse.Namespace) -> int:
    repo = Repository.init(Path.cwd())
    print(f"initialised empty repository in {repo.dv}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    repo = Repository.find()
    staged = repo.add([Path(p) for p in args.paths])
    for rel in staged:
        print(f"staged {rel}")
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    for rel in Repository.find().unstage(args.paths):
        print(f"unstaged {rel}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    status = Repository.find().status()
    if args.json:
        _emit(status.to_dict())
        return 0
    print(f"on branch {status.branch}" if status.branch else f"HEAD detached at {_short(status.head or '')}")
    groups = [
        ("staged new", status.staged_added), ("staged changed", status.staged_modified),
        ("staged removed", status.staged_removed), ("modified in working tree", status.modified),
        ("missing from working tree", status.missing),
    ]
    for title, paths in groups:
        for path in paths:
            print(f"  {title + ':':<28}{path}")
    if status.clean:
        print("nothing to commit, working tree clean")
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    repo = Repository.find()
    preprocess = _load_json(args.preprocess) if args.preprocess else None
    checks = _load_json(args.checks) if args.checks else None
    commit = repo.commit(
        args.message,
        preprocess=preprocess,
        quality_config=checks,
        allow_failures=args.allow_failures,
        annotations=dict(kv.split("=", 1) for kv in args.annotate),
    )
    if args.json:
        _emit({"id": commit.id, "content_id": commit.content_id})
    else:
        print(f"[{repo.head().branch} {_short(commit.id)}] {commit.message}")
        if commit.quality:
            print(f"quality: {commit.quality['errors']} errors, {commit.quality['warnings']} warnings")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    repo = Repository.find()
    entries = repo.history(args.ref, args.limit)
    labels: dict[str, list[str]] = {}
    for name, cid in repo.branches().items():
        labels.setdefault(cid, []).append(name)
    for name, cid in repo.tags().items():
        labels.setdefault(cid, []).append(f"tag: {name}")
    if args.json:
        _emit([
            {"id": cid, "parents": list(c.parents), "message": c.message, "author": c.author,
             "created_at": c.created_at, "content_id": c.content_id, "refs": labels.get(cid, [])}
            for cid, c in entries
        ])
        return 0
    for cid, commit in entries:
        refs = f" ({', '.join(labels[cid])})" if cid in labels else ""
        print(f"{_short(cid)}{refs}\n  {commit.created_at}  {commit.author}\n  {commit.message}\n")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    repo = Repository.find()
    cid = repo.resolve(args.ref)
    data = repo.load_commit(cid).to_dict()
    data["id"] = cid
    if args.json:
        _emit(data)
        return 0
    print(f"commit {cid}\ncontent {data['content_id']}\nauthor {data['author']}  {data['created_at']}\n\n  {data['message']}\n")
    if data["preprocessing"]:
        enabled = [k for k, v in data["preprocessing"].items() if v]
        print(f"preprocessing: {', '.join(enabled) or 'none'}")
    for path, entry in data["files"].items():
        print(f"  {entry['sha256'][:12]}  {entry['size']:>10}  {path}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    result = diff_commits(Repository.find(), args.a, args.b)
    if args.json:
        _emit(result)
        return 0
    print(f"{_short(result['a']['id'])} -> {_short(result['b']['id'])}"
          f"{'  (identical content)' if result['identical_content'] else ''}")
    files = result["files"]
    for item in files["added"]:
        print(f"  + {item['path']}")
    for item in files["removed"]:
        print(f"  - {item['path']}")
    for item in files["modified"]:
        print(f"  ~ {item['path']}  {item['size_a']} -> {item['size_b']} bytes")
    for step, change in result["preprocessing"].items():
        print(f"  preprocessing {step}: {change['a']} -> {change['b']}")
    for path, changes in result["stats"].items():
        for key in ("rows", "duplicate_rows"):
            if key in changes:
                c = changes[key]
                print(f"  {path} {key}: {c['a']} -> {c['b']}")
        for col, c in changes.get("columns", {}).items():
            for stat, value in c.items():
                print(f"  {path} [{col}] {stat}: {json.dumps(value, sort_keys=True)}")
        for key, c in changes.get("text", {}).items():
            print(f"  {path} {key}: {c['a']} -> {c['b']}")
    return 0


def cmd_branch(args: argparse.Namespace) -> int:
    repo = Repository.find()
    if args.delete:
        repo.delete_branch(args.delete)
        print(f"deleted branch {args.delete}")
    elif args.name:
        repo.create_branch(args.name, args.start)
        print(f"created branch {args.name}")
    else:
        current = repo.head().branch
        for name, cid in repo.branches().items():
            print(f"{'*' if name == current else ' '} {name}  {_short(cid)}")
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    repo = Repository.find()
    if args.delete:
        repo.delete_tag(args.delete)
        print(f"deleted tag {args.delete}")
    elif args.name:
        repo.create_tag(args.name, args.start)
        print(f"created tag {args.name}")
    else:
        for name, cid in repo.tags().items():
            print(f"{name}  {_short(cid)}")
    return 0


def cmd_checkout(args: argparse.Namespace) -> int:
    commit = Repository.find().checkout(args.ref, force=args.force)
    print(f"checked out {args.ref}: {len(commit.files)} files")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    written = Repository.find().export(args.ref, Path(args.destination), force=args.force)
    print(f"wrote {len(written)} files to {args.destination}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .verify import verify

    report = verify(Repository.find(), deep=args.all, version=args.version)
    if args.json:
        _emit(report.to_dict())
    else:
        for f in report.findings:
            print(f"{f.severity.upper():<8}{f.code:<22}{f.subject}: {f.message}")
        mode = "deep" if args.all or args.version else "structural"
        print(f"{mode} check: {report.commits_checked} commits, {report.objects_checked} object references, "
              f"{report.experiments_checked} experiments, {len(report.errors)} errors")
    return 0 if report.ok else 1


def cmd_gc(args: argparse.Namespace) -> int:
    from .gc import collect

    result = collect(Repository.find(), apply=args.apply, grace_seconds=args.grace)
    if args.json:
        _emit(result.to_dict())
        return 0
    verb = "removed" if args.apply else "would remove"
    print(f"{verb} {len(result.unreachable_commits)} commits, {len(result.unreferenced_objects)} objects "
          f"({result.bytes_reclaimable} bytes), {len(result.stale_temp)} temp files")
    if result.kept_by_grace:
        print(f"kept {len(result.kept_by_grace)} unreferenced objects younger than {args.grace}s")
    if not args.apply:
        print("dry run; pass --apply to delete")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    repo = Repository.find()
    if args.action == "get":
        _emit(repo.config())
    else:
        try:
            value = json.loads(args.value)
        except ValueError:
            value = args.value
        repo.set_config(args.key, value)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .quality import run_checks

    repo = Repository.find()
    commit = repo.load_commit(repo.resolve(args.ref))
    files = {p: repo.objects.read_bytes(e.sha256) for p, e in commit.files.items()}
    report = run_checks(_load_json(args.rules), files)
    if args.json:
        _emit(report.to_dict())
    else:
        print(f"{report.rules_run} checks, {report.error_count} errors, {report.warning_count} warnings")
        print("\n".join(report.render_failures()))
    return 1 if report.error_count else 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from . import experiments as ex

    repo = Repository.find()
    if args.action == "run":
        record = ex.run_experiment(
            repo, args.ref, _load_json(args.config) if args.config else {}, parent_experiment=args.parent
        )
        if args.json:
            _emit(record)
        elif record["status"] == "completed":
            t = record["metrics"]["test"]
            print(f"experiment {_short(record['id'])} completed: accuracy {t['accuracy']}, macro_f1 {t['macro_f1']}, n_test {t['n']}")
        else:
            print(f"experiment {_short(record['id'])} failed: {record['error']}", file=sys.stderr)
        return 0 if record["status"] == "completed" else 1
    if args.action == "list":
        records = ex.list_experiments(repo)
        if args.json:
            _emit(records)
            return 0
        for r in records:
            m = r["metrics"]["test"] if r["metrics"] else None
            score = f"acc={m['accuracy']} f1={m['macro_f1']}" if m else r["error"]
            print(f"{_short(r['id'])}  {r['status']:<9}  data={_short(r['dataset']['commit'])}  {score}")
        return 0
    if args.action == "show":
        _emit(ex.get_experiment(repo, args.ref))
        return 0
    if args.action == "compare":
        result = ex.compare_experiments(ex.get_experiment(repo, args.ref), ex.get_experiment(repo, args.other))
        if args.json:
            _emit(result)
            return 0
        print(f"{_short(result['a'])} -> {_short(result['b'])}  same dataset content: {result['same_dataset_content']}")
        for key, change in result["config_changes"].items():
            print(f"  setting {key}: {change['a']} -> {change['b']}")
        for name, m in result["metrics"].items():
            print(f"  {name}: {m['a']} -> {m['b']}  ({m.get('delta'):+})" if "delta" in m else f"  {name}: {m['a']} -> {m['b']}")
        for note in result["notes"]:
            print(f"  note: {note}")
        return 0
    result = ex.reproduce(repo, args.ref, record=args.record)
    _emit(result)
    return 0 if all(result[k] for k in ("same_dataset_content", "same_config", "same_metrics")) else 1


def _provider(repo: Repository):
    from .ai.provider import build_provider

    return build_provider(repo.config())


def _print_ai(result: Any) -> None:
    for section, items in result.analysis.sections.items():
        print(f"\n{section.replace('_', ' ').upper()}")
        for item in items:
            if item.kind == "note":
                print(f"  - {item.text}")
                continue
            tag = {"grounded": "grounded", "hypothesis": "HYPOTHESIS", "unsupported": "UNSUPPORTED"}[item.status]
            refs = f" {','.join(item.evidence)}" if item.evidence else ""
            print(f"  - [{tag}{refs}] {item.text}")
            for problem in item.problems:
                print(f"      ! {problem}")
    stats = result.analysis.stats()
    print(f"\nclaims: {stats['grounded']} grounded, {stats['hypothesis']} hypotheses, {stats['unsupported']} unsupported")
    print("Generated by a language model. Only 'grounded' items passed the automated evidence check.")
    print(f"evidence and raw response saved: .datavault/ai/{result.saved_as}")


def cmd_ai(args: argparse.Namespace) -> int:
    from .ai import analysts, testgen

    repo = Repository.find()
    try:
        if args.action == "run-tests":
            _emit(testgen.run_proposal(repo, args.proposal, args.approve))
            return 0
        provider = _provider(repo)
        if args.action == "suggest-tests":
            proposal = testgen.propose_tests(
                repo, provider, args.target, requirements_file=args.requirements, defects=args.defect
            )
            print(f"proposal: {proposal['path']}  status: {proposal['status']}")
            for p in proposal["review_problems"]:
                print(f"  rejected: {p}")
            print(f"code sha256: {proposal['code_sha256']}")
            print("Review the file, then run:\n  datavault ai run-tests "
                  f"{proposal['path']} --approve {proposal['code_sha256'][:12]}")
            return 1 if proposal["review_problems"] else 0
        if args.action == "dataset":
            result = analysts.analyze_dataset_change(repo, args.a, args.b, provider, redact_values=args.redact_values)
        elif args.action == "experiments":
            result = analysts.analyze_experiments(repo, args.a, args.b, provider)
        else:
            result = analysts.summarize_experiment(repo, args.ref, provider)
    except AIError as exc:
        print(f"AI analysis unavailable: {exc}", file=sys.stderr)
        print("Deterministic commands (diff, experiment compare, verify) work without AI.", file=sys.stderr)
        return AI_UNAVAILABLE
    if args.json:
        _emit(result.to_dict())
    else:
        _print_ai(result)
    return 0


def cmd_import_legacy(args: argparse.Namespace) -> int:
    from .legacy import import_legacy

    result = import_legacy(Repository.find(), Path(args.path))
    print(f"imported {result['imported']} versions; skipped {len(result['skipped'])} duplicates")
    for line in result["skipped"]:
        print(f"  skipped {line}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web.server import serve

    serve(Repository.find(), args.host, args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datavault", description="Dataset version control and experiment tracking.")
    parser.add_argument("--version", action="version", version=f"datavault {__version__}")
    parser.add_argument("-C", dest="directory", metavar="DIR", help="run as if started in DIR")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Callable[..., int], help_text: str, json_flag: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        if json_flag:
            p.add_argument("--json", action="store_true", help="machine-readable output")
        return p

    add("init", cmd_init, "create a repository in the current directory")
    add("add", cmd_add, "stage files or directories").add_argument("paths", nargs="+")
    add("rm", cmd_rm, "unstage tracked files").add_argument("paths", nargs="+")
    add("status", cmd_status, "show staged and working-tree changes", True)
    p = add("commit", cmd_commit, "snapshot staged files as a dataset version", True)
    p.add_argument("-m", "--message", required=True)
    p.add_argument("--preprocess", metavar="FILE", help="preprocessing config applied to labeled text files")
    p.add_argument("--checks", metavar="FILE", help="quality rules to enforce")
    p.add_argument("--allow-failures", action="store_true", help="commit even if quality errors are found")
    p.add_argument("--annotate", action="append", default=[], metavar="KEY=VALUE")
    p = add("log", cmd_log, "show history", True)
    p.add_argument("ref", nargs="?")
    p.add_argument("-n", "--limit", type=int)
    p = add("show", cmd_show, "show one version", True)
    p.add_argument("ref")
    p = add("diff", cmd_diff, "compare two versions", True)
    p.add_argument("a")
    p.add_argument("b")
    p = add("branch", cmd_branch, "list, create or delete branches")
    p.add_argument("name", nargs="?")
    p.add_argument("start", nargs="?", help="commit to point at (default HEAD)")
    p.add_argument("-d", "--delete")
    p = add("tag", cmd_tag, "list, create or delete tags")
    p.add_argument("name", nargs="?")
    p.add_argument("start", nargs="?", help="commit to point at (default HEAD)")
    p.add_argument("-d", "--delete")
    p = add("checkout", cmd_checkout, "restore the working tree to a version")
    p.add_argument("ref")
    p.add_argument("--force", action="store_true")
    p = add("export", cmd_export, "write a version's files to a directory")
    p.add_argument("ref")
    p.add_argument("destination")
    p.add_argument("--force", action="store_true")
    p = add("verify", cmd_verify, "check repository integrity", True)
    p.add_argument("--all", action="store_true", help="rehash every stored object")
    p.add_argument("--version", metavar="REF", help="deeply verify a single version")
    p = add("gc", cmd_gc, "remove unreferenced data (dry run by default)", True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--grace", type=int, default=3600, help="keep unreferenced objects newer than this many seconds")
    p = add("config", cmd_config, "read or change repository configuration")
    p.add_argument("action", choices=["get", "set"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p = add("check", cmd_check, "run quality rules against a version", True)
    p.add_argument("ref")
    p.add_argument("--rules", required=True, metavar="FILE")

    p = add("experiment", cmd_experiment, "run and compare experiments", True)
    p.add_argument("action", choices=["run", "list", "show", "compare", "reproduce"])
    p.add_argument("ref", nargs="?", help="dataset version (run) or experiment id")
    p.add_argument("other", nargs="?", help="second experiment id (compare)")
    p.add_argument("--config", metavar="FILE")
    p.add_argument("--parent", metavar="ID")
    p.add_argument("--record", action="store_true", help="store the reproduction as a new experiment")

    p = add("ai", cmd_ai, "AI-assisted analysis (optional)", True)
    ai_sub = p.add_subparsers(dest="action", required=True)
    for name, helptext in (("dataset", "explain the change between two dataset versions"),
                           ("experiments", "explain the difference between two experiments")):
        q = ai_sub.add_parser(name, help=helptext)
        q.add_argument("a")
        q.add_argument("b")
        if name == "dataset":
            q.add_argument("--redact-values", action="store_true", help="do not send category values")
    ai_sub.add_parser("summarize", help="summarise one experiment").add_argument("ref")
    q = ai_sub.add_parser("suggest-tests", help="propose unit tests for a function")
    q.add_argument("target", help="path/to/module.py::function")
    q.add_argument("--requirements", metavar="FILE")
    q.add_argument("--defect", action="append", default=[])
    q = ai_sub.add_parser("run-tests", help="run a reviewed proposal in a resource-limited subprocess")
    q.add_argument("proposal")
    q.add_argument("--approve", required=True, metavar="SHA256", help="sha256 (prefix) of the reviewed file")

    add("import-legacy", cmd_import_legacy, "import versions from an old dataset_repo directory").add_argument("path")
    p = add("serve", cmd_serve, "serve the read-only dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config" and args.action == "set" and not (args.key and args.value is not None):
        parser.error("config set requires KEY and VALUE")
    if args.command == "experiment":
        needs = {"run": 1, "show": 1, "reproduce": 1, "compare": 2}.get(args.action, 0)
        given = sum(x is not None for x in (args.ref, args.other))
        if given < needs:
            parser.error(f"experiment {args.action} needs {needs} argument(s)")
    try:
        if args.directory:
            import os
            os.chdir(args.directory)
        return args.func(args)
    except DatavaultError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
