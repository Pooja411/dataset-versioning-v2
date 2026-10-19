from __future__ import annotations

import json
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..diff import diff_commits
from ..errors import DatavaultError
from ..experiments import compare_experiments, get_experiment, list_experiments
from ..repo import Repository
from ..verify import verify

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
CSP = (
    "default-src 'none'; script-src 'self' https://cdnjs.cloudflare.com; "
    "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'"
)
SAFE_AI_NAME = re.compile(r"^[a-z_]+-[0-9a-f]{16}\.json$")


def build_graph(repo: Repository) -> dict[str, Any]:
    branches, tags = repo.branches(), repo.tags()
    head = repo.head()
    refs: dict[str, dict[str, list[str]]] = {}
    for name, cid in branches.items():
        refs.setdefault(cid, {"branches": [], "tags": []})["branches"].append(name)
    for name, cid in tags.items():
        refs.setdefault(cid, {"branches": [], "tags": []})["tags"].append(name)
    commits, edges = [], []
    for cid in repo.commit_ids():
        commit = repo.load_commit(cid)
        commits.append({
            "id": cid, "message": commit.message, "author": commit.author, "created_at": commit.created_at,
            "parents": list(commit.parents), "content_id": commit.content_id,
            "files": len(commit.files), "bytes": sum(e.size for e in commit.files.values()),
            "preprocessing": commit.preprocessing, "quality": commit.quality,
            "generation": repo.generation(cid), **refs.get(cid, {"branches": [], "tags": []}),
        })
        edges += [{"source": p, "target": cid, "kind": "parent"} for p in commit.parents]
    experiments = []
    for rec in list_experiments(repo):
        test = (rec["metrics"] or {}).get("test")
        experiments.append({
            "id": rec["id"], "status": rec["status"], "created_at": rec["created_at"],
            "commit": rec["dataset"]["commit"], "parent": rec.get("parent_experiment"),
            "accuracy": test["accuracy"] if test else None, "macro_f1": test["macro_f1"] if test else None,
            "code_revision": rec["environment"].get("code_revision"),
        })
        edges.append({"source": rec["dataset"]["commit"], "target": rec["id"], "kind": "experiment"})
        if rec.get("parent_experiment"):
            edges.append({"source": rec["parent_experiment"], "target": rec["id"], "kind": "rerun"})
    return {
        "head": {"branch": head.branch, "commit": head.commit},
        "commits": commits, "experiments": experiments, "edges": edges,
    }


def list_ai(repo: Repository) -> list[dict[str, Any]]:
    out = []
    for path in sorted(repo.ai_dir.glob("*.json")):
        if SAFE_AI_NAME.match(path.name):
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append({"file": path.name, "task": data["task"], "created_at": data["created_at"],
                        "inputs": data["inputs"], "stats": data["analysis"]["stats"]})
    return out


def hostname(header: str) -> str:
    host = header.strip()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if ":" in host else host


class Handler(BaseHTTPRequestHandler):
    repo: Repository
    server_version = "datavault"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any, status: int = 200) -> None:
        self._send(status, json.dumps(value).encode(), "application/json")

    def do_GET(self) -> None:
        if hostname(self.headers.get("Host") or "") not in LOCAL_HOSTS and not self.server.allow_remote:  # type: ignore[attr-defined]
            return self._json({"error": "host not allowed"}, HTTPStatus.FORBIDDEN)
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            self._route(url.path, query)
        except DatavaultError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self._json({"error": "internal error"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route(self, path: str, query: dict[str, str]) -> None:
        repo = self.repo
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send(200, (STATIC / "app.js").read_bytes(), "text/javascript; charset=utf-8")
        if path == "/api/graph":
            return self._json(build_graph(repo))
        if path.startswith("/api/commit/"):
            cid = repo.resolve(path.rsplit("/", 1)[1])
            data = repo.load_commit(cid).to_dict()
            data["id"] = cid
            if data["quality"]:
                from ..quality import run_checks

                cfg = json.loads(repo.objects.read_bytes(data["quality"]["config_sha256"]))
                files = {p: repo.objects.read_bytes(e["sha256"]) for p, e in data["files"].items()}
                data["quality_report"] = run_checks(cfg, files).to_dict()
            return self._json(data)
        if path == "/api/diff":
            return self._json(diff_commits(repo, query.get("a", ""), query.get("b", "")))
        if path.startswith("/api/experiment/"):
            return self._json(get_experiment(repo, path.rsplit("/", 1)[1]))
        if path == "/api/compare":
            return self._json(compare_experiments(get_experiment(repo, query.get("a", "")),
                                                  get_experiment(repo, query.get("b", ""))))
        if path == "/api/verify":
            return self._json(verify(repo).to_dict())
        if path == "/api/ai":
            return self._json(list_ai(repo))
        if path.startswith("/api/ai/"):
            name = path.rsplit("/", 1)[1]
            if not SAFE_AI_NAME.match(name) or not (repo.ai_dir / name).is_file():
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return self._send(200, (repo.ai_dir / name).read_bytes(), "application/json")
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def make_server(repo: Repository, host: str, port: int, allow_remote: bool = False) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"repo": repo})
    server = ThreadingHTTPServer((host, port), handler)
    server.allow_remote = allow_remote or host not in ("127.0.0.1", "localhost", "::1")  # type: ignore[attr-defined]
    return server


def serve(repo: Repository, host: str, port: int, open_browser: bool = True) -> None:
    server = make_server(repo, host, port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"dashboard: {url}  (read-only, Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
