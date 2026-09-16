"""Read-only, crawlable W&B experiment portal with a persistent local cache.

Run with the repository virtualenv. Only destinations discovered in local experiment
metadata are exposed. W&B is the source of truth; histories are never sampled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import math
import mimetypes
import netrc
import os
from pathlib import Path, PurePosixPath
import re
import threading
import tempfile
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit

import requests

LOG = logging.getLogger("experiment-portal")
FIELDS = """name displayName group jobType state config summaryMetrics systemMetrics
createdAt updatedAt heartbeatAt historyLineCount eventsLineCount tags notes"""
LOCAL_FILES = {
    "eligibility.csv",
    "phase1_trials.csv",
    "phase1_ranking.csv",
    "phase1_leaderboard.csv",
    "phase2_predictions.csv",
    "leaderboard.csv",
    "metric_definitions.json",
    "failures.csv",
    "phase1_failures.csv",
    "phase2_failures.csv",
    "run_config.json",
    "data_manifest.json",
    "preparation.json",
    "exogenous_selection.csv",
    "phase2_leaderboard.json",
    "best_models.csv",
    "dashboard.json",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return quote(str(value), safe="")


def escaped(value):
    return html.escape(str(value), quote=True)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    temporary.write_text(dumps(value))
    temporary.replace(path)


def safe_path(root, name):
    """Reject traversal, absolute paths, and symlinks outside the cache."""
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid path")
    parts = PurePosixPath(name)
    if parts.is_absolute() or any(p in ("..", ".") for p in parts.parts):
        raise ValueError("Invalid path")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("Invalid path")
    return target


def table(rows, columns=None):
    if not rows:
        return '<p class="muted">No records yet.</p>'
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    head = "".join(f"<th>{escaped(k)}</th>" for k in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{escaped(dumps(row.get(k)) if isinstance(row.get(k), (dict, list)) else row.get(k, ''))}</td>"
            for k in columns
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def link(url, label):
    return f'<a href="{escaped(url)}">{escaped(label)}</a>'


def chart(series, title):
    """Render every finite observation, without downsampling."""
    cleaned = {}
    for name, points in series.items():
        valid = [
            (float(x), float(y))
            for x, y in points
            if isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and math.isfinite(x)
            and math.isfinite(y)
        ]
        if valid:
            cleaned[name] = valid
    if not cleaned:
        return ""
    all_points = [p for values in cleaned.values() for p in values]
    xmin, xmax = min(x for x, _ in all_points), max(x for x, _ in all_points)
    ymin, ymax = min(y for _, y in all_points), max(y for _, y in all_points)
    dx, dy = xmax - xmin or 1, ymax - ymin or 1
    colors = ["#177b66", "#cf6c40", "#6954b0", "#2480a0"]
    lines, legend = [], []
    for i, (name, points) in enumerate(cleaned.items()):
        color = colors[i % len(colors)]
        coords = " ".join(
            f"{70 + (x - xmin) / dx * 680:.3f},{270 - (y - ymin) / dy * 220:.3f}"
            for x, y in points
        )
        lines.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend.append(
            f'<span style="color:{color}">{escaped(name)} ({len(points)} points)</span>'
        )
    return (
        f'<h3>{escaped(title)}</h3><div class="chart"><svg viewBox="0 0 800 320" role="img" aria-label="{escaped(title)}">'
        '<path d="M70 40 V270 H760" fill="none" stroke="#b7c7bf"/>'
        f'<text x="4" y="55">{ymax:.5g}</text><text x="4" y="270">{ymin:.5g}</text>'
        f'<text x="70" y="300">{xmin:.5g}</text><text x="720" y="300">{xmax:.5g}</text>'
        + "".join(lines)
        + "</svg><p>"
        + " · ".join(legend)
        + "</p></div>"
    )


def page(title, body):
    css = (Path(__file__).parent / "style.css").read_text()
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escaped(title)} · Experiment Observatory</title><style>{css}</style>"
        "</head><body><header><a href='/'>◈ Experiment Observatory</a>"
        "<nav><a href='/api/experiments'>JSON API</a> <a href='/llms.txt'>AI reader guide</a>"
        "<a href='/healthz'>Sync status</a></nav></header><main>"
        f"<h1>{escaped(title)}</h1>{body}</main><footer>W&B source data · Read-only · "
        "Live snapshots and full-resolution run data</footer></body></html>"
    )


class Portal:
    def __init__(self, results, cache, interval=120):
        self.results = Path(results).resolve()
        self.cache = Path(cache).resolve()
        self.interval = interval
        self.cache.mkdir(parents=True, exist_ok=True)
        self.projects = {}
        self.errors = {}
        self.lock = threading.RLock()
        self.remote_slots = threading.BoundedSemaphore(6)
        self.key_locks = {}
        self.sync_progress = time.monotonic()
        self.discover()

    def discover(self):
        found = {}
        for path in sorted(self.results.glob("*/run_config.json")):
            try:
                config = json.loads(path.read_text())
                destination = config.get("wandb")
                if not destination:
                    continue
                entity, project = destination["entity"], destination["project"]
                if not all(
                    re.fullmatch(r"[A-Za-z0-9_.-]+", s) for s in (entity, project)
                ):
                    continue
                slug = f"{entity}--{project}"
                info = found.setdefault(
                    slug, dict(id=slug, entity=entity, project=project, experiments=[])
                )
                info["experiments"].append(
                    dict(
                        name=path.parent.name,
                        group=destination.get("group"),
                        status=config.get("status", "unknown"),
                    )
                )
            except (OSError, ValueError, KeyError):
                LOG.warning("Could not read experiment metadata: %s", path.name)
        # Include aggregate dashboards explicitly produced by this repository.
        for path in sorted(self.results.glob("*/dashboard.json")):
            try:
                d = json.loads(path.read_text())
                entity, project = d["entity"], d["project"]
                if not all(
                    re.fullmatch(r"[A-Za-z0-9_.-]+", s) for s in (entity, project)
                ):
                    continue
                slug = f"{entity}--{project}"
                found.setdefault(
                    slug,
                    dict(
                        id=slug,
                        entity=entity,
                        project=project,
                        experiments=[
                            dict(
                                name=path.parent.name,
                                group="completed-phase2",
                                status="completed",
                            )
                        ],
                    ),
                )
            except (OSError, ValueError, KeyError):
                continue
        with self.lock:
            self.projects = found

    def project(self, slug):
        with self.lock:
            if slug not in self.projects:
                raise KeyError("Unknown project")
            return dict(self.projects[slug])

    def gql(self, query, variables):
        base = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
        key = os.environ.get("WANDB_API_KEY")
        if not key:
            credential = netrc.netrc().authenticators(urlsplit(base).hostname)
            key = credential[2] if credential else None
        if not key:
            raise RuntimeError("W&B credentials unavailable")
        with self.remote_slots:
            for attempt in range(3):
                r = requests.post(
                    base + "/graphql",
                    auth=("api", key),
                    json=dict(query=query, variables=variables),
                    timeout=(10, 60),
                )
                if r.status_code in (429, 502, 503, 504) and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                r.raise_for_status()
                payload = r.json()
                if payload.get("errors"):
                    raise RuntimeError("W&B query failed")
                return payload["data"]
        raise RuntimeError("W&B unavailable")

    def cached(self, key, loader, ttl=None):
        digest = hashlib.sha256(key.encode()).hexdigest()
        path = self.cache / "objects" / f"{digest}.json"
        with self.lock:
            lock = self.key_locks.setdefault(digest, threading.Lock())
        with lock:
            if path.exists() and time.time() - path.stat().st_mtime < (
                ttl or self.interval
            ):
                return json.loads(path.read_text())
            data = loader()
            atomic_json(path, data)
            return data

    def refresh(self, slug):
        p = self.project(slug)
        query = """query($entity:String!,$project:String!,$after:String){
          project(name:$project,entityName:$entity){runCount runs(first:100,after:$after){
            edges{node{FIELDS}} pageInfo{endCursor hasNextPage}}}}
        """.replace("FIELDS", FIELDS)
        runs, cursor, seen = [], None, set()
        while True:
            d = self.gql(
                query, dict(entity=p["entity"], project=p["project"], after=cursor)
            )["project"]
            self.sync_progress = time.monotonic()
            if d is None:
                raise RuntimeError("Source project unavailable")
            for edge in d["runs"]["edges"]:
                row = edge["node"]
                if row["name"] not in seen:
                    seen.add(row["name"])
                    runs.append(row)
            info = d["runs"]["pageInfo"]
            if not info["hasNextPage"]:
                break
            if info["endCursor"] == cursor:
                raise RuntimeError("W&B pagination stalled")
            cursor = info["endCursor"]
        snapshot = dict(
            **p,
            fetched_at=now(),
            source_run_count=d["runCount"],
            fetched_run_count=len(runs),
            runs=runs,
        )
        atomic_json(self.cache / "projects" / f"{slug}.json", snapshot)
        with self.lock:
            self.errors.pop(slug, None)
        LOG.info("Synced %s: %s runs", slug, len(runs))

    def snapshot(self, slug):
        self.project(slug)
        path = self.cache / "projects" / f"{slug}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def catalog(self):
        rows = []
        for slug in sorted(self.projects):
            p = self.project(slug)
            snap = self.snapshot(slug)
            p.update(
                url=f"/projects/{slug}",
                runs_url=f"/api/projects/{slug}/runs",
                snapshot_url=f"/api/projects/{slug}/snapshot",
            )
            if snap:
                p.update(
                    {
                        k: snap[k]
                        for k in ("fetched_at", "source_run_count", "fetched_run_count")
                    }
                )
                age = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(snap["fetched_at"])
                ).total_seconds()
                p["snapshot_age_seconds"] = round(age, 1)
                p["count_matches_source"] = (
                    snap["source_run_count"] == snap["fetched_run_count"]
                )
                p["sync_status"] = (
                    "stale"
                    if self.errors.get(slug) or age > self.interval * 3
                    else "current"
                    if p["count_matches_source"]
                    else "updating"
                )
            else:
                p["sync_status"] = "pending"
            if slug in self.errors:
                p["sync_error"] = self.errors[slug]
            rows.append(p)
        return dict(
            generated_at=now(),
            refresh_seconds=self.interval,
            sync_worker_healthy=time.monotonic() - self.sync_progress
            < max(900, self.interval * 4),
            data_policy="W&B original config/summary strings; full unsampled run histories; files and artifacts fetched on demand and cached. Not a complete offline backup.",
            projects=rows,
        )

    def analysis_export(self):
        """Provide a directly attachable report without rounding source CSV values."""
        lines = [
            "# NeuralForecast experiment results",
            f"Generated: {now()}",
            "",
            "Source: local experiment outputs plus W&B run inventory.",
            "Running experiments are provisional. Compare models only within matching evaluation windows and scales.",
            "This report contains result tables and experiment context, not every training step.",
            "All run metadata is available at each /api/projects/{id}/snapshot; follow run history/files/artifacts for full data.",
        ]
        for p in self.catalog()["projects"]:
            lines += [
                "",
                f"## {p['entity']}/{p['project']}",
                f"W&B snapshot: {p.get('fetched_at', 'pending')}",
                f"Runs: {p.get('fetched_run_count', 0)} / {p.get('source_run_count', 'unknown')}",
                f"Full metadata: {p['snapshot_url']}",
            ]
            for e in p["experiments"]:
                lines += [
                    "",
                    f"### {e['name']}",
                    f"Status: {e['status']}",
                    f"Group: {e['group']}",
                ]
                for name in (
                    "leaderboard.csv",
                    "phase1_leaderboard.csv",
                    "metric_definitions.json",
                    "run_config.json",
                ):
                    path = self.results / e["name"] / name
                    if path.is_file():
                        lines += [
                            "",
                            f"#### {name}",
                            "```" + path.suffix[1:],
                            path.read_text(),
                            "```",
                        ]
        return "\n".join(lines)

    def watch(self):
        while True:
            self.discover()

            def update(slug):
                try:
                    self.refresh(slug)
                except Exception as exc:
                    LOG.warning("Sync failed for %s: %s", slug, type(exc).__name__)
                    with self.lock:
                        self.errors[slug] = dict(at=now(), type=type(exc).__name__)

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(update, list(self.projects)))
            self.sync_progress = time.monotonic()
            time.sleep(self.interval)

    def run(self, slug, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise KeyError("Invalid run")
        p = self.project(slug)

        def fetch():
            q = "query($entity:String!,$project:String!,$run:String!){project(name:$project,entityName:$entity){run(name:$run){FIELDS}}}".replace(
                "FIELDS", FIELDS
            )
            row = self.gql(
                q, dict(entity=p["entity"], project=p["project"], run=run_id)
            )["project"]["run"]
            if row is None:
                raise KeyError("Unknown run")
            return dict(fetched_at=now(), run=row)

        return self.cached(f"run:{slug}:{run_id}", fetch)

    def sdk_run(self, slug, run_id):
        import wandb

        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise KeyError("Invalid run")
        p = self.project(slug)
        return wandb.Api(timeout=60).run(f"{p['entity']}/{p['project']}/{run_id}")

    def history(self, slug, run_id, system=False):
        def fetch():
            meta = self.run(slug, run_id)["run"]
            with self.remote_slots:
                run = self.sdk_run(slug, run_id)
                if system:
                    expected = meta.get("eventsLineCount") or 0
                    rows = run.history(
                        samples=max(expected + 1, 1), stream="system", pandas=False
                    )
                else:
                    expected = meta.get("historyLineCount")
                    rows = list(run.scan_history(page_size=1000, use_cache=False))
            return dict(
                fetched_at=now(),
                sampled=False if not system else len(rows) < expected,
                stream="system" if system else "default",
                source_line_count=expected,
                returned_rows=len(rows),
                count_matches_source=len(rows) == expected,
                records=rows,
            )

        return self.cached(f"history:{slug}:{run_id}:{system}", fetch)

    def files(self, slug, run_id):
        def fetch():
            with self.remote_slots:
                run = self.sdk_run(slug, run_id)
                return dict(
                    fetched_at=now(),
                    files=[
                        dict(
                            name=f.name,
                            size=f.size,
                            md5=f.md5,
                            url=f"/api/projects/{slug}/runs/{run_id}/file?name={encoded(f.name)}",
                        )
                        for f in run.files(per_page=100)
                    ],
                )

        return self.cached(f"files:{slug}:{run_id}", fetch)

    def file(self, slug, run_id, name):
        if not any(f["name"] == name for f in self.files(slug, run_id)["files"]):
            raise KeyError("Unknown file")
        root = self.cache / "downloads" / slug / run_id
        target = safe_path(root, name)
        if target.exists() and time.time() - target.stat().st_mtime < self.interval:
            return target
        root.mkdir(parents=True, exist_ok=True)
        with self.remote_slots, tempfile.TemporaryDirectory(dir=self.cache) as staging:
            run = self.sdk_run(slug, run_id)
            handle = run.file(name).download(root=staging, replace=True)
            handle.close()
            target.parent.mkdir(parents=True, exist_ok=True)
            safe_path(Path(staging), name).replace(target)
        return target

    def artifacts(self, slug, run_id):
        def fetch():
            with self.remote_slots:
                run = self.sdk_run(slug, run_id)
                data = []
                for kind, items in (
                    ("output", run.logged_artifacts()),
                    ("input", run.used_artifacts()),
                ):
                    for a in items:
                        data.append(
                            dict(
                                name=a.name,
                                qualified_name=a.qualified_name,
                                type=a.type,
                                id=a.id,
                                version=a.version,
                                digest=a.digest,
                                aliases=a.aliases,
                                metadata=a.metadata,
                                direction=kind,
                                url=f"/api/projects/{slug}/runs/{run_id}/artifact?name={encoded(a.name)}",
                            )
                        )
            return dict(fetched_at=now(), artifacts=data)

        return self.cached(f"artifacts:{slug}:{run_id}", fetch)

    def project_artifacts(self, slug, artifact_type=None, collection=None):
        """Expose all historical versions, including those absent from run lists."""
        p = self.project(slug)
        for value in (artifact_type, collection):
            if value is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError("Invalid artifact name")

        def fetch():
            import wandb

            with self.remote_slots:
                api = wandb.Api(timeout=60)
                destination = f"{p['entity']}/{p['project']}"
                if artifact_type is None:
                    return dict(
                        types=[
                            dict(
                                name=t.name,
                                url=f"/api/projects/{slug}/artifact-collections?type={encoded(t.name)}",
                            )
                            for t in api.artifact_types(project=destination)
                        ]
                    )
                if collection is None:
                    kind = api.artifact_type(artifact_type, project=destination)
                    return dict(
                        collections=[
                            dict(
                                name=c.name,
                                url=f"/api/projects/{slug}/artifact-versions?type={encoded(artifact_type)}&collection={encoded(c.name)}",
                            )
                            for c in kind.collections(per_page=100)
                        ]
                    )
                return dict(
                    versions=[
                        dict(
                            name=a.name,
                            digest=a.digest,
                            version=a.version,
                            metadata=a.metadata,
                            aliases=a.aliases,
                            url=f"/api/projects/{slug}/artifact?name={encoded(a.name)}",
                        )
                        for a in api.artifact_versions(
                            artifact_type, f"{destination}/{collection}", per_page=100
                        )
                    ]
                )

        return self.cached(
            f"project-artifacts:{slug}:{artifact_type}:{collection}", fetch
        )

    def artifact(self, slug, run_id, name, filename=None):
        import wandb

        p = self.project(slug)
        if run_id is not None:
            allowed = self.artifacts(slug, run_id)["artifacts"]
            matches = [a for a in allowed if a["name"] == name]
            if not matches:
                raise KeyError("Unknown artifact")
            qualified = matches[0].get(
                "qualified_name", f"{p['entity']}/{p['project']}/{name}"
            )
            base = f"/api/projects/{slug}/runs/{run_id}/artifact"
        else:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+:v[0-9]+", name):
                raise ValueError("An explicit artifact version is required")
            qualified = f"{p['entity']}/{p['project']}/{name}"
            base = f"/api/projects/{slug}/artifact"
        with self.remote_slots:
            a = wandb.Api(timeout=60).artifact(qualified)
            if filename is None:
                return dict(
                    name=a.name,
                    digest=a.digest,
                    metadata=a.metadata,
                    files=[
                        dict(
                            name=k,
                            size=v.size,
                            digest=v.digest,
                            url=f"{base}?name={encoded(name)}&file={encoded(k)}",
                        )
                        for k, v in a.manifest.entries.items()
                    ],
                )
            if filename not in a.manifest.entries:
                raise KeyError("Unknown artifact file")
            root = self.cache / "artifacts" / hashlib.sha256(a.id.encode()).hexdigest()
            target = safe_path(root, filename)
            if not target.exists():
                root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=self.cache) as staging:
                    a.get_entry(filename).download(root=staging)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    safe_path(Path(staging), filename).replace(target)
            return target

    def local_files(self, slug):
        p = self.project(slug)
        return [
            dict(
                experiment=e["name"],
                name=name,
                url=f"/api/projects/{slug}/local?experiment={encoded(e['name'])}&name={encoded(name)}",
            )
            for e in p["experiments"]
            for name in sorted(LOCAL_FILES)
            if (self.results / e["name"] / name).is_file()
        ]

    def local_file(self, slug, experiment, name):
        if not any(
            x["experiment"] == experiment and x["name"] == name
            for x in self.local_files(slug)
        ):
            raise KeyError("Unknown result file")
        return safe_path(self.results, f"{experiment}/{name}")


class Handler(BaseHTTPRequestHandler):
    server_version = "ExperimentPortal/1"

    def log_message(self, fmt, *args):
        # Do not log query strings; only paths and response codes are useful here.
        LOG.info("HTTP %s %s", self.command, urlsplit(self.path).path)

    def reply(self, data, content_type="application/json; charset=utf-8", status=200):
        if not isinstance(data, bytes):
            data = (data if isinstance(data, str) else dumps(data)).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_file(self, path):
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".json", ".jsonl"):
            mime = "application/json"
        # Never execute uploaded HTML/SVG/JS in the portal's origin.
        if mime in (
            "text/html",
            "image/svg+xml",
            "text/javascript",
            "application/javascript",
        ):
            mime = "text/plain"
        with path.open("rb") as handle:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mime + "; charset=utf-8" if mime.startswith("text/") else mime,
            )
            self.send_header("Content-Length", str(os.fstat(handle.fileno()).st_size))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; sandbox")
            self.end_headers()
            if self.command != "HEAD":
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self.dispatch()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (KeyError, ValueError, FileNotFoundError):
            self.reply(dict(error="Not found or invalid request"), status=404)
        except Exception as exc:
            LOG.warning("Request failed: %s", type(exc).__name__)
            self.reply(
                dict(
                    error="Source data temporarily unavailable; retry shortly",
                    type=type(exc).__name__,
                ),
                status=502,
            )

    def dispatch(self):
        portal = self.server.portal
        url = urlsplit(self.path)
        path = [unquote(p) for p in url.path.strip("/").split("/") if p]
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/robots.txt":
            return self.reply("User-agent: *\nAllow: /\n", "text/plain")
        if url.path == "/export/analysis.md":
            return self.reply(portal.analysis_export(), "text/markdown; charset=utf-8")
        if url.path == "/llms.txt":
            catalog = portal.catalog()
            lines = [
                "# NeuralForecast Experiment Observatory",
                "",
                "Read /api/experiments first. All URLs are relative to this site's origin.",
                "Each project contains all W&B runs, including prior groups and failed runs.",
                "The project page shows local result tables separately from W&B source data.",
                "Use /api/projects/{id}/snapshot for every run's original metadata, config, summary and system summary.",
                "W&B config and summaryMetrics are original JSON strings: parse them before analysis.",
                "Use /api/projects/{id}/runs?page=1&limit=100 for pagination; follow next_url.",
                "For each run use /api/projects/{id}/runs/{run}/history (unsampled JSON), /history.jsonl, /history.csv, /system, /files, /artifacts.",
                "Follow file/artifact URLs to read original tables, predictions and all attachments.",
                "Use /api/projects/{id}/artifact-types and follow collection/version links for all historical artifact versions.",
                "Data is refreshed automatically. Check fetched_at, source counts and /healthz.",
                "Do not compare raw errors across different targets/scales/evaluation windows.",
                "Running results are provisional. Missing results are not zero.",
                "Files and histories are fetched on demand. This site is not an independent full offline backup.",
                "If your web reader cannot access this temporary domain, download /export/analysis.md in a browser and attach it to your conversation.",
                "",
            ]
            lines += [
                f"- {p['project']}: {p['url']} | {p['snapshot_url']}"
                for p in catalog["projects"]
            ]
            return self.reply("\n".join(lines), "text/plain; charset=utf-8")
        if url.path in ("/healthz", "/api/experiments"):
            return self.reply(portal.catalog())
        if not path:
            catalog = portal.catalog()
            cards = ""
            for p in catalog["projects"]:
                statuses = ", ".join(e["status"] for e in p["experiments"])
                cards += (
                    f'<article><span class="eyebrow">{escaped(p["entity"])}</span>'
                    f'<h2>{link(p["url"], p["project"])}</h2><p class="count">{p.get("fetched_run_count", "…")} <small>runs</small></p>'
                    f'<p>{escaped(statuses)}</p><p class="muted">Snapshot: {escaped(p.get("fetched_at", "initial sync in progress"))}</p>'
                    f"<p>{link(p['snapshot_url'], 'Full run snapshot JSON')}</p></article>"
                )
            body = (
                "<p class='intro'>Every experiment, one readable source. Model results, training history, "
                "forecasts and original W&B files — accessible without a W&B login.</p>"
                "<p>"
                + link("/export/analysis.md", "Download analysis document for ChatGPT")
                + "</p>"
                f"<p class='notice'>Automatic refresh every {portal.interval}s after each sync cycle. "
                "Run histories and attachments load from W&B on demand. This is a live portal, not a full offline backup.</p>"
                f"<section class='cards'>{cards}</section>"
            )
            return self.reply(
                page("Experiment overview", body), "text/html; charset=utf-8"
            )
        if len(path) >= 2 and path[0] == "projects":
            slug = path[1]
            p = portal.project(slug)
            if len(path) == 2:
                body = (
                    f"<p>{link('/', '← All experiments')} · {link(f'/api/projects/{slug}/snapshot', 'Full W&B snapshot JSON')} · "
                    f"{link(f'/api/projects/{slug}/runs', 'Paginated runs JSON')}</p>"
                )
                body += (
                    "<p>"
                    + link(
                        f"/api/projects/{slug}/artifact-types",
                        "All artifact collections and historical versions",
                    )
                    + "</p>"
                )
                for e in p["experiments"]:
                    body += f"<h2>{escaped(e['name'])}</h2><p>Status: {escaped(e['status'])} · Group: {escaped(e['group'])}</p>"
                    for name in ("leaderboard.csv", "phase1_leaderboard.csv"):
                        f = portal.results / e["name"] / name
                        if f.is_file():
                            with f.open() as handle:
                                rows = list(csv.DictReader(handle))
                            body += (
                                f"<h3>{escaped(name)} — local result file</h3>"
                                + table(rows)
                            )
                body += (
                    "<h2>Result downloads</h2><ul>"
                    + "".join(
                        f"<li>{link(f['url'], f['experiment'] + '/' + f['name'])}</li>"
                        for f in portal.local_files(slug)
                    )
                    + "</ul>"
                )
                snap = portal.snapshot(slug)
                if snap:
                    rows = self.filtered_runs(snap["runs"], query)
                    num = max(1, int(query.get("page", 1)))
                    body += f'<h2>W&B runs ({len(rows)})</h2><p class="muted">Snapshot {escaped(snap["fetched_at"])}</p>'
                    body += (
                        '<form method="get"><input name="q" placeholder="Search model, run or group" value="'
                        + escaped(query.get("q", ""))
                        + '"><button>Search</button></form>'
                    )
                    body += (
                        '<ul class="runs">'
                        + "".join(
                            f"<li>{link('/projects/{}/runs/{}'.format(slug, r['name']), r['displayName'] or r['name'])} <span>{escaped(r['state'])} · {escaped(r['jobType'])} · {escaped(r['group'])}</span></li>"
                            for r in rows[(num - 1) * 100 : num * 100]
                        )
                        + "</ul>"
                    )
                    if num > 1:
                        body += link(
                            f"?page={num - 1}&q={encoded(query.get('q', ''))}",
                            "← Previous ",
                        )
                    if num * 100 < len(rows):
                        body += link(
                            f"?page={num + 1}&q={encoded(query.get('q', ''))}",
                            "Next 100 →",
                        )
                else:
                    body += "<p>Initial W&B synchronization in progress. Reload shortly.</p>"
                return self.reply(page(p["project"], body), "text/html; charset=utf-8")
            if len(path) in (4, 5) and path[2] == "runs":
                run_id = path[3]
                data = portal.run(slug, run_id)
                run = data["run"]
                base = f"/api/projects/{slug}/runs/{run_id}"
                body = f"<p>{link(f'/projects/{slug}', '← Project')} · {escaped(run['state'])} · {escaped(run['group'])}</p>"
                summary = json.loads(run["summaryMetrics"] or "{}")
                actual, prediction = (
                    summary.get("forecast/actual"),
                    summary.get("forecast/prediction"),
                )
                if isinstance(actual, list) and isinstance(prediction, list):
                    body += chart(
                        {
                            "Actual": list(enumerate(actual, 1)),
                            "Prediction": list(enumerate(prediction, 1)),
                        },
                        "Forecast vs actual",
                    )
                body += (
                    "<p>"
                    + link(
                        f"/projects/{slug}/runs/{run_id}/charts",
                        "View full-resolution training charts",
                    )
                    + "</p>"
                )
                if len(path) == 5:
                    if path[4] != "charts":
                        raise KeyError("Unknown route")
                    records = portal.history(slug, run_id)["records"]
                    keys = sorted(
                        {
                            k
                            for row in records
                            for k in row
                            if k.startswith(("train/", "validation/"))
                            and k != "train/global_step"
                        }
                    )
                    for key in keys:
                        body += chart(
                            {
                                key: [
                                    (
                                        r.get("train/global_step") or r.get("_step"),
                                        r.get(key),
                                    )
                                    for r in records
                                ]
                            },
                            key,
                        )
                    return self.reply(
                        page(run["displayName"] or run_id, body),
                        "text/html; charset=utf-8",
                    )
                body += (
                    '<nav class="downloads">'
                    + " ".join(
                        link(base + suffix, title)
                        for suffix, title in (
                            ("", "Original run JSON"),
                            ("/history", "Full history JSON"),
                            ("/history.csv", "History CSV"),
                            ("/history.jsonl", "History JSONL"),
                            ("/system", "System metrics"),
                            ("/files", "All run files"),
                            ("/artifacts", "Artifacts"),
                        )
                    )
                    + "</nav>"
                )
                for key in ("summaryMetrics", "config", "systemMetrics"):
                    raw = run.get(key)
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    body += f"<h2>{escaped(key)}</h2><pre>{escaped(json.dumps(parsed, indent=2, ensure_ascii=False))}</pre>"
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if (
                                isinstance(v, dict)
                                and v.get("_type") == "table-file"
                                and v.get("path")
                            ):
                                body += f"<p>{link(base + '/file?name=' + encoded(v['path']), k + ' — original table JSON')}</p>"
                return self.reply(
                    page(run["displayName"] or run_id, body), "text/html; charset=utf-8"
                )
        if len(path) >= 3 and path[:2] == ["api", "projects"]:
            slug = path[2]
            portal.project(slug)
            if len(path) == 4:
                action = path[3]
                if action == "artifact-types":
                    return self.reply(portal.project_artifacts(slug))
                if action == "artifact-collections":
                    return self.reply(portal.project_artifacts(slug, query["type"]))
                if action == "artifact-versions":
                    return self.reply(
                        portal.project_artifacts(
                            slug, query["type"], query["collection"]
                        )
                    )
                if action == "artifact":
                    data = portal.artifact(slug, None, query["name"], query.get("file"))
                    return (
                        self.send_file(data)
                        if isinstance(data, Path)
                        else self.reply(data)
                    )
                if action == "local":
                    return self.send_file(
                        portal.local_file(slug, query["experiment"], query["name"])
                    )
                snap = portal.snapshot(slug)
                if not snap:
                    return self.reply(dict(status="initial_sync"), status=503)
                if action == "snapshot":
                    return self.reply(snap)
                if action == "runs":
                    rows = self.filtered_runs(snap["runs"], query)
                    num = max(1, int(query.get("page", 1)))
                    limit = min(500, max(1, int(query.get("limit", 100))))
                    page_rows = rows[(num - 1) * limit : num * limit]
                    for row in page_rows:
                        row["url"] = f"/api/projects/{slug}/runs/{row['name']}"
                    return self.reply(
                        dict(
                            fetched_at=snap["fetched_at"],
                            total=len(rows),
                            page=num,
                            runs=page_rows,
                            next_url=(
                                f"/api/projects/{slug}/runs?page={num + 1}&limit={limit}&q={encoded(query.get('q', ''))}"
                                if num * limit < len(rows)
                                else None
                            ),
                        )
                    )
            if len(path) >= 5 and path[3] == "runs":
                run_id = path[4]
                action = path[5] if len(path) == 6 else ""
                if len(path) > 6:
                    raise KeyError("Unknown route")
                if not action:
                    return self.reply(portal.run(slug, run_id))
                if action in ("history", "history.jsonl", "history.csv", "system"):
                    data = portal.history(slug, run_id, system=action == "system")
                    if action == "history.jsonl":
                        return self.reply(
                            "\n".join(dumps(r) for r in data["records"]) + "\n",
                            "application/x-ndjson",
                        )
                    if action == "history.csv":
                        out = io.StringIO()
                        cols = list(
                            dict.fromkeys(k for r in data["records"] for k in r)
                        )
                        writer = csv.DictWriter(out, fieldnames=cols)
                        writer.writeheader()
                        for row in data["records"]:
                            writer.writerow(
                                {
                                    k: dumps(v) if isinstance(v, (dict, list)) else v
                                    for k, v in row.items()
                                }
                            )
                        return self.reply(out.getvalue(), "text/csv; charset=utf-8")
                    return self.reply(data)
                if action == "files":
                    return self.reply(portal.files(slug, run_id))
                if action == "file":
                    return self.send_file(portal.file(slug, run_id, query["name"]))
                if action == "artifacts":
                    return self.reply(portal.artifacts(slug, run_id))
                if action == "artifact":
                    data = portal.artifact(
                        slug, run_id, query["name"], query.get("file")
                    )
                    return (
                        self.send_file(data)
                        if isinstance(data, Path)
                        else self.reply(data)
                    )
        raise KeyError("Unknown route")

    @staticmethod
    def filtered_runs(rows, query):
        text = query.get("q", "").lower()
        return [
            r
            for r in rows
            if not text
            or text
            in " ".join(
                str(r.get(k, ""))
                for k in ("name", "displayName", "group", "jobType", "state")
            ).lower()
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=Path("results/experiment-portal"))
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    portal = Portal(args.results, args.cache, args.interval)
    threading.Thread(target=portal.watch, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.portal = portal
    LOG.info("Serving on 127.0.0.1:%s", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
