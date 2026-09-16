"""Regression checks for lossless pagination and safe public result access."""

import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import requests


@pytest.fixture
def module():
    path = Path(__file__).parents[1] / "scripts/experiment_portal/server.py"
    spec = importlib.util.spec_from_file_location("experiment_portal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def portal(module, tmp_path):
    results = tmp_path / "results"
    experiment = results / "experiment"
    experiment.mkdir(parents=True)
    (experiment / "run_config.json").write_text(
        json.dumps(
            {
                "status": "running",
                "wandb": {"entity": "team", "project": "p", "group": "g"},
            }
        )
    )
    (experiment / "leaderboard.csv").write_text(
        "candidate,rmse\nNaive,1.123456789012345\n"
    )
    (experiment / "private.env").write_text("PRIVATE_KEY=not-public")
    return module.Portal(results, tmp_path / "cache")


def test_all_pages_keep_original_json_and_precision(portal, monkeypatch):
    original = '{"rmse":1.123456789012345,"nested":{"x":[1,2]}}'
    pages = [
        {
            "runCount": 2,
            "runs": {
                "edges": [{"node": {"name": "a", "summaryMetrics": original}}],
                "pageInfo": {"endCursor": "next", "hasNextPage": True},
            },
        },
        {
            "runCount": 2,
            "runs": {
                "edges": [{"node": {"name": "b", "summaryMetrics": original}}],
                "pageInfo": {"endCursor": "end", "hasNextPage": False},
            },
        },
    ]
    cursors = []

    def query(q, variables):
        cursors.append(variables["after"])
        return {"project": pages.pop(0)}

    monkeypatch.setattr(portal, "gql", query)
    portal.refresh("team--p")
    data = portal.snapshot("team--p")
    assert cursors == [None, "next"]
    assert data["source_run_count"] == data["fetched_run_count"] == 2
    assert [r["name"] for r in data["runs"]] == ["a", "b"]
    assert all(r["summaryMetrics"] == original for r in data["runs"])


def test_failed_sync_preserves_previous_snapshot(module, portal, monkeypatch):
    path = portal.cache / "projects/team--p.json"
    module.atomic_json(path, {"runs": [{"name": "old"}]})

    def fail(*args):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(portal, "gql", fail)
    with pytest.raises(RuntimeError):
        portal.refresh("team--p")
    assert portal.snapshot("team--p")["runs"] == [{"name": "old"}]


@pytest.mark.parametrize(
    "name", ["../key", "/etc/passwd", "x/../../key", "a\\b", "\x00"]
)
def test_no_path_escape(module, tmp_path, name):
    with pytest.raises(ValueError):
        module.safe_path(tmp_path, name)


def test_symlinks_and_unlisted_files_are_not_published(module, portal, tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("secret")
    (portal.cache / "link").symlink_to(secret)
    with pytest.raises(ValueError):
        module.safe_path(portal.cache, "link")
    with pytest.raises(KeyError):
        portal.local_file("team--p", "experiment", "private.env")
    with pytest.raises(KeyError):
        portal.local_file("team--p", "../experiment", "leaderboard.csv")
    with pytest.raises(KeyError):
        portal.project("other--private")
    path = portal.local_file("team--p", "experiment", "leaderboard.csv")
    assert "1.123456789012345" in path.read_text()


def test_history_is_unsampled_and_keeps_sparse_keys(portal, monkeypatch):
    rows = [{"_step": i, "metric" if i % 2 else "other": i / 7} for i in range(1205)]
    calls = []

    def scan_history(**kwargs):
        calls.append(kwargs)
        return iter(rows)

    monkeypatch.setattr(
        portal, "sdk_run", lambda *a: SimpleNamespace(scan_history=scan_history)
    )
    monkeypatch.setattr(
        portal, "run", lambda *a: {"run": {"historyLineCount": len(rows)}}
    )
    data = portal.history("team--p", "run")
    assert data["records"] == rows
    assert data["sampled"] is False
    assert data["count_matches_source"] is True
    assert calls == [{"page_size": 1000, "use_cache": False}]


def test_system_history_reports_upstream_truncation(portal, monkeypatch):
    monkeypatch.setattr(
        portal, "sdk_run", lambda *a: SimpleNamespace(history=lambda **k: [{"x": 1}])
    )
    monkeypatch.setattr(portal, "run", lambda *a: {"run": {"eventsLineCount": 20}})
    data = portal.history("team--p", "run", system=True)
    assert data["sampled"] is True
    assert data["count_matches_source"] is False


def test_http_pages_are_crawlable_and_paginate(module, portal):
    rows = [
        {
            "name": str(i),
            "displayName": "<script>alert(1)</script>",
            "group": "g",
            "jobType": "phase2",
            "state": "finished",
        }
        for i in range(105)
    ]
    module.atomic_json(
        portal.cache / "projects/team--p.json",
        {
            "fetched_at": "2026-09-16T00:00:00+00:00",
            "source_run_count": 105,
            "fetched_run_count": 105,
            "runs": rows,
        },
    )
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    server.portal = portal
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        r = requests.get(base + "/projects/team--p", timeout=5)
        assert r.status_code == 200
        assert "Naive" in r.text
        assert "&lt;script&gt;" in r.text
        assert "<script>alert" not in r.text
        data = requests.get(base + "/api/projects/team--p/runs", timeout=5).json()
        assert len(data["runs"]) == 100
        second = requests.get(base + data["next_url"], timeout=5).json()
        assert len(second["runs"]) == 5
        assert second["next_url"] is None
        assert requests.get(base + "/.netrc", timeout=5).status_code == 404
        assert (
            requests.get(
                base + "/api/projects/unlisted/snapshot", timeout=5
            ).status_code
            == 404
        )
        guide = requests.get(base + "/llms.txt", timeout=5)
        assert "/api/projects/team--p/snapshot" in guide.text
        report = requests.get(base + "/export/analysis.md", timeout=5)
        assert report.status_code == 200
        assert "Naive,1.123456789012345" in report.text
        assert "PRIVATE_KEY" not in report.text
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_chart_preserves_all_finite_points_and_escapes_labels(module):
    plot = module.chart({"<bad>": list(enumerate(range(1500)))}, "<title>")
    assert "1500 points" in plot
    assert "&lt;bad&gt;" in plot
    assert "&lt;title&gt;" in plot
    assert len(plot.split('points="')[1].split('"')[0].split()) == 1500


def test_historical_artifact_versions_remain_available(portal, monkeypatch):
    import wandb

    calls = []

    def versions(kind, destination, per_page):
        calls.append((kind, destination, per_page))
        return [
            SimpleNamespace(
                name=f"results:v{i}",
                digest=f"digest{i}",
                version=f"v{i}",
                metadata={},
                aliases=[],
            )
            for i in range(3)
        ]

    monkeypatch.setattr(
        wandb, "Api", lambda **k: SimpleNamespace(artifact_versions=versions)
    )
    data = portal.project_artifacts("team--p", "benchmark-results", "results")
    assert [v["version"] for v in data["versions"]] == ["v0", "v1", "v2"]
    assert calls == [("benchmark-results", "team/p/results", 100)]
    for name in ("../../private:v0", "other/project/results:v0", "results:latest"):
        with pytest.raises(ValueError):
            portal.artifact("team--p", None, name)


def test_health_reports_stalled_sync_worker(portal, module, monkeypatch):
    portal.sync_progress = 100
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000)
    assert portal.catalog()["sync_worker_healthy"] is False
    portal.sync_progress = 990
    assert portal.catalog()["sync_worker_healthy"] is True
