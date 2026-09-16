import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def watchdog():
    path = Path(__file__).parents[1] / "scripts/experiment_portal/watchdog.py"
    spec = importlib.util.spec_from_file_location("portal_watchdog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("origin_down", [False, True])
def test_recovers_correct_service_after_three_failures(watchdog, origin_down):
    recovered = []

    def probe(url):
        return not origin_down and url.startswith("http://127.")

    state = {}
    for _ in range(2):
        state = watchdog.check(
            state,
            "https://test.trycloudflare.com",
            probe,
            lambda unit: True,
            recovered.append,
        )
    assert recovered == []
    state = watchdog.check(
        state,
        "https://test.trycloudflare.com",
        probe,
        lambda unit: True,
        recovered.append,
    )
    assert recovered == [watchdog.ORIGIN if origin_down else watchdog.TUNNEL]
    assert state["recovered"] == recovered[0]


def test_healthy_probe_resets_failures(watchdog):
    state = watchdog.check(
        {"origin_failures": 2, "tunnel_failures": 2},
        "https://test.trycloudflare.com",
        lambda url: True,
        lambda unit: True,
        lambda unit: pytest.fail("unexpected restart"),
    )
    assert state["status"] == "healthy"
    assert state["origin_failures"] == state["tunnel_failures"] == 0


def test_deliberately_stopped_origin_is_not_restarted(watchdog):
    state = watchdog.check(
        {},
        None,
        lambda url: pytest.fail("unexpected probe"),
        lambda unit: False,
        lambda unit: pytest.fail("unexpected restart"),
    )
    assert state["status"] == "origin_stopped"


def test_new_tunnel_does_not_inherit_old_failures(watchdog):
    state = watchdog.check(
        {"public_url": "https://old.trycloudflare.com", "tunnel_failures": 2},
        "https://new.trycloudflare.com",
        lambda url: url.startswith("http://127."),
        lambda unit: True,
        lambda unit: pytest.fail("unexpected restart"),
    )
    assert state["tunnel_failures"] == 1
