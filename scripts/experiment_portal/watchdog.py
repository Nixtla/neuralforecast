"""Check origin and public tunnel; recover only after repeated failures."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

import requests


ORIGIN = "neuralforecast-portal.service"
TUNNEL = "neuralforecast-portal-tunnel.service"


def healthy(url):
    try:
        response = requests.get(url.rstrip("/") + "/healthz", timeout=(5, 15))
        response.raise_for_status()
        data = response.json()
        # A W&B outage is not a stalled worker: preserve snapshots and retry normally.
        return data.get("sync_worker_healthy") is True and "projects" in data
    except (requests.RequestException, ValueError):
        return False


def active(unit):
    return (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit], check=False
        ).returncode
        == 0
    )


def restart(unit):
    subprocess.run(["systemctl", "--user", "restart", unit], check=True, timeout=45)


def check(state, public_url, probe=healthy, is_active=active, recover=restart):
    """Debounce failures; never restart the tunnel because the origin is unhealthy."""
    result = dict(state)
    if result.get("public_url") != public_url:
        result["tunnel_failures"] = 0
    result.update(
        public_url=public_url, checked_at=datetime.now(timezone.utc).isoformat()
    )
    result["recovered"] = None
    if not is_active(ORIGIN):
        result.update(origin_failures=0, tunnel_failures=0, status="origin_stopped")
        return result  # Respect deliberate systemctl stop.
    origin_ok = probe("http://127.0.0.1:18081")
    result["origin_failures"] = 0 if origin_ok else result.get("origin_failures", 0) + 1
    if not origin_ok:
        result.update(tunnel_failures=0, status="origin_unhealthy")
        if result["origin_failures"] >= 3:
            recover(ORIGIN)
            result.update(origin_failures=0, recovered=ORIGIN)
        return result
    if not is_active(TUNNEL):
        result.update(tunnel_failures=0, status="tunnel_stopped")
        return result
    if not public_url or not re.fullmatch(
        r"https://[a-z0-9-]+\.trycloudflare\.com", public_url
    ):
        tunnel_ok = False
    else:
        tunnel_ok = probe(public_url)
    result["tunnel_failures"] = 0 if tunnel_ok else result.get("tunnel_failures", 0) + 1
    result["status"] = "healthy" if tunnel_ok else "tunnel_unhealthy"
    if result["tunnel_failures"] >= 3:
        recover(TUNNEL)
        result.update(tunnel_failures=0, recovered=TUNNEL)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    path = args.cache / "watchdog.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    try:
        public_url = json.loads((args.cache / "public-url.json").read_text())["url"]
    except (OSError, ValueError, KeyError):
        public_url = None
    result = check(state, public_url)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
