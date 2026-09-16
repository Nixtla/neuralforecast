"""Run a Quick Tunnel and persist its current public URL after every restart."""

import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    process = subprocess.Popen(
        [
            str(Path.home() / ".local/bin/cloudflared"),
            "tunnel",
            "--no-autoupdate",
            "--protocol",
            "http2",
            "--url",
            f"http://127.0.0.1:{args.port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def stop(signum, frame):
        process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            args.state.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.state.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "url": match.group(),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
            temporary.replace(args.state)
    raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
