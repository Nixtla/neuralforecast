"""Real-worker smoke test using a trusted saved GRU input from this benchmark."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--local-gpus", default="0")
    parser.add_argument("--colab-sessions", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--timeout", type=int, default=2400)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError("Use a fresh validation output directory")
    os.environ["NF_COLAB_SESSIONS"] = str(options.colab_sessions)
    os.environ["NF_TASKVINE_LOCAL_GPUS"] = options.local_gpus
    sys.path[:0] = [
        str(options.source.resolve() / "experiments/commodity_sota"),
        str(options.source.resolve()),
    ]
    from taskvine_queue import TaskVineQueue
    from neuralforecast.benchmark import Fold

    args = list(pickle.loads(options.input.read_bytes()))
    if args[1]["name"] != "GRU":
        raise ValueError("Use a saved GRU input for this bounded smoke test")
    args[1] = dict(args[1], tracking=None)
    args[4] = Fold(0, 96, 96 + args[3]["h"])
    args[5], args[6] = 10, None
    args[7] = options.output.resolve() / "checkpoints_phase1"
    queue = TaskVineQueue(options.output)
    deadline = time.monotonic() + options.timeout

    def collect(tokens):
        results = []
        while tokens:
            queue.check_manager()
            if time.monotonic() > deadline:
                raise TimeoutError("TaskVine validation exceeded its deadline")
            ready = [
                t for t in tokens if (queue.session / "jobs" / t / "done.json").exists()
            ]
            for token in ready:
                result = queue.get(token)
                if not result.get("ok"):
                    raise RuntimeError(result)
                results.append(result)
                tokens.remove(token)
            time.sleep(0.1)
        return results

    try:
        expected = (
            len([s for s in options.local_gpus.split(",") if s])
            + options.colab_sessions
        )
        if not expected:
            raise ValueError("At least one worker is required")
        while True:
            queue.check_manager()
            if time.monotonic() > deadline:
                raise TimeoutError("Workers did not become ready")
            path = queue.root / "state.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            if (
                sum(s is not None for s in state.get("devices", {}).values())
                >= expected
            ):
                break
            time.sleep(1)
        tokens = []
        for cid in range(expected):
            payload = list(args)
            payload[2] = 1000 + cid
            tokens.append(queue.submit(*payload))
        initial = collect(tokens)
        assert len({r["execution_host"] for r in initial}) == expected
        args[2] = initial[0]["config_id"]
        args[5], args[6] = 20, initial[0]["checkpoint"]
        continued = collect([queue.submit(*args)])[0]
        assert continued["actual_steps"] == 20
        assert continued["resource_reserved"]["ram_bytes"] <= 4 * 2**30
        shared = []
        if expected == 1:
            args[5], args[6] = 30, None
            tokens = []
            for cid in (2000, 2001):
                payload = list(args)
                payload[2] = cid
                tokens.append(queue.submit(*payload))
            shared = collect(tokens)
            intervals = [r["execution_interval"] for r in shared]
            assert max(r["start"] for r in intervals) < min(r["end"] for r in intervals)
            assert all(r["resource_reserved"]["ram_bytes"] <= 4 * 2**30 for r in shared)
        report = dict(initial=initial, continued=continued, shared=shared)
        (options.output / "validation.json").write_text(json.dumps(report, indent=2))
        print(f"PASS: {options.output / 'validation.json'}")
    finally:
        queue.shutdown()


if __name__ == "__main__":
    main()
