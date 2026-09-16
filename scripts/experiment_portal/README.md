# Public experiment portal

A read-only HTML/JSON portal for this repository's W&B experiments. It runs on
127.0.0.1:18081 and is exposed using a Cloudflare Quick Tunnel. No domain or
Cloudflare account is needed. The computer, portal and tunnel must stay online.
The random URL changes when the tunnel process restarts; it is not durable
Cloudflare storage or a permanent deployment.

## Data and fidelity

- Discover all immediate `results/*/run_config.json` W&B destinations and this
  repository's `results/*/dashboard.json` destinations, including future experiments.
- Fetch every page of every project's run inventory, including old groups and failed
  runs. Preserve W&B config, summary and system-summary JSON strings as returned.
- Commit each complete project snapshot atomically. Compare source and fetched run
  counts, expose timestamps, and retain the previous snapshot if a refresh fails.
- Refresh the catalog after each synchronization cycle, waiting 120 seconds between
  cycles. Running data remains provisional and W&B itself can have ingestion delays.
- Fetch full training history with `scan_history` without a metric-key filter or
  sampling. JSON/JSONL preserve values; CSV provides the union of all metric keys.
- Request all system events using the source event count and explicitly flag a
  mismatch/truncation. Original history/event parquet artifacts are also available.
- Expose every run-file listing and input/output artifact listing returned by W&B.
  Serve original file bytes and artifact entries through the portal; signed upstream
  URLs and W&B credentials remain server-side. Keep downloaded files in local cache.
- Display local leaderboard CSVs separately and provide exact source-file downloads.
- Charts show every finite point; original data downloads retain the full values.

**This is not a claim of a 100% independent offline mirror.** Run histories, file
contents and artifacts are loaded on demand from W&B and cached. A deleted or
inaccessible W&B object that has not been cached cannot be recovered here. W&B's
proprietary dashboard layouts, workspace/report UI and account management are not
replicated. The portal is read-only and cannot modify experiment runs.

## URLs

- `/`: experiment overview (server-rendered; no JavaScript required)
- `/projects/{entity}--{project}`: experiment results and searchable run pages
- `/projects/{id}/runs/{run}`: exact summary/config and forecast visualization
- `/projects/{id}/runs/{run}/charts`: full-resolution training curves
- `/llms.txt`: instructions and links for AI web readers
- `/export/analysis.md`: attachable result tables and experimental context; use when
  a web reader refuses access to the temporary domain. This is a summary export,
  not a replacement for the full run history and artifacts.
- `/api/experiments`, `/healthz`: timestamps, counts and synchronization status
- `/api/projects/{id}/snapshot`: all run metadata, original JSON strings preserved
- `/api/projects/{id}/runs?page=1&limit=100&q=...`: paginated searchable inventory
- `/api/projects/{id}/runs/{run}`: current original run metadata
- Append `/history`, `/history.jsonl`, `/history.csv`, `/system`, `/files`, or
  `/artifacts` to a run API URL. Follow listed URLs to read original files and
  artifact manifests/contents. `source_line_count`, `returned_rows`, `sampled` and
  `count_matches_source` make retrieval coverage explicit.

## Running

```sh
.venv/bin/python scripts/experiment_portal/server.py --port 18081 --interval 120
```

Authentication uses `WANDB_API_KEY` or the current user's W&B netrc credentials.
`WANDB_BASE_URL` supports another W&B server. No credentials are put in URLs.
Only result destinations discovered in the above metadata files can be queried;
the server does not expose the repository filesystem as a directory listing.

Systemd user unit templates are in `systemd/`, assuming `%h/neuralforecast`.
The tunnel wrapper writes its current URL to
`results/experiment-portal/public-url.json` on each restart.

```sh
systemctl --user status neuralforecast-portal neuralforecast-portal-tunnel
cat results/experiment-portal/public-url.json
journalctl --user -u neuralforecast-portal.service -n 30
```

Both units restart on failure and are enabled at login; user lingering keeps them
running after logout. Stop publication with:

```sh
systemctl --user disable --now neuralforecast-portal-tunnel neuralforecast-portal
```

## Validation

```sh
.venv/bin/pytest tests/test_experiment_portal.py --no-cov -q
```

Checks cover multi-page completeness, precision preservation, failed-sync atomicity,
unsampled sparse histories, system-history truncation detection, path and symlink
isolation, read-only HTTP routing, crawler-visible HTML, pagination and chart points.

## Continuous synchronization and recovery

The origin and tunnel are enabled user services with `Restart=always` and no
start-rate lockout, so repeated failures do not permanently disable retries.
User lingering is enabled on this host to start services at boot and keep them
running after logout.

`neuralforecast-portal-watchdog.timer` checks the origin and public URL about
once a minute. Three consecutive unhealthy checks restart only the affected
service. A failed W&B request alone does not trigger restarts: old snapshots
remain available and the normal synchronization loop retries. The origin reports
whether the synchronization worker has made progress within 15 minutes, allowing
the watchdog to recover a stalled worker even if HTTP is still responding.
Failure counters reset after recovery or a changed tunnel URL. Deliberately
stopped services are left stopped.

Inspect the last check in `results/experiment-portal/watchdog.json` and the latest
address in `results/experiment-portal/public-url.json`. New experiment metadata is
discovered on every synchronization cycle. For a full shutdown, also disable the
watchdog timer:

```sh
systemctl --user disable --now neuralforecast-portal-watchdog.timer
systemctl --user disable --now neuralforecast-portal-tunnel neuralforecast-portal
```

A Quick Tunnel still requires this host and its internet connection to be online;
its address can change if the tunnel is restarted. Automatic recovery does not
make the address permanent or copy all remote W&B files into independent storage.
