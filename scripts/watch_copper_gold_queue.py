"""Recover the reserved queue after the observed transient W&B timeout."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path('/home/t-lab01/neuralforecast')
QUEUE = ROOT / 'results/copper-gold-queue'
UNIT = 'neuralforecast-copper-gold-queue.service'
INITIAL_FILES = {'eligibility.csv', 'wandb', 'wandb_run.json', 'weekly.pkl'}


def transient_wandb_timeout(log):
    # Inspect only the final traceback, not an older failure in the append-only log.
    traceback = log.rsplit('wandb.errors.errors.AuthenticationError:', 1)
    return len(traceback) == 2 and (
        'the service process is busy and did not respond in time.'
        in traceback[1].split('\n', 1)[0]
    ) and 'CalledProcessError' in traceback[1] and (
        traceback[1].count('Traceback (most recent call last):') <= 1
    )


def preserve_initialization(output, backup):
    if not output.exists() or (output / 'run_config.json').exists():
        return
    if not {p.name for p in output.iterdir()} <= INITIAL_FILES:
        raise RuntimeError(f'Refusing to move unrecognized experiment data: {output}')
    backup.mkdir(parents=True, exist_ok=True)
    output.rename(backup / output.name)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    active = subprocess.run(
        ['systemctl', '--user', 'show', UNIT, '--property=ActiveState', '--value'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if active not in {'failed', 'inactive'}:
        return
    state = json.loads((QUEUE / 'state.json').read_text())
    if state.get('status') != 'stopped':
        return
    log_path = QUEUE / 'session.log'
    with log_path.open('rb') as stream:
        stream.seek(max(0, log_path.stat().st_size - 16000))
        log = stream.read().decode(errors='replace')
    if not transient_wandb_timeout(log):
        return
    record_path = QUEUE / 'auto-recovery.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    failure = state['updated_at']
    if record.get('failure') == failure or record.get('attempts', 0) >= 3:
        return
    # Preserve pre-run files; the original launcher otherwise adds --resume even
    # when W&B failed before run_config.json was written.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = QUEUE / ('auto-recovery-' + stamp)
    for name in ('copper', 'copper-diff', 'copper-exog', 'gold', 'gold-diff', 'gold-exog'):
        preserve_initialization(ROOT / 'results' / name, backup)
    save(record_path, {'failure': failure, 'attempts': record.get('attempts', 0) + 1,
                       'requested_at': stamp, 'status': 'restart-requested'})
    subprocess.run(['systemctl', '--user', 'restart', UNIT], check=True)
    print(f'Restarted {UNIT} after transient W&B timeout', flush=True)


if __name__ == '__main__':
    main()
