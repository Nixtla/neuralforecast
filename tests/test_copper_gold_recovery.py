"""Regression coverage for recovery from W&B startup timeouts."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'recovery', Path(__file__).parents[1] / 'scripts/watch_copper_gold_queue.py'
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_timeout_classification():
    message = ('wandb.errors.errors.AuthenticationError: Failed to verify credentials: '
               'the service process is busy and did not respond in time.\n'
               'subprocess.CalledProcessError: exit status 1')
    assert recovery.transient_wandb_timeout(message)
    assert not recovery.transient_wandb_timeout(
        message + '\nTraceback (most recent call last):\nRuntimeError: CUDA'
        + '\nTraceback (most recent call last):\nCalledProcessError: exit status 1'
    )
    assert not recovery.transient_wandb_timeout(message.replace(
        'the service process is busy and did not respond in time.', 'invalid API key'))
    assert not recovery.transient_wandb_timeout('RuntimeError: CUDA out of memory')


def test_initialization_is_preserved(tmp_path):
    output = tmp_path / 'gold'
    output.mkdir()
    (output / 'weekly.pkl').write_bytes(b'data')
    backup = tmp_path / 'backup'
    recovery.preserve_initialization(output, backup)
    assert (backup / 'gold/weekly.pkl').read_bytes() == b'data'
    assert not output.exists()


def test_checkpoint_and_unknown_files_are_protected(tmp_path):
    output = tmp_path / 'gold'
    output.mkdir()
    (output / 'run_config.json').write_text('{}')
    recovery.preserve_initialization(output, tmp_path / 'backup')
    assert (output / 'run_config.json').exists()
    (output / 'run_config.json').unlink()
    (output / 'checkpoint.ckpt').write_bytes(b'checkpoint')
    with pytest.raises(RuntimeError):
        recovery.preserve_initialization(output, tmp_path / 'backup')
    assert (output / 'checkpoint.ckpt').exists()
