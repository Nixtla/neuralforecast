"""Owned Colab runtimes connected as ordinary TaskVine workers over SSH."""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

from taskvine_resources import atomic_json

PACKAGES = (
    "torch",
    "torchvision",
    "pytorch-lightning",
    "lightning",
    "torchmetrics",
    "ray",
    "pyarrow",
    "numpy",
    "pandas",
    "scipy",
    "coreforecast",
    "utilsforecast",
    "fsspec",
    "optuna",
    "psutil",
    "wandb",
    "transformers",
    "peft",
    "accelerate",
    "einops",
    "xlstm",
    "timm",
    "setuptools",
)

BOOTSTRAP = r"""
import json, os, pathlib, subprocess, tarfile, urllib.request
root = pathlib.Path('/content/nf-vine')
root.mkdir(exist_ok=True)
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:' + os.environ.get('LD_LIBRARY_PATH', '')
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
assert 'L4' in gpu, gpu
subprocess.run(['pip', 'install', 'uv'], check=True)
subprocess.run(['uv', 'venv', '--python', '3.11', str(root/'train')], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', str(root/'train/bin/python'),
    '--extra-index-url', 'https://download.pytorch.org/whl/cu124',
    '--index-strategy', 'unsafe-best-match', '-r', str(root/'requirements.txt')], check=True)
urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/linux-64/latest', root/'mamba.tar.bz2')
with tarfile.open(root/'mamba.tar.bz2') as archive:
    archive.extract('bin/micromamba', root, filter='data')
subprocess.run([str(root/'bin/micromamba'), 'create', '-y', '-p', str(root/'vine'),
    '-c', 'conda-forge', 'python=3.11', 'ndcctools=7.17.1'], check=True)
subprocess.run([str(root/'train/bin/python'), '-c', 'import torch; assert torch.cuda.is_available()'], check=True)
mem = dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
import shutil
print('NF_VINE_READY=' + json.dumps(dict(memory=int(int(mem['MemAvailable'].split()[0])/1024*.85),
    cores=len(os.sched_getaffinity(0)), disk=int(shutil.disk_usage(root).free/2**20*.8))))
"""


class Session:
    def __init__(self, root, index, stop):
        self.root = Path(root)
        self.directory = self.root / "colab" / str(index)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = f"nf-vine-{self.root.name[-8:]}-{index}"
        self.config = self.directory / "sessions.json"
        self.key = self.directory / "id_ed25519"
        self.stop_event = stop
        self.process = None
        self.spec = self.root / "devices" / f"{self.name}.json"
        self.log = (self.directory / "bootstrap.log").open("a", buffering=1)
        if not self.key.exists():
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(self.key), "-q"],
                check=True,
            )

    def cli(self, *args, timeout=180):
        return self.run(
            ["colab", "--auth", "oauth2", "--config", str(self.config), *args], timeout
        )

    def run(self, command, timeout, text=None, capture=False, cleanup=False):
        self.log.write("COMMAND " + shlex.join(command[:3]) + "\n")
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE if capture else self.log,
            stderr=self.log,
            text=True,
        )
        deadline = time.monotonic() + timeout
        first = True
        try:
            while True:
                if time.monotonic() > deadline or (
                    self.stop_event.is_set() and not cleanup
                ):
                    raise TimeoutError("Colab command stopped or timed out")
                try:
                    output, _ = process.communicate(
                        input=text if first else None, timeout=1
                    )
                    if process.returncode:
                        raise RuntimeError(
                            f"Colab command failed ({process.returncode}); see {self.log.name}"
                        )
                    return output
                except subprocess.TimeoutExpired:
                    first = False
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def ssh(self):
        proxy = shlex.join(
            [
                "colab",
                "--auth",
                "oauth2",
                "--config",
                str(self.config),
                "ssh",
                "--proxy-mode",
                "-s",
                self.name,
                "-i",
                str(self.key),
            ]
        )
        return [
            "ssh",
            "-i",
            str(self.key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.directory/'known_hosts'}",
            "-o",
            f"HostKeyAlias={self.name}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"ProxyCommand={proxy}",
        ]

    def prepare(self):
        self.cli("new", "-s", self.name, "--gpu", "L4")
        # Each newly owned runtime has a new SSH host key, even when its name
        # is reused after a disconnect. Other hosts' trust files are untouched.
        (self.directory / "known_hosts").unlink(missing_ok=True)
        self.run(self.ssh() + ["root@colab", "mkdir -p /content/nf-vine"], 60)
        requirements = self.directory / "requirements.txt"
        pins = []
        for package in PACKAGES:
            try:
                name = "ray[train,tune]" if package == "ray" else package
                pins.append(f"{name}=={version(package)}")
            except PackageNotFoundError:
                pass
        requirements.write_text("\n".join(pins) + "\n")
        for local, remote in [
            (requirements, "/content/nf-vine/requirements.txt"),
            (self.root / "password", "/content/nf-vine/password"),
        ]:
            self.cli("upload", "-s", self.name, str(local), remote, timeout=180)
        output = self.run(
            self.ssh() + ["root@colab", "python3 -"], 1800, text=BOOTSTRAP, capture=True
        )
        self.log.write(output)
        ready = next(
            line.split("=", 1)[1]
            for line in output.splitlines()
            if line.startswith("NF_VINE_READY=")
        )
        spec = dict(
            json.loads(ready),
            name=self.name,
            device="0",
            local=False,
            python="/content/nf-vine/train/bin/python",
            ld_library_path="/usr/lib64-nvidia",
        )
        port = json.loads((self.root / "ready.json").read_text())["port"]
        command = [
            "/content/nf-vine/vine/bin/vine_worker",
            "--gpus=0",
            f"--cores={spec['cores']}",
            f"--memory={spec['memory']}",
            f"--disk={spec['disk']}",
            "--feature",
            self.name,
            "--password",
            "/content/nf-vine/password",
            "--single-shot",
            "--idle-timeout=86400",
            "--parent-death",
            "localhost",
            "9123",
        ]
        remote = (
            "chmod 600 /content/nf-vine/password; export LD_LIBRARY_PATH=/usr/lib64-nvidia; exec "
            + shlex.join(command)
        )
        self.process = subprocess.Popen(
            self.ssh() + ["-R", f"9123:127.0.0.1:{port}", "root@colab", remote],
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=self.log,
        )
        atomic_json(self.spec, spec)

    def stop(self):
        self.spec.unlink(missing_ok=True)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.config.exists():
            try:
                self.run(
                    [
                        "colab",
                        "--auth",
                        "oauth2",
                        "--config",
                        str(self.config),
                        "stop",
                        "-s",
                        self.name,
                    ],
                    45,
                    cleanup=True,
                )
                # Confirm the owned runtime is stopped before permitting its
                # attempts to execute elsewhere (including W&B side effects).
                atomic_json(self.spec.with_suffix(".lost"), dict(timestamp=time.time()))
            except Exception as exc:
                self.log.write(f"CLEANUP ERROR: {exc}\n")

    def supervise(self):
        failures = 0
        while not self.stop_event.is_set():
            try:
                self.prepare()
                atomic_json(
                    self.directory / "status.json",
                    dict(status="connected", timestamp=time.time()),
                )
                while not self.stop_event.wait(1):
                    if self.process.poll() is not None:
                        raise RuntimeError("Colab SSH worker disconnected")
            except Exception as exc:
                failures += 1
                atomic_json(
                    self.directory / "status.json",
                    dict(error=str(exc), failures=failures, timestamp=time.time()),
                )
            finally:
                self.stop()
            self.stop_event.wait(min(900, 60 * 2 ** min(failures, 4)))
        self.log.close()


def main():
    root = Path(sys.argv[1]).resolve()
    settings = json.loads((root / "settings.json").read_text())
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    count = settings["colab_sessions"]
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(Session(root, i, stop).supervise) for i in range(count)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
