#!/usr/bin/env bash
# Isolated scheduler installation: never changes the training virtualenv.
set -euo pipefail
vine_home="${NF_TASKVINE_HOME:-$HOME/.local/share/taskvine}"
mkdir -p "$vine_home/bootstrap"
curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$vine_home/bootstrap/micromamba.tar.bz2"
python3 - "$vine_home/bootstrap" <<'PY'
import pathlib, sys, tarfile
p = pathlib.Path(sys.argv[1])
with tarfile.open(p / 'micromamba.tar.bz2') as archive:
    archive.extract('bin/micromamba', p, filter='data')
PY
"$vine_home/bootstrap/bin/micromamba" create -y -p "$vine_home/env" -c conda-forge python=3.11 ndcctools=7.17.1
"$vine_home/bootstrap/bin/micromamba" list -p "$vine_home/env" --explicit > "$vine_home/environment.lock"
