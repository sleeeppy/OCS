#!/usr/bin/env bash
# Create the OCS virtualenv on macOS / Linux.
#
# The macOS counterpart to setup_env.ps1, with one deliberate difference: the
# GPU half is opt-in rather than mandatory.
#
# setup_env.ps1 installs torch from the CUDA wheel index and exits non-zero when
# torch.cuda.is_available() is false. That is the right call on Windows, where a
# missing GPU means a broken install. On an Apple Silicon Mac there is no CUDA to
# find, and failing there would block a stack that is otherwise entirely usable:
# see-through's decomposition is the only GPU stage. Everything downstream --
# cleanup, silhouette, bone placement, limb partition, meshing, weighting, atlas
# packing, Spine export, preview -- is numpy/opencv/scipy and runs fine on CPU.
#
# So the default install skips torch and see-through's requirements entirely
# (~12 GB of weights and a pinned diffusers/transformers/PyQt6 stack OCS never
# imports on this path) and takes under a minute. Use --with-gpu on a CUDA
# machine to get the full thing.
#
# Usage:
#   ./scripts/setup_env.sh              # OCS only; no GPU stage
#   ./scripts/setup_env.sh --with-gpu   # + torch + see-through requirements
set -euo pipefail

PYTHON_VERSION="3.12"
TORCH_VERSION="2.8.0"
WITH_GPU=0

while [ $# -gt 0 ]; do
  case "$1" in
    --with-gpu) WITH_GPU=1 ;;
    --python) PYTHON_VERSION="$2"; shift ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
SEE_THROUGH="$ROOT/external/see-through"

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it:" >&2
  echo "  brew install uv        # macOS" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "== creating venv (Python $PYTHON_VERSION)"
# uv downloads a managed CPython if the system has no 3.12, so this works
# without touching brew/pyenv.
uv venv --python "$PYTHON_VERSION" "$VENV"
export VIRTUAL_ENV="$VENV"

echo "== installing OCS"
uv pip install -e "$ROOT[dev]"

if [ "$WITH_GPU" -eq 1 ]; then
  if [ ! -f "$SEE_THROUGH/requirements.txt" ]; then
    echo "== fetching see-through submodule"
    git -C "$ROOT" submodule update --init --recursive
  fi

  echo "== installing torch $TORCH_VERSION"
  if [ "$(uname -s)" = "Darwin" ]; then
    # No CUDA wheels for macOS; the default index gives the MPS-capable build.
    uv pip install "torch==$TORCH_VERSION" "torchvision==0.23.0" "torchaudio==$TORCH_VERSION"
  else
    uv pip install "torch==$TORCH_VERSION" "torchvision==0.23.0" "torchaudio==$TORCH_VERSION" \
      --index-url https://download.pytorch.org/whl/cu128
  fi

  echo "== installing see-through requirements"
  # cwd must be the submodule root: its requirements.txt has relative editable
  # entries (-e ./common, -e ./annotators).
  ( cd "$SEE_THROUGH" && uv pip install -r requirements.txt )

  # see-through's own setup instructions end with this. Its scripts default to
  # assets/... paths that only resolve through the link, so --srcp with a default
  # fails without it.
  echo "== linking see-through assets"
  ln -sfn common/assets "$SEE_THROUGH/assets"
fi

echo "== checking the GPU stage"
# Reported, never fatal. OCS itself is already installed and working at this
# point; this only decides whether the see-through step is available.
"$PY" -c "
from ocs import seethrough
info = seethrough.check_environment()
if info.get('ok'):
    mem = f\"  {info['memory_gb']} GB\" if info.get('memory_gb') else ''
    print(f\"  torch {info['torch']}  {info['device']}  {info['name']}{mem}\")
    print('  see-through decomposition: available')
    if info['device'] == 'mps':
        print('  keep group offload ON; without it a 1280 run swaps on 24 GB')
else:
    print('  see-through decomposition: NOT available on this machine')
    print(f\"  {info.get('error', 'no accelerator found')}\".rstrip())
    print('  Everything after decomposition still works. To get an input:')
    print('    python scripts/make_demo_project.py --all   # synthetic, no GPU')
    print('    python scripts/import_psd.py <file.psd>     # PSD from elsewhere')
"

echo
echo "Done. Next:"
echo "  ./scripts/fetch_spine_player.sh   # optional, for offline previews"
echo "  ./scripts/run_ocs.sh              # start the editor"
