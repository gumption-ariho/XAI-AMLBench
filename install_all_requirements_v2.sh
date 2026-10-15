#!/usr/bin/env bash
# install_all_requirements_v2.sh  -  install everything needed to run XAI-AMLBench's four application
# services locally (no Docker image builds), per the README's "Run everything on this laptop" section.
#
# What it does:
#   1. Creates ./.venv if it doesn't exist yet, and uses it from here on (never touches system Python).
#   2. Installs PyTorch 2.4.1 first, and separately from everything else, because it needs its own index
#      on Linux/Windows for a CPU-only build (the plain PyPI wheel defaults to a multi-GB CUDA build).
#      macOS has no separate CPU index -- PyPI's own wheel is already CPU/MPS, so it's installed plainly there.
#   3. Installs requirements.txt (the root file, which pulls in aml_synth/, gnn_aml_core/, backend/ and
#      xai_explainer/'s own per-service requirements files -- see requirements.txt's own comment for why
#      those stay separate instead of being merged: each Docker image installs only its own service's deps).
#
# Usage:
#   bash install_all_requirements_v2.sh                 # CPU-only PyTorch (default: works on any laptop)
#   bash install_all_requirements_v2.sh --gpu            # skip the CPU-only index, let pip resolve a CUDA build
#   bash install_all_requirements_v2.sh --dev            # also installs pytest/pytest-cov and the xgboost baseline
#   bash install_all_requirements_v2.sh --with-frontend  # also runs `npm install` in frontend/ (needs Node 18.18+)
#   bash install_all_requirements_v2.sh --venv .myenv    # use a different venv directory
#
# Safe to re-run: every step here is idempotent (venv creation, pip install, npm install).
set -euo pipefail

VENV_DIR=".venv"
TORCH_VERSION="2.4.1"
GPU=0
DEV=0
WITH_FRONTEND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=1; shift ;;
    --dev) DEV=1; shift ;;
    --with-frontend) WITH_FRONTEND=1; shift ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

cd "$(dirname "$0")"

PY="$(command -v python3.11 || command -v python3.10 || command -v python3 || true)"
if [[ -z "$PY" ]]; then
  echo "no python3 found on PATH (XAI-AMLBench needs Python >= 3.10)" >&2
  exit 1
fi
PY_OK=$("$PY" -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [[ "$PY_OK" != "1" ]]; then
  echo "found $("$PY" --version), but XAI-AMLBench needs Python >= 3.10" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> creating $VENV_DIR with $("$PY" --version)"
  "$PY" -m venv "$VENV_DIR"
else
  echo "==> reusing existing $VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "expected $VENV_PY after venv creation but didn't find it (Windows venv? use $VENV_DIR\\Scripts\\python.exe and adapt this script)" >&2
  exit 1
fi

echo "==> upgrading pip"
"$VENV_PY" -m pip install --upgrade pip --quiet

echo "==> installing PyTorch $TORCH_VERSION"
OS_NAME="$(uname -s)"
if [[ "$GPU" == "1" ]]; then
  echo "    (--gpu: letting pip resolve whatever build it finds, usually CUDA)"
  "$VENV_PY" -m pip install "torch==${TORCH_VERSION}"
elif [[ "$OS_NAME" == "Darwin" ]]; then
  echo "    (macOS: PyPI's own wheel is CPU/MPS already, no separate index needed)"
  "$VENV_PY" -m pip install "torch==${TORCH_VERSION}"
else
  echo "    (CPU-only build, smaller download, no GPU needed -- pass --gpu to override)"
  "$VENV_PY" -m pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> installing requirements.txt (aml_synth + gnn_aml_core + backend + xai_explainer)"
"$VENV_PY" -m pip install -r requirements.txt

if [[ "$DEV" == "1" ]]; then
  echo "==> --dev: installing pytest, pytest-cov and the xgboost baseline"
  "$VENV_PY" -m pip install "pytest>=7" pytest-cov xgboost
fi

if [[ "$WITH_FRONTEND" == "1" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "warning: --with-frontend was passed but npm is not on PATH; skipping frontend/ install" >&2
  else
    echo "==> --with-frontend: npm install in frontend/"
    (cd frontend && npm install)
  fi
fi

echo ""
echo "==> done. Activate with:  source ${VENV_DIR}/bin/activate"
echo "    Then, per the README's 'Run everything on this laptop' section:"
echo "      ${VENV_PY} -m gnn_aml_core.train --data data --out models"
echo "      python3 run_local.py"
