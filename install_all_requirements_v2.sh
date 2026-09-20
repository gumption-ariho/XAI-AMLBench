#!/usr/bin/env bash
# install_all_requirements_v2.sh  -  installs EVERY dependency of XAI-AMLBench on this computer, fast.
#
#   bash install_all_requirements_v2.sh           # do it (typically 1-4 minutes)
#   bash install_all_requirements_v2.sh --dry-run  # only print what it would do
#
# Options:
#   --cuda           install the default (GPU/CUDA) PyTorch instead of the small CPU build
#   --no-frontend    skip `npm install` in frontend/
#   --no-verify      skip the import check and the 20-second training smoke test
#   --docker-build   afterwards also pre-build all Docker images in parallel (app + ml + obs profiles)
#   --recreate       throw away and rebuild the .venv even if it already looks right
#
# It creates ONE root requirements.txt that includes the four per-service files (aml_synth, gnn_aml_core,
# backend, xai_explainer), so you only ever deal with a single file. The per-service files stay because each
# Docker image installs only its own service's packages.
#
# Run it from the project root (the folder that contains backend/, frontend/, docker-compose.yml).
# Safe to run again: it only redoes what is missing.
#
# Why a virtual environment and not "system-wide"?  Ubuntu refuses system-wide pip installs (PEP 668), and
# Python 3.14 has no PyTorch wheels yet. A project .venv with Python 3.11 (the same as the Docker images)
# also fixes the red import underlines in PyCharm.
set -euo pipefail

PY_VERSION="3.11"
TORCH_VERSION="2.4.1"          # same as the Docker image, so models trained here load there and vice versa
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DRY=0; DO_FRONTEND=1; DO_VERIFY=1; DO_DOCKER=0; CUDA=0; RECREATE=0
usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --cuda) CUDA=1 ;;
    --no-frontend) DO_FRONTEND=0 ;;
    --no-verify) DO_VERIFY=0 ;;
    --docker-build) DO_DOCKER=1 ;;
    --recreate) RECREATE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $a"; echo; usage; exit 2 ;;
  esac
done

if [[ -t 1 ]]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=""; G=""; Y=""; R=""; N=""; fi
step() { echo; echo "${B}== $*${N}"; }
ok()   { echo "  ${G}OK${N}    $*"; }
warn() { echo "  ${Y}WARN${N}  $*"; }
die()  { echo "  ${R}ERROR${N} $*" >&2; exit 1; }
run()  { printf '  $'; printf ' %q' "$@"; echo; if [[ $DRY -eq 0 ]]; then "$@"; fi; }

echo "install_all_requirements_v2.sh  (one root requirements.txt)"
[[ $DRY -eq 1 ]] && echo "${Y}DRY RUN: nothing will be installed or changed.${N}"
echo "Project: $ROOT"

# ------------------------------------------------------------------------------------------ 1. requirement files
step "1/6  Requirement files"
SERVICE_FILES=(aml_synth/requirements.txt gnn_aml_core/requirements.txt backend/requirements.txt xai_explainer/requirements.txt)
PRESENT=()
for r in "${SERVICE_FILES[@]}"; do
  if [[ -f "$r" ]]; then ok "$r"; PRESENT+=("$r"); else warn "$r not found (run setup_backend_v2.py first)"; fi
done
[[ ${#PRESENT[@]} -gt 0 ]] || die "No requirements.txt files here. Run this script from the project root."
echo "  (node_agent needs nothing: it only uses the Python standard library)"

write_root_requirements() {
  {
    cat <<'HDR'
# XAI-AMLBench  -  ONE file to install everything for local development
#
#     pip install -r requirements.txt          (or run:  bash install_all_requirements_v2.sh   -> does it all, fast)
#
# It only INCLUDES the per-service files below, so nothing is listed twice. Those files stay separate on purpose:
# each Docker image installs just its own service's packages (the backend image must not download PyTorch).
#
# Not in this file (needs a special source, the install script handles both):
#   torch==2.4.1   CPU build:  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
#   vllm           GPU only; it lives inside the xai-narrative-api Docker image

numpy<2            # PyTorch 2.4.1 was built for NumPy 1.x (the Docker image uses 1.26)

HDR
    for r in "${PRESENT[@]}"; do echo "-r $r"; done
  } > requirements.txt
}
if [[ -f requirements.txt ]]; then
  ok "requirements.txt (root file, already there - left untouched)"
elif [[ $DRY -eq 1 ]]; then
  echo "  (dry run) would create ONE root requirements.txt that includes: ${PRESENT[*]}"
else
  write_root_requirements
  ok "created requirements.txt: one file that includes all the per-service files"
fi

# ------------------------------------------------------------------------------------------ 2. uv
step "2/6  uv (fast Python installer)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  ok "found $(uv --version)"
else
  command -v curl >/dev/null 2>&1 || die "curl is missing. Install it with:  sudo apt-get install -y curl   then run this script again."
  run bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if [[ $DRY -eq 0 ]]; then command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH. Open a new terminal and run this script again."; fi
fi

# ------------------------------------------------------------------------------------------ 3. virtual environment
step "3/6  Virtual environment (.venv, Python $PY_VERSION)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
NEED_VENV=1
if [[ -x "$PY" ]]; then
  CUR="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo unknown)"
  if [[ "$CUR" == "$PY_VERSION" && $RECREATE -eq 0 ]]; then
    ok "reusing existing .venv (Python $CUR)"; NEED_VENV=0
  else
    warn "existing .venv is Python $CUR; replacing it (the old one is kept in _old_backup/)"
    run mkdir -p "$ROOT/_old_backup"
    run mv "$VENV" "$ROOT/_old_backup/venv-py${CUR}-$(date +%Y%m%d%H%M%S)"
  fi
fi
if [[ $NEED_VENV -eq 1 ]]; then
  run uv venv --python "$PY_VERSION" "$VENV"     # downloads a standalone Python $PY_VERSION if the system has none
fi

# ------------------------------------------------------------------------------------------ 4. python packages
step "4/6  Python packages"
if [[ $CUDA -eq 1 ]]; then
  echo "  PyTorch (default build with CUDA support, large download)"
  run uv pip install --python "$PY" "torch==$TORCH_VERSION"
else
  echo "  PyTorch (CPU build, about 200 MB)"
  if ! run uv pip install --python "$PY" "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu; then
    warn "CPU wheel index not reachable; falling back to the standard PyTorch build (larger download)"
    run uv pip install --python "$PY" "torch==$TORCH_VERSION"
  fi
fi
echo "  Everything else, from the single root requirements.txt (resolved together in one go):"
run uv pip install --python "$PY" -r requirements.txt

# ------------------------------------------------------------------------------------------ 5. verify
step "5/6  Verifying"
if [[ $DO_VERIFY -eq 0 ]]; then
  echo "  skipped (--no-verify)"
elif [[ $DRY -eq 1 ]]; then
  echo "  (dry run) would import every package, then generate a tiny dataset and train for 15 epochs as a smoke test"
else
  "$PY" - <<'PYEOF'
import importlib, sys
mods = ["numpy", "pandas", "sklearn", "torch", "torch_geometric", "fastapi", "uvicorn", "pydantic", "httpx",
        "psycopg", "redis", "kafka", "neo4j", "immudb", "prometheus_client", "prometheus_fastapi_instrumentator", "networkx"]
missing = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m:<36}{getattr(mod, '__version__', '')}")
    except Exception as exc:                      # noqa: BLE001
        missing.append(m)
        print(f"  FAIL  {m:<36}{exc.__class__.__name__}: {exc}")
import torch
print(f"\n  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
sys.exit(1 if missing else 0)
PYEOF
  echo
  echo "  Smoke test: tiny synthetic dataset + 15 training epochs (about 20 seconds)"
  SMOKE="$(mktemp -d)"
  "$PY" -m aml_synth.graph_generator --accounts 800 --background-tx 6000 --patterns 4 --out "$SMOKE/data" --to csv 2>&1 | grep -E "generated" | cut -c1-150
  "$PY" -m gnn_aml_core.train --data "$SMOKE/data" --out "$SMOKE/models" --epochs 15 2>&1 | grep -E "graph:|TEST:|saved" | cut -c1-200
  rm -rf "$SMOKE"
  ok "the whole pipeline (data -> features -> GATv2 training) runs on this machine"
fi

# ------------------------------------------------------------------------------------------ 6. frontend (+ docker)
step "6/6  Frontend (Node.js) and Docker"
if [[ $DO_FRONTEND -eq 1 && -f frontend/package.json ]]; then
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 \
     && node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>18||(a===18&&b>=18)?0:1)'; then
    ok "Node $(node -v), npm $(npm -v)"
    ( cd frontend && run npm install --no-audit --no-fund --loglevel=error )
  else
    warn "Node.js 18.18 or newer not found, so frontend/ was skipped. Fast install on Ubuntu:"
    echo "        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
    echo "        then run this script again (it only redoes what is missing)."
  fi
else
  echo "  frontend skipped"
fi
if [[ $DO_DOCKER -eq 1 ]]; then
  if command -v docker >/dev/null 2>&1 && { [[ $DRY -eq 1 ]] || docker info >/dev/null 2>&1; }; then
    run docker compose --profile app --profile ml --profile obs build
  else
    warn "Docker is not available or not running; skipped --docker-build"
  fi
fi

# ------------------------------------------------------------------------------------------ done
echo
echo "${B}${G}Done in ${SECONDS}s.${N}"
cat <<EOF

Use it:
  source .venv/bin/activate
  python -m aml_synth.graph_generator --out data --to csv       # make data
  python -m gnn_aml_core.train --data data --out models         # train (CPU, minutes) - the Docker API reads ./models

PyCharm: File > Settings > Project > Python Interpreter > Add Interpreter > Existing >
         $VENV/bin/python      (the red import underlines disappear once it is indexed)
EOF