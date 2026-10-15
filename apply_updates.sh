#!/usr/bin/env bash
# apply_updates.sh  -  applies whatever files are in the paired zip to your XAI-AMLBench checkout. Run this
# FROM INSIDE your project root (~/Desktop/XAI-AMLBench), with this script and the zip sitting in the same
# folder you downloaded them to (e.g. ~/Downloads).
#
# Usage, from ~/Desktop/XAI-AMLBench:
#   bash ~/Downloads/apply_updates.sh ~/Downloads/xai-amlbench-ci-latency-fixes-vN.zip
#
# What it does (generic -- works for whatever's actually in the zip, not a fixed file list):
#   - for every file in the zip that ALREADY EXISTS in your project and differs from the incoming copy,
#     backs it up to *.bak-<timestamp> first, so nothing you've customised is silently lost
#   - copies every file from the zip into place at the matching path (creating directories as needed)
#   - makes any top-level *.sh file it installs executable
set -euo pipefail

ZIP="${1:-}"
if [[ -z "$ZIP" || ! -f "$ZIP" ]]; then
  echo "error: zip not found at '${ZIP:-<none given>}'" >&2
  echo "usage: bash apply_updates.sh /path/to/xai-amlbench-ci-latency-fixes-vN.zip" >&2
  exit 1
fi

if [[ ! -f "pyproject.toml" || ! -d "gnn_aml_core" ]]; then
  echo "error: this doesn't look like the XAI-AMLBench project root (no pyproject.toml / gnn_aml_core/ here)." >&2
  echo "cd into ~/Desktop/XAI-AMLBench first, then re-run this script." >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
UNZIP_DIR="$(mktemp -d)"
trap 'rm -rf "$UNZIP_DIR"' EXIT

echo "==> extracting $ZIP"
unzip -q -o "$ZIP" -d "$UNZIP_DIR"

SELF_NAME="$(basename "$0")"
CHANGED=()
NEW=()

while IFS= read -r -d '' src; do
  rel="${src#"$UNZIP_DIR"/}"
  [[ "$rel" == "$SELF_NAME" ]] && continue   # don't try to "install" this script over itself
  if [[ -f "$rel" ]]; then
    if cmp -s "$rel" "$src"; then
      continue   # identical already -- nothing to do, no backup noise
    fi
    echo "==> backing up existing $rel -> $rel.bak-$TS"
    cp "$rel" "$rel.bak-$TS"
    CHANGED+=("$rel")
  else
    NEW+=("$rel")
  fi
  mkdir -p "$(dirname "$rel")"
  cp "$src" "$rel"
  if [[ "$rel" == *.sh && "$(dirname "$rel")" == "." ]]; then
    chmod +x "$rel"
  fi
done < <(find "$UNZIP_DIR" -type f -print0)

echo ""
echo "==> done."
if [[ ${#NEW[@]} -gt 0 ]]; then
  echo "    new files:"
  for f in "${NEW[@]}"; do echo "      $f"; done
fi
if [[ ${#CHANGED[@]} -gt 0 ]]; then
  echo "    updated files (backed up first):"
  for f in "${CHANGED[@]}"; do echo "      $f"; done
fi
echo ""
echo "If you have a real .env (not just .env.example), new GNN_* / NEO4J_* vars default sensibly even without"
echo "adding them there -- only add a line if you want a non-default value."
echo ""
echo "Next: restart whatever's running (python3 run_local.py, or docker compose restart <service>), then:"
echo "    python3 check_latency_v1.py --gnn http://127.0.0.1:8001 --concurrency 10"
