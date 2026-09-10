#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/fetch_results.sh <run_id>" >&2
  exit 2
fi

RUN_ID="$1"
REMOTE_HOST="${REMOTE_HOST:-root@<实例>}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/RadioMind}"
mkdir -p "results/${RUN_ID}"
scp -r "${REMOTE_HOST}:${REMOTE_ROOT}/results/${RUN_ID}/." "results/${RUN_ID}/"
echo "Fetched results/${RUN_ID}"
