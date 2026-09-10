#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/fetch_results.sh <run_id>" >&2
  exit 2
fi

RUN_ID="$1"
REMOTE_HOST="${REMOTE_HOST:-root@<实例>}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/RadioMind}"
REMOTE_PORT="${REMOTE_PORT:-}"
mkdir -p "results/${RUN_ID}"
scp_args=(-r)
if [[ -n "${REMOTE_PORT}" ]]; then
  scp_args=(-P "${REMOTE_PORT}" "${scp_args[@]}")
fi
scp "${scp_args[@]}" "${REMOTE_HOST}:${REMOTE_ROOT}/results/${RUN_ID}/." "results/${RUN_ID}/"
echo "Fetched results/${RUN_ID}"
