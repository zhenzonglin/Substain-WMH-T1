#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
participant="${1:-all}"
echo "[05/06] 按人工QC导出正式40维表: ${participant}"
PYTHONPATH="${root}/src" "${root}/envs/core-venv/bin/python" -m substain_features.cli export \
  --config-file "${root}/config/config.yaml" --participant-id "${participant}"
