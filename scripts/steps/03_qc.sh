#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
participant="${1:-all}"
echo "[04/06] 启动可恢复四图人工QC: ${participant}"
PYTHONPATH="${root}/src" "${root}/envs/core-venv/bin/python" -m substain_features.cli qc \
  --config-file "${root}/config/config.yaml" --participant-id "${participant}"
