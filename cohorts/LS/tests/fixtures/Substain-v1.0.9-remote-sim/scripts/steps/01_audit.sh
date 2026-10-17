#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
participant="${1:-all}"
echo "[02/06] 输入、FSL MNI152病灶与资源审计: ${participant}"
PYTHONPATH="${root}/src" "${root}/envs/core-venv/bin/python" -m substain_features.cli audit \
  --config-file "${root}/config/config.yaml" --participant-id "${participant}"
