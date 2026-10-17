#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
echo "[01/06] 递归匹配输入并生成无session软链接"
PYTHONPATH="${root}/src" "${root}/envs/core-venv/bin/python" -m substain_features.cli prepare-inputs \
  --config-file "${root}/config/config.yaml"
