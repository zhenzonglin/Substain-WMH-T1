#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
echo "[辅助] 离线资源完整性检查（不创建压缩包）"
PYTHONPATH="${root}/src" "${root}/envs/core-venv/bin/python" -m substain_features.cli verify-offline \
  --config-file "${root}/config/config.yaml" "$@"
