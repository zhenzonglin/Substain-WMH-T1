#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
participant="${1:-all}"
profile="${2:-auto}"
cores="${3:-96}"
echo "[03/06] WMH20 + T1-20特征及四图生成: ${participant} (${profile}, total_cores=${cores})"

# 正式GPU全量运行统一进入严格完例波次；单例和CPU调试保持原有直接Snakemake入口。
if [[ "${participant}" == "all" && "${profile}" == "gpu" ]]; then
  exec "${root}/scripts/finish_backlog_then_all.sh" "${cores}"
fi

PYTHONPATH="${root}/src" "${root}/envs/core-venv/bin/python" -m substain_features.cli run \
  --config-file "${root}/config/config.yaml" --participant-id "${participant}" \
  --profile "${profile}" --cores "${cores}" --skip-prepare
