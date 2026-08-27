#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MAX_PARALLEL_JOBS=200
DEFAULT_TOTAL_CORES=96
GPU_POLICY="auto_one_job_per_device"
mode="${1:-all}"
participant="${2:-all}"
profile="${3:-auto}"
# 工作站有112个逻辑线程；默认使用96个，给系统、文件服务和人工监控预留16个。
# 第4个位置参数仍可临时下调或上调，但GPU任务始终由设备锁限制为每卡1个。
cores="${4:-${DEFAULT_TOTAL_CORES}}"

case "${mode}" in
  all)
    "${root}/scripts/steps/00_prepare.sh"
    "${root}/scripts/steps/01_audit.sh" "${participant}"
    "${root}/scripts/steps/02_features.sh" "${participant}" "${profile}" "${cores}"
    echo "批量特征提取及四张QC图已完成；人工QC不会阻断计算。"
    echo "需要审核时运行: ./run_pipeline.sh qc ${participant}"
    echo "全部审核后运行: ./run_pipeline.sh export ${participant} && ./run_pipeline.sh verify"
    ;;
  prepare) "${root}/scripts/steps/00_prepare.sh" ;;
  audit) "${root}/scripts/steps/01_audit.sh" "${participant}" ;;
  run|features) "${root}/scripts/steps/02_features.sh" "${participant}" "${profile}" "${cores}" ;;
  qc) "${root}/scripts/steps/03_qc.sh" "${participant}" ;;
  export) "${root}/scripts/steps/04_export.sh" "${participant}" ;;
  verify) "${root}/scripts/steps/05_verify.sh" ;;
  offline) "${root}/scripts/steps/06_offline_check.sh" ;;
  offline-smoke)
    shift
    "${root}/scripts/steps/06_offline_check.sh" --smoke-test "$@"
    ;;
  *)
    echo "用法: ./run_pipeline.sh {all|prepare|audit|run|qc|export|verify|offline|offline-smoke} [参数]" >&2
    exit 2
    ;;
esac
