#!/usr/bin/env bash
set -euo pipefail

# 新项目完成prepare和audit后，用独立进程组启动可恢复的全量计算。
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
profile="${PIPELINE_PROFILE:-auto}"
cores="${TOTAL_CORES:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)}"
cpu_threads="${CPU_THREADS_PER_JOB:-8}"
batch_size="${BATCH_SIZE:-200}"
pid_file="${root}/logs/full_run.pid"
log_pointer="${root}/logs/full_run.logpath"

if [[ ! "${profile}" =~ ^(auto|gpu|cpu)$ ]]; then
  echo "PIPELINE_PROFILE只允许auto/gpu/cpu，收到: ${profile}" >&2
  exit 2
fi
for value_name in cores cpu_threads batch_size; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name}必须为正整数，收到: ${value}" >&2
    exit 2
  fi
done

audit_report="${root}/derivatives/substain_features/audit/audit_report.json"
core_python="${root}/envs/core-venv/bin/python"
if [[ ! -x "${core_python}" ]]; then
  echo "核心Python不存在: ${core_python}" >&2
  exit 2
fi
if [[ ! -f "${audit_report}" ]]; then
  echo "请先执行 ./run_pipeline.sh prepare 和 ./run_pipeline.sh audit all" >&2
  exit 2
fi
if ! "${core_python}" - "${audit_report}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "pass" else 1)
PY
then
  echo "audit未通过: ${audit_report}" >&2
  exit 2
fi

if [[ "${profile}" == "auto" ]]; then
  visible_gpu="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${visible_gpu}" && ! "${visible_gpu}" =~ ^[0-9]+$ && "${visible_gpu}" != "-1" && "${visible_gpu}" != "none" && "${visible_gpu}" != "None" ]]; then
    echo "本发布版的全量GPU入口只接受一张GPU编号；收到CUDA_VISIBLE_DEVICES=${visible_gpu}" >&2
    exit 2
  fi
  if [[ "${visible_gpu}" =~ ^[0-9]+$ ]] || { [[ -z "${visible_gpu}" ]] && nvidia-smi -L >/dev/null 2>&1; }; then
    profile="gpu"
  else
    profile="cpu"
  fi
fi
if [[ "${profile}" == "gpu" && ! "${CUDA_VISIBLE_DEVICES:-0}" =~ ^[0-9]+$ ]]; then
  echo "本发布版的全量GPU入口只接受一张GPU编号；收到CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 2
fi

mkdir -p "${root}/logs"
if [[ -s "${pid_file}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "已有全量主进程在运行: ${old_pid}" >&2
    exit 2
  fi
fi

run_tag="$(date +%Y%m%d_%H%M%S)"
run_log="${root}/logs/full_run_${profile}_${cpu_threads}threads_${run_tag}.log"
cd "${root}"
export CPU_THREADS_PER_JOB="${cpu_threads}"
export BATCH_SIZE="${batch_size}"
setsid nohup bash "${root}/run_pipeline.sh" run all "${profile}" "${cores}" \
  >"${run_log}" 2>&1 < /dev/null &
run_pid=$!
printf '%s\n' "${run_pid}" > "${pid_file}"
printf '%s\n' "${run_log}" > "${log_pointer}"

sleep 2
if ! kill -0 "${run_pid}" 2>/dev/null; then
  echo "启动后立即退出；日志: ${run_log}" >&2
  tail -n 100 "${run_log}" >&2 || true
  exit 1
fi
run_pgid="$(ps -o pgid= -p "${run_pid}" | tr -d ' ')"
if [[ "${run_pgid}" != "${run_pid}" ]]; then
  echo "启动进程未成为独立进程组: PID=${run_pid}, PGID=${run_pgid}" >&2
  exit 1
fi
echo "启动成功: PID=${run_pid}, PGID=${run_pgid}, profile=${profile}, cores=${cores}, threads=${cpu_threads}, batch=${batch_size}"
echo "日志: ${run_log}"
