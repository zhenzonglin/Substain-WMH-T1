#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
pid_file="${root}/logs/full_run_v1.0.9.pid"
log_pointer="${root}/logs/full_run_v1.0.9.logpath"
run_tag="$(date +%Y%m%d_%H%M%S)"
run_log="${root}/logs/full_run_v1.0.9_4threads_${run_tag}.log"

mkdir -p "${root}/logs"
if [[ -s "${pid_file}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "已有v1.0.9主进程在运行: ${old_pid}" >&2
    exit 2
  fi
fi
if [[ ! -s "${root}/logs/ws1_v1.0.9_rerun_manifest.path" ]]; then
  echo "请先执行 prepare_ws1_v1_0_9_rerun.py --apply" >&2
  exit 2
fi

cd "${root}"
setsid nohup env \
  CPU_THREADS_PER_JOB=4 \
  BATCH_SIZE=200 \
  CUDA_VISIBLE_DEVICES=0 \
  bash "${root}/scripts/finish_ws1_v1_0_9.sh" 96 \
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
echo "启动成功: PID=${run_pid}, PGID=${run_pgid}"
echo "日志: ${run_log}"
