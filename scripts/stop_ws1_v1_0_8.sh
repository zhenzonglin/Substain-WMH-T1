#!/usr/bin/env bash
set -euo pipefail

default_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
root_arg="${1:-${default_root}}"
if [[ ! -d "${root_arg}" ]]; then
  echo "项目目录不存在: ${root_arg}" >&2
  exit 2
fi
root="$(cd "${root_arg}" && pwd -P)"
pid_file="${root}/logs/full_run_v1.0.8.pid"
if [[ ! -s "${pid_file}" ]]; then
  echo "PID文件不存在或为空: ${pid_file}" >&2
  exit 2
fi
run_pid="$(tr -d '[:space:]' < "${pid_file}")"
if [[ ! "${run_pid}" =~ ^[0-9]+$ ]]; then
  echo "PID无效: ${run_pid}" >&2
  exit 2
fi
if [[ ! -d "/proc/${run_pid}" ]]; then
  echo "PID已不存在: ${run_pid}"
  exit 0
fi
run_cwd="$(readlink -f "/proc/${run_pid}/cwd")"
run_pgid="$(ps -o pgid= -p "${run_pid}" | tr -d ' ')"
if [[ "${run_cwd}" != "${root}" ]]; then
  echo "PID不属于当前项目: PID=${run_pid}, CWD=${run_cwd}" >&2
  exit 2
fi
if [[ "${run_pgid}" != "${run_pid}" ]]; then
  echo "PGID不等于主PID，停止操作: PID=${run_pid}, PGID=${run_pgid}" >&2
  exit 2
fi

echo "===== 即将停止的整个进程组 ====="
ps -eo pid=,ppid=,pgid=,stat=,etime=,args= | awk -v group="${run_pgid}" '$3 == group'
echo "向进程组 ${run_pgid} 发送TERM"
kill -TERM -- "-${run_pgid}"

for _ in $(seq 1 60); do
  if ! ps -eo pgid= | awk -v group="${run_pgid}" '$1 == group {found=1} END {exit !found}'; then
    echo "工作站1进程组已全部退出"
    exit 0
  fi
  sleep 1
done

echo "TERM 60秒后仍有残留；不自动KILL，请保留以下输出并人工检查：" >&2
ps -eo pid=,ppid=,pgid=,stat=,etime=,args= | awk -v group="${run_pgid}" '$3 == group' >&2
exit 1
