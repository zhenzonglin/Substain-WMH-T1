#!/usr/bin/env bash
set -euo pipefail

root="${1:-/data/usersdir/linzhenzong/Substain}"
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
patch_file="${bundle_dir}/ws1_gpu_throughput_first.patch"
checksum_file="${bundle_dir}/SHA256SUMS"
pid_file="${root}/logs/full_run_v1.0.9.pid"
log_pointer="${root}/logs/full_run_v1.0.9.logpath"
config="${root}/config/config.yaml"
snakefile="${root}/workflow/Snakefile"
start_script="${root}/scripts/start_ws1_v1_0_9.sh"
core_python="${root}/envs/core-venv/bin/python"
derivatives="${root}/derivatives/substain_features"

if [[ ! -d "${root}" || -L "${root}" ]]; then
  echo "项目根目录不存在或是软链接: ${root}" >&2
  exit 2
fi
root="$(cd "${root}" && pwd -P)"
if [[ "${root}" != "/data/usersdir/linzhenzong/Substain" ]]; then
  echo "拒绝修改非预期项目: ${root}" >&2
  exit 2
fi
for required in "${patch_file}" "${checksum_file}" "${snakefile}" \
  "${start_script}" "${core_python}" "${config}"; do
  if [[ ! -e "${required}" || -L "${required}" ]]; then
    echo "必需文件不存在或是软链接: ${required}" >&2
    exit 2
  fi
done
if [[ ! -x "${core_python}" || ! -x "${start_script}" ]]; then
  echo "项目Python或启动脚本不可执行" >&2
  exit 2
fi

(cd "${bundle_dir}" && sha256sum -c SHA256SUMS)
nvidia-smi -i 0 --query-gpu=index --format=csv,noheader,nounits | grep -qx '0'

state="$("${core_python}" - "${snakefile}" "${start_script}" <<'PY'
import re
import sys
from pathlib import Path

snake = Path(sys.argv[1]).read_text(encoding="utf-8")
start = Path(sys.argv[2]).read_text(encoding="utf-8")

def rule(name):
    match = re.search(r"^rule {}:\n(?P<body>.*?)(?=^rule |\Z)".format(name), snake, re.M | re.S)
    if match is None:
        raise SystemExit("缺少规则: {}".format(name))
    return match.group("body")

t1 = rule("t1")
wmh = rule("wmh_segmentation")
old = 'gpu=1 if PROFILE == "gpu" else 0' in wmh
new = 'gpu=GPU_SLOTS_PER_DEVICE if PROFILE == "gpu" else 0' in wmh
if new and not old:
    print("already")
    raise SystemExit(0)
if not old or new:
    raise SystemExit("WMH GPU令牌现场上下文不匹配")
required = (
    'gpu=1 if PROFILE == "gpu" else 0' in t1,
    "gpu_slots_required=GPU_SLOTS_PER_DEVICE" in wmh,
    'wmh_exclusive=1 if PROFILE == "gpu" else 0' in wmh,
    "GPU_SLOTS_PER_DEVICE=2" in start,
    "CUDA_VISIBLE_DEVICES=0" in start,
)
if not all(required):
    raise SystemExit("现场不是GPU0双T1、WMH独占版本")
print("ready")
PY
)"
if [[ "${state}" == "already" ]]; then
  echo "吞吐优先WMH双令牌补丁已经存在；未重复停止分析。"
  exit 0
fi
if [[ "${state}" != "ready" ]]; then
  echo "现场核对失败: ${state}" >&2
  exit 2
fi

cd "${root}"
git apply --check --whitespace=error "${patch_file}"

if [[ ! -s "${pid_file}" ]]; then
  echo "缺少活动PID文件: ${pid_file}" >&2
  exit 2
fi
pid="$(tr -d '[:space:]' < "${pid_file}")"
if [[ ! "${pid}" =~ ^[1-9][0-9]*$ || ! -d "/proc/${pid}" ]]; then
  echo "PID无效或主分析已经退出: ${pid}" >&2
  exit 2
fi
pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
leader_cwd="$(readlink -f "/proc/${pid}/cwd")"
leader_cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
if [[ "${pid}" != "${pgid}" || "${leader_cwd}" != "${root}" ]]; then
  echo "拒绝停止：PID/PGID/cwd不匹配: PID=${pid}, PGID=${pgid}, cwd=${leader_cwd}" >&2
  exit 2
fi
if [[ "${leader_cmd}" != *"finish_ws1_v1_0_9.sh"* ]]; then
  echo "拒绝停止：主进程命令不匹配: ${leader_cmd}" >&2
  exit 2
fi

run_tag="$(date +%Y%m%d_%H%M%S)"
archive_dir="${root}/archive/gpu-throughput-first-${run_tag}"
if [[ -e "${archive_dir}" ]]; then
  echo "归档目录已存在: ${archive_dir}" >&2
  exit 2
fi
mkdir -p "${archive_dir}/before"
cp -a "${snakefile}" "${archive_dir}/before/Snakefile"
sha256sum "${archive_dir}/before/Snakefile" > "${archive_dir}/before/SHA256SUMS"
ps -eo pid,ppid,pgid,stat,etime,args | awk -v group="${pgid}" '$3 == group' \
  > "${archive_dir}/process_group.before.txt"

"${core_python}" - "${pgid}" "${archive_dir}/active_stages.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

pgid = int(sys.argv[1])
output = Path(sys.argv[2])
active = set()
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if os.getpgid(int(proc.name)) != pgid:
            continue
        args = [
            item.decode("utf-8", "replace")
            for item in (proc / "cmdline").read_bytes().split(b"\0")
            if item
        ]
    except (OSError, ProcessLookupError):
        continue
    if "stage" not in args or "--participant-id" not in args:
        continue
    try:
        stage = args[args.index("stage") + 1].replace("-", "_")
        participant = args[args.index("--participant-id") + 1]
    except (IndexError, ValueError):
        continue
    if participant != "all":
        active.add((participant, stage))
payload = {
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "pgid": pgid,
    "active": [
        {"participant_id": participant, "stage": stage}
        for participant, stage in sorted(active)
    ],
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("停止前活动阶段数: {}".format(len(active)))
PY

echo "发送TERM停止旧进程组: ${pgid}"
kill -TERM -- "-${pgid}"
for _ in $(seq 1 60); do
  remaining="$(ps -eo pgid= | awk -v group="${pgid}" '$1 == group {n++} END {print n+0}')"
  [[ "${remaining}" -eq 0 ]] && break
  sleep 1
done
remaining="$(ps -eo pgid= | awk -v group="${pgid}" '$1 == group {n++} END {print n+0}')"
if [[ "${remaining}" -ne 0 ]]; then
  echo "TERM后仍有${remaining}个进程；未使用KILL，也未应用补丁。" >&2
  ps -eo pid,ppid,pgid,stat,etime,args | awk -v group="${pgid}" '$3 == group' >&2
  exit 1
fi

"${core_python}" - "${archive_dir}/active_stages.json" "${derivatives}" \
  "${archive_dir}/interrupted-status" <<'PY'
import json
import os
import sys
from pathlib import Path

active = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["active"]
derivatives = Path(sys.argv[2]).resolve()
archive = Path(sys.argv[3]).resolve()
moved = 0
for item in active:
    source = derivatives / "sub-{}".format(item["participant_id"]) / "status" / "{}.json".format(item["stage"])
    if not source.is_file() or source.is_symlink():
        continue
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SystemExit("无法解析中断状态: {}".format(source))
    if payload.get("status") != "fail":
        continue
    destination = archive / "sub-{}".format(item["participant_id"]) / "status" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    moved += 1
print("已归档TERM产生的fail状态: {}".format(moved))
PY

patched=0
restore_on_error() {
  status=$?
  trap - EXIT
  if [[ "${status}" -ne 0 && "${patched}" -eq 1 ]]; then
    cp -a "${archive_dir}/before/Snakefile" "${snakefile}"
    echo "验证失败，已恢复原Snakefile；分析保持停止。" >&2
  fi
  exit "${status}"
}
trap restore_on_error EXIT

git apply --whitespace=error "${patch_file}"
patched=1

"${core_python}" - "${snakefile}" <<'PY'
import re
import sys
from pathlib import Path

snake = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"^rule wmh_segmentation:\n(?P<body>.*?)(?=^rule |\Z)", snake, re.M | re.S)
if match is None:
    raise SystemExit("缺少wmh_segmentation规则")
wmh = match.group("body")
assert 'gpu=GPU_SLOTS_PER_DEVICE if PROFILE == "gpu" else 0' in wmh
assert 'wmh_exclusive=1 if PROFILE == "gpu" else 0' in wmh
assert "gpu_slots_required=GPU_SLOTS_PER_DEVICE" in wmh
assert 'gpu=1 if PROFILE == "gpu" else 0' not in wmh
print("吞吐优先静态检查: WMH请求2个Snakemake令牌并持有2个物理槽=PASS")
PY

PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
  --snakefile "${snakefile}" --configfile "${config}" --list \
  --config "active_config_file=${config}" selected_participant=all \
    profile=gpu gpu_devices=0 gpu_slots_per_device=2 \
  > "${archive_dir}/snakemake-rules.txt"
grep -qx 'wmh_segmentation' "${archive_dir}/snakemake-rules.txt"

PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
  --snakefile "${snakefile}" --configfile "${config}" --cores 1 --unlock \
  --config "active_config_file=${config}" selected_participant=all \
    profile=gpu gpu_devices=0 gpu_slots_per_device=2

trap - EXIT
patched=2
bash "${start_script}"
sleep 3

new_pid="$(tr -d '[:space:]' < "${pid_file}")"
if [[ ! "${new_pid}" =~ ^[1-9][0-9]*$ || ! -d "/proc/${new_pid}" ]]; then
  echo "重启后PID无效: ${new_pid}" >&2
  exit 1
fi
new_pgid="$(ps -o pgid= -p "${new_pid}" | tr -d ' ')"
new_cwd="$(readlink -f "/proc/${new_pid}/cwd")"
if [[ "${new_pid}" != "${new_pgid}" || "${new_cwd}" != "${root}" ]]; then
  echo "重启验收失败: PID=${new_pid}, PGID=${new_pgid}, cwd=${new_cwd}" >&2
  exit 1
fi
log_path="$(tr -d '[:space:]' < "${log_pointer}")"
if ! kill -0 "${new_pid}" 2>/dev/null; then
  echo "新主进程提前退出；日志: ${log_path}" >&2
  tail -n 100 "${log_path}" >&2 || true
  exit 1
fi

echo "吞吐优先补丁部署完成: PID=${new_pid}, PGID=${new_pgid}"
echo "物理GPU0；T1每例1令牌/1槽，最多同时2例；WMH请求2令牌/2槽独占"
echo "注意：存在可运行T1时，WMH可能延后到两个令牌同时空闲。"
echo "日志: ${log_path}"
echo "部署前归档: ${archive_dir}"
