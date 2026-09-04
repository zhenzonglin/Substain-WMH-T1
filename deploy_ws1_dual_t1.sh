#!/usr/bin/env bash
set -euo pipefail

root="${1:-/data/usersdir/linzhenzong/Substain}"
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
patch_file="${bundle_dir}/ws1_dual_t1_wmh_exclusive.patch"
verify_script="${bundle_dir}/verify_ws1_gpu_slots.py"
checksum_file="${bundle_dir}/SHA256SUMS"
pid_file="${root}/logs/full_run_v1.0.9.pid"
config="${root}/config/config.yaml"
snakefile="${root}/workflow/Snakefile"
gpu_pool="${root}/src/substain_features/gpu_pool.py"
finish_script="${root}/scripts/finish_ws1_v1_0_9.sh"
start_script="${root}/scripts/start_ws1_v1_0_9.sh"
core_python="${root}/envs/core-venv/bin/python"

if [[ ! -d "${root}" || -L "${root}" ]]; then
  echo "项目根目录不存在或是软链接: ${root}" >&2
  exit 2
fi
root="$(cd "${root}" && pwd -P)"
if [[ "${root}" != "/data/usersdir/linzhenzong/Substain" ]]; then
  echo "拒绝修改非预期项目: ${root}" >&2
  exit 2
fi
for required in "${patch_file}" "${verify_script}" "${checksum_file}" \
  "${snakefile}" "${gpu_pool}" "${finish_script}" "${start_script}"; do
  if [[ ! -e "${required}" || -L "${required}" ]]; then
    echo "必需文件不存在或是软链接: ${required}" >&2
    exit 2
  fi
done
if [[ ! -x "${core_python}" ]]; then
  echo "core Python不可执行: ${core_python}" >&2
  exit 2
fi

(cd "${bundle_dir}" && sha256sum -c SHA256SUMS)
nvidia-smi -i 0 --query-gpu=index --format=csv,noheader,nounits | grep -qx '0'

cd "${root}"
state="$("${core_python}" - "${snakefile}" "${gpu_pool}" "${finish_script}" "${start_script}" <<'PY'
import re
import sys
from pathlib import Path

snake = Path(sys.argv[1]).read_text(encoding="utf-8")
pool = Path(sys.argv[2]).read_text(encoding="utf-8")
finish = Path(sys.argv[3]).read_text(encoding="utf-8")
start = Path(sys.argv[4]).read_text(encoding="utf-8")

def rule(name):
    match = re.search(r"^rule {}:\n(?P<body>.*?)(?=^rule |\Z)".format(name), snake, re.M | re.S)
    if match is None:
        raise SystemExit("缺少规则: {}".format(name))
    return match.group("body")

t1 = rule("t1")
wmh = rule("wmh_segmentation")
registration = rule("registration")
cleanup = rule("cleanup")
new_markers = (
    "GPU_SLOTS_PER_DEVICE" in snake,
    "--slots-per-gpu" in pool,
    "gpu_slots=2" in finish,
    "GPU_SLOTS_PER_DEVICE=2" in start,
)
if all(new_markers):
    print("already")
    raise SystemExit(0)
if any(new_markers):
    raise SystemExit("现场存在部分双槽修改，拒绝继续")
if "ROLLING_ORDER_VALUE" not in snake or "run_rolling" not in finish:
    raise SystemExit("现场没有滚动200例补丁")
if 'gpu=1 if PROFILE == "gpu" else 0' not in t1 or "needs_gpu=True" not in t1:
    raise SystemExit("T1当前不是GPU规则")
if 'gpu=1 if PROFILE == "gpu" else 0' not in wmh or "needs_gpu=True" not in wmh:
    raise SystemExit("WMH当前不是GPU规则")
if "threads: 8" not in registration or "needs_gpu=False" not in registration:
    raise SystemExit("registration当前不是CPU 8线程规则")
if "stage cleanup" not in cleanup:
    raise SystemExit("cleanup规则缺失")
if '--resources "gpu=1"' not in finish or '"gpu_devices=0"' not in finish:
    raise SystemExit("当前启动资源不是GPU0单令牌")
if "CUDA_VISIBLE_DEVICES=0" not in start:
    raise SystemExit("启动脚本没有固定物理GPU0")
print("ready")
PY
)"
if [[ "${state}" == "already" ]]; then
  echo "双T1、WMH独占补丁已经应用；未重复停止或重启。"
  exit 0
fi
if [[ "${state}" != "ready" ]]; then
  echo "现场规则核对失败: ${state}" >&2
  exit 2
fi
echo "现场规则核对: 滚动200例、T1 GPU、WMH GPU、registration CPU8、cleanup、物理GPU0=PASS"

git apply --check --whitespace=error "${patch_file}"

if [[ ! -s "${pid_file}" ]]; then
  echo "缺少当前PID文件: ${pid_file}" >&2
  exit 2
fi
pid="$(tr -d '[:space:]' < "${pid_file}")"
if [[ ! "${pid}" =~ ^[1-9][0-9]*$ || ! -d "/proc/${pid}" ]]; then
  echo "PID无效或进程已退出: ${pid}" >&2
  exit 2
fi
pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
leader_cwd="$(readlink -f "/proc/${pid}/cwd")"
leader_cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
if [[ "${pgid}" != "${pid}" || "${leader_cwd}" != "${root}" ]]; then
  echo "拒绝停止：PID/PGID/cwd校验失败: PID=${pid}, PGID=${pgid}, cwd=${leader_cwd}" >&2
  exit 2
fi
if [[ "${leader_cmd}" != *"finish_ws1_v1_0_9.sh"* ]]; then
  echo "拒绝停止：主进程命令不匹配: ${leader_cmd}" >&2
  exit 2
fi

run_tag="$(date +%Y%m%d_%H%M%S)"
archive_dir="${root}/archive/dual-t1-wmh-exclusive-${run_tag}"
if [[ -e "${archive_dir}" ]]; then
  echo "归档目录已存在: ${archive_dir}" >&2
  exit 2
fi
mkdir -p "${archive_dir}/before"
cp -a "${snakefile}" "${archive_dir}/before/Snakefile"
cp -a "${gpu_pool}" "${archive_dir}/before/gpu_pool.py"
cp -a "${finish_script}" "${archive_dir}/before/finish_ws1_v1_0_9.sh"
cp -a "${start_script}" "${archive_dir}/before/start_ws1_v1_0_9.sh"
sha256sum "${archive_dir}/before/"* > "${archive_dir}/before/SHA256SUMS"
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
    pid = int(proc.name)
    try:
        if os.getpgid(pid) != pgid:
            continue
        args = [
            value.decode("utf-8", "replace")
            for value in (proc / "cmdline").read_bytes().split(b"\0")
            if value
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
print("捕获当前阶段: {}".format(len(active)))
PY

echo "发送TERM停止旧进程组: ${pgid}"
kill -TERM -- "-${pgid}"
for _ in $(seq 1 60); do
  remaining="$(ps -eo pgid= | awk -v group="${pgid}" '$1 == group {count++} END {print count+0}')"
  [[ "${remaining}" -eq 0 ]] && break
  sleep 1
done
remaining="$(ps -eo pgid= | awk -v group="${pgid}" '$1 == group {count++} END {print count+0}')"
if [[ "${remaining}" -ne 0 ]]; then
  echo "TERM后仍有${remaining}个进程；未使用KILL，也未应用补丁。" >&2
  ps -eo pid,ppid,pgid,stat,etime,args | awk -v group="${pgid}" '$3 == group' >&2
  exit 1
fi
echo "旧进程组已停止"

"${core_python}" - "${archive_dir}/active_stages.json" \
  "${root}/derivatives/substain_features" "${archive_dir}/interrupted-status" <<'PY'
import json
import os
import sys
from pathlib import Path

active_path = Path(sys.argv[1])
derivatives = Path(sys.argv[2]).resolve()
archive = Path(sys.argv[3]).resolve()
payload = json.loads(active_path.read_text(encoding="utf-8"))
moved = 0
for item in payload["active"]:
    participant = item["participant_id"]
    stage = item["stage"]
    source = derivatives / "sub-{}".format(participant) / "status" / "{}.json".format(stage)
    if not source.is_file() or source.is_symlink():
        continue
    try:
        status = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SystemExit("无法解析中断状态: {}".format(source))
    if status.get("status") != "fail":
        continue
    destination = archive / "sub-{}".format(participant) / "status" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    moved += 1
print("已归档中断fail状态: {}".format(moved))
PY

patched=0
restore_on_error() {
  status=$?
  trap - EXIT
  if [[ "${status}" -ne 0 && "${patched}" -eq 1 ]]; then
    cp -a "${archive_dir}/before/Snakefile" "${snakefile}"
    cp -a "${archive_dir}/before/gpu_pool.py" "${gpu_pool}"
    cp -a "${archive_dir}/before/finish_ws1_v1_0_9.sh" "${finish_script}"
    cp -a "${archive_dir}/before/start_ws1_v1_0_9.sh" "${start_script}"
    echo "补丁验证失败，已恢复部署前四个文件；分析保持停止。" >&2
  fi
  exit "${status}"
}
trap restore_on_error EXIT

git apply --whitespace=error "${patch_file}"
patched=1
chmod 755 "${finish_script}" "${start_script}"
"${core_python}" -m py_compile "${gpu_pool}"
bash -n "${finish_script}"
bash -n "${start_script}"
PYTHONPATH="${root}/src" "${core_python}" "${verify_script}" "${root}"

"${core_python}" - "${snakefile}" "${gpu_pool}" "${finish_script}" "${start_script}" <<'PY'
import re
import sys
from pathlib import Path

snake = Path(sys.argv[1]).read_text(encoding="utf-8")
pool = Path(sys.argv[2]).read_text(encoding="utf-8")
finish = Path(sys.argv[3]).read_text(encoding="utf-8")
start = Path(sys.argv[4]).read_text(encoding="utf-8")

def rule(name):
    match = re.search(r"^rule {}:\n(?P<body>.*?)(?=^rule |\Z)".format(name), snake, re.M | re.S)
    if match is None:
        raise SystemExit("缺少规则: {}".format(name))
    return match.group("body")

t1 = rule("t1")
wmh = rule("wmh_segmentation")
assert "GPU_SLOTS_PER_DEVICE" in snake
assert "gpu_slots_required=GPU_SLOTS_PER_DEVICE" in wmh
assert "wmh_exclusive=1" in wmh
assert "gpu_slots_required" not in t1
assert "--slots-per-gpu" in pool and "--slots-required" in pool
assert "gpu_slots=2" in finish
assert '--resources "gpu=${gpu_slots}" "wmh_exclusive=1"' in finish
assert '"gpu_devices=0"' in finish
assert "CUDA_VISIBLE_DEVICES=0" in start
assert "GPU_SLOTS_PER_DEVICE=2" in start
print("双T1、WMH独占、物理GPU0静态检查: PASS")
PY

validation_order="${archive_dir}/validation.order"
PYTHONPATH="${root}/src" "${core_python}" - "${config}" "${validation_order}" <<'PY'
import sys
from pathlib import Path
from substain_features.schema import load_config, load_participants

config_path = Path(sys.argv[1]).resolve()
root = config_path.parent.parent
config = load_config(config_path)
participants_path = Path(str(config["participants"]))
if not participants_path.is_absolute():
    participants_path = root / participants_path
participants = load_participants(participants_path, root)
if not participants:
    raise SystemExit("participants.tsv为空")
Path(sys.argv[2]).write_text(participants[0].participant_id + "\n", encoding="utf-8")
PY

PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
  --snakefile "${snakefile}" --configfile "${config}" --list \
  --config \
    "active_config_file=${config}" selected_participant=all profile=gpu gpu_devices=0 \
    gpu_slots_per_device=2 \
    "rolling_order_file=${validation_order}" \
    "rolling_token_dir=${archive_dir}/validation-tokens" \
    rolling_window=2 rolling_poll_seconds=1 \
  > "${archive_dir}/snakemake-rules.txt"
grep -qx 'rolling_completion' "${archive_dir}/snakemake-rules.txt"

PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
  --snakefile "${snakefile}" --configfile "${config}" --cores 1 --unlock \
  --config "active_config_file=${config}" selected_participant=all profile=gpu \
    gpu_devices=0 gpu_slots_per_device=2

trap - EXIT
patched=2
bash "${start_script}"
sleep 3
new_pid="$(tr -d '[:space:]' < "${pid_file}")"
new_pgid="$(ps -o pgid= -p "${new_pid}" | tr -d ' ')"
new_cwd="$(readlink -f "/proc/${new_pid}/cwd")"
if [[ ! "${new_pid}" =~ ^[1-9][0-9]*$ || "${new_pid}" != "${new_pgid}" || "${new_cwd}" != "${root}" ]]; then
  echo "重启验收失败: PID=${new_pid}, PGID=${new_pgid}, cwd=${new_cwd}" >&2
  exit 1
fi
log_path="$(tr -d '[:space:]' < "${root}/logs/full_run_v1.0.9.logpath")"
marker_found=0
for _ in $(seq 1 60); do
  if ! kill -0 "${new_pid}" 2>/dev/null; then
    echo "新主进程提前退出；日志: ${log_path}" >&2
    tail -n 100 "${log_path}" >&2 || true
    exit 1
  fi
  if grep -q 'gpu_slots=2' "${log_path}" 2>/dev/null; then
    marker_found=1
    break
  fi
  sleep 1
done

echo "部署并重启完成: PID=${new_pid}, PGID=${new_pgid}"
echo "物理设备=GPU0；逻辑槽位=2；T1=每例1槽；WMH=2槽独占"
echo "CPU总预算=96；普通CPU重任务=4线程；registration=8线程；滚动窗口=200"
if [[ "${marker_found}" -eq 1 ]]; then
  echo "启动日志资源标记: PASS"
else
  echo "启动日志尚未出现gpu_slots=2，请继续查看日志；主进程仍存活。"
fi
echo "日志: ${log_path}"
echo "部署前归档: ${archive_dir}"
