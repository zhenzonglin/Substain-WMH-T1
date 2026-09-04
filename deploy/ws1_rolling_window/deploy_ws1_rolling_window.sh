#!/usr/bin/env bash
set -euo pipefail

root="${1:-/data/usersdir/linzhenzong/Substain}"
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
patch_file="${bundle_dir}/ws1_rolling_window.patch"
pid_file="${root}/logs/full_run_v1.0.9.pid"
config="${root}/config/config.yaml"
snakefile="${root}/workflow/Snakefile"
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
if [[ ! -f "${patch_file}" || ! -x "${core_python}" ]]; then
  echo "补丁或core Python缺失: ${patch_file}" >&2
  exit 2
fi

cd "${root}"
"${core_python}" - "${snakefile}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")

def rule(name):
    match = re.search(r"^rule {}:\n(?P<body>.*?)(?=^rule |\Z)".format(name), text, re.M | re.S)
    if match is None:
        raise SystemExit("缺少规则: {}".format(name))
    return match.group("body")

t1 = rule("t1")
registration = rule("registration")
if 'gpu=1 if PROFILE == "gpu" else 0' not in t1 or "needs_gpu=True" not in t1:
    raise SystemExit("T1当前不是GPU规则，停止部署")
if "threads: 8" not in registration or "needs_gpu=False" not in registration:
    raise SystemExit("registration当前不是CPU 8线程规则，停止部署")
if "ROLLING_ORDER_VALUE" in text:
    raise SystemExit("滚动窗口补丁似乎已经应用，停止重复部署")
print("现场规则核对: T1 GPU=PASS, registration CPU8=PASS")
PY

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
archive_dir="${root}/archive/rolling-window-${run_tag}"
if [[ -e "${archive_dir}" ]]; then
  echo "归档目录已存在: ${archive_dir}" >&2
  exit 2
fi
mkdir -p "${archive_dir}"
cp -a "${snakefile}" "${archive_dir}/Snakefile.before"
cp -a "${root}/scripts/finish_ws1_v1_0_9.sh" "${archive_dir}/finish_ws1_v1_0_9.sh.before"
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
        parts = (proc / "cmdline").read_bytes().split(b"\0")
        args = [value.decode("utf-8", "replace") for value in parts if value]
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
  echo "TERM后仍有${remaining}个进程，未使用KILL，也未应用补丁。" >&2
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

git apply --whitespace=error "${patch_file}"
chmod 755 "${root}/scripts/rolling_admission.py" "${root}/scripts/finish_ws1_v1_0_9.sh"
"${core_python}" -m py_compile "${root}/scripts/rolling_admission.py"
bash -n "${root}/scripts/finish_ws1_v1_0_9.sh"
bash -n "${root}/scripts/start_ws1_v1_0_9.sh"

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
    "rolling_order_file=${validation_order}" \
    "rolling_token_dir=${archive_dir}/validation-tokens" \
    rolling_window=200 rolling_poll_seconds=10 \
  | grep -q '^rolling_completion$'

PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
  --snakefile "${snakefile}" --configfile "${config}" --cores 1 --unlock \
  --config "active_config_file=${config}" selected_participant=all profile=gpu gpu_devices=0

bash "${root}/scripts/start_ws1_v1_0_9.sh"
sleep 3
new_pid="$(tr -d '[:space:]' < "${pid_file}")"
new_pgid="$(ps -o pgid= -p "${new_pid}" | tr -d ' ')"
if [[ ! "${new_pid}" =~ ^[1-9][0-9]*$ || "${new_pid}" != "${new_pgid}" ]]; then
  echo "重启验收失败: PID=${new_pid}, PGID=${new_pgid}" >&2
  exit 1
fi
log_path="$(tr -d '[:space:]' < "${root}/logs/full_run_v1.0.9.logpath")"
echo "部署并重启完成: PID=${new_pid}, PGID=${new_pgid}"
echo "CPU总预算=96；普通CPU重任务=4线程；registration=8线程；GPU令牌=1"
echo "滚动窗口=200；完成1例cleanup后立即补入1例"
echo "日志: ${log_path}"
echo "归档: ${archive_dir}"
