#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cores="${1:-96}"
cpu_threads="${CPU_THREADS_PER_JOB:-8}"

if [[ ! "${cores}" =~ ^[1-9][0-9]*$ ]]; then
  echo "cores必须为正整数: ${cores}" >&2
  exit 2
fi
if [[ ! "${cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS_PER_JOB必须为正整数: ${cpu_threads}" >&2
  exit 2
fi

config="${root}/config/config.yaml"
core_python="${root}/envs/core-venv/bin/python"
snakefile="${root}/workflow/Snakefile"
derivatives="${root}/derivatives/substain_features"
targets_file="${root}/logs/backlog_cleanup_targets_v1.0.5.txt"
participants_file="${root}/logs/backlog_participants_v1.0.5.txt"
archive_root="${root}/archive/interrupted-status-before-backlog-$(date +%Y%m%d_%H%M%S)"

mkdir -p "${root}/logs"
cd "${root}"

if (( cores >= 3 * cpu_threads )); then
  skullstrip_slots=2
else
  skullstrip_slots=1
fi
reserved_threads=$((skullstrip_slots * cpu_threads))
finish_cpu_slots=$(((cores - reserved_threads) / cpu_threads))
if (( finish_cpu_slots < 1 )); then
  finish_cpu_slots=1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpu_threads}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${cpu_threads}}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-${cpu_threads}}"

"${core_python}" - "${derivatives}" "${targets_file}" "${participants_file}" "${archive_root}" <<'PY'
import json
import shutil
import sys
from pathlib import Path


derivatives = Path(sys.argv[1]).resolve()
targets_file = Path(sys.argv[2]).resolve()
participants_file = Path(sys.argv[3]).resolve()
archive_root = Path(sys.argv[4]).resolve()

if not derivatives.is_dir():
    raise SystemExit("衍生结果目录不存在: {}".format(derivatives))


def load_status(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_pass(path):
    status = load_status(path)
    return bool(status and status.get("status") == "pass")


# 仅归档明确由中断产生的fail状态；真实分析失败保持原样。
interruption_markers = (
    "exit=-15",
    "exit = -15",
    "sigterm",
    "keyboardinterrupt",
    "terminated",
    "signal 15",
    "returncode -15",
)
archived = 0
for status_path in sorted(derivatives.glob("sub-*/status/*.json")):
    status = load_status(status_path)
    if not status or status.get("status") != "fail":
        continue
    text = json.dumps(status, ensure_ascii=False).lower()
    if not any(marker in text for marker in interruption_markers):
        continue
    destination = archive_root / status_path.relative_to(derivatives)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(status_path), str(destination))
    archived += 1

targets = []
participants = []
for subject_dir in sorted(derivatives.glob("sub-*")):
    status_dir = subject_dir / "status"
    if not is_pass(status_dir / "skullstrip.json"):
        continue
    if not is_pass(status_dir / "wmh_seg.json"):
        continue
    cleanup = status_dir / "cleanup.json"
    if cleanup.exists():
        continue
    targets.append(str(cleanup))
    participants.append(subject_dir.name[4:])

targets_file.write_text(
    "".join("{}\n".format(target) for target in targets),
    encoding="utf-8",
)
participants_file.write_text(
    "".join("{}\n".format(participant) for participant in participants),
    encoding="utf-8",
)
print("归档的明确中断fail状态数: {}".format(archived))
print("冻结积压病例数: {}".format(len(targets)))
print("积压名单: {}".format(participants_file))
PY

mapfile -t targets < "${targets_file}"
echo "积压阶段资源: cores=${cores}, finish_cpu=${finish_cpu_slots}, skullstrip_cpu=${skullstrip_slots}, gpu=1"

if [[ "${BACKLOG_LIST_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if (( ${#targets[@]} > 0 )); then
  PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
    --snakefile "${snakefile}" \
    --configfile "${config}" \
    --cores "${cores}" \
    --keep-going \
    --printshellcmds \
    "${targets[@]}" \
    --resources \
    "gpu=1" \
    "finish_cpu=${finish_cpu_slots}" \
    "skullstrip_cpu=${skullstrip_slots}" \
    --config \
    "active_config_file=${config}" \
    "selected_participant=all" \
    "profile=gpu" \
    "gpu_devices=0"
fi

echo "冻结积压病例已产生完整状态链，开始9213例全量断点续跑。"
exec "${root}/run_pipeline.sh" run all gpu "${cores}"
