#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cores="${1:-96}"
cpu_threads="${CPU_THREADS_PER_JOB:-8}"
batch_size="${BATCH_SIZE:-200}"

if [[ ! "${cores}" =~ ^[1-9][0-9]*$ ]]; then
  echo "cores必须为正整数: ${cores}" >&2
  exit 2
fi
if [[ ! "${cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS_PER_JOB必须为正整数: ${cpu_threads}" >&2
  exit 2
fi
if [[ ! "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE必须为正整数: ${batch_size}" >&2
  exit 2
fi

config="${root}/config/config.yaml"
core_python="${root}/envs/core-venv/bin/python"
snakefile="${root}/workflow/Snakefile"
derivatives="${root}/derivatives/substain_features"
targets_file="${root}/logs/backlog_cleanup_targets_v1.0.5.txt"
participants_file="${root}/logs/backlog_participants_v1.0.5.txt"
remaining_targets_file="${root}/logs/wave_cleanup_targets_v1.0.6.txt"
remaining_participants_file="${root}/logs/wave_participants_v1.0.6.txt"
archive_root="${root}/archive/interrupted-status-before-backlog-$(date +%Y%m%d_%H%M%S)"

mkdir -p "${root}/logs"
cd "${root}"

if (( cores >= 3 * cpu_threads )); then
  skullstrip_slots=2
else
  skullstrip_slots=1
fi
t1_slots=1
if (( cores >= 2 * cpu_threads )); then
  t1_slots=2
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
    # 冻结所有已有任一成功阶段、但尚未完成cleanup的病例；包括只有
    # skullstrip、只有T1或已完成WMH-SynthSeg的旧积压，不再只挑某一种形态。
    progress_stages = ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc")
    if not any(is_pass(status_dir / "{}.json".format(stage)) for stage in progress_stages):
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
echo "积压阶段资源: cores=${cores}, finish_cpu=${finish_cpu_slots}, skullstrip_cpu=${skullstrip_slots}, t1_cpu=${t1_slots}, gpu=1"

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
    "t1_cpu=${t1_slots}" \
    --config \
    "active_config_file=${config}" \
    "selected_participant=all" \
    "profile=gpu" \
    "gpu_devices=0"
fi

echo "冻结积压病例已产生完整状态链；生成剩余病例的固定波次名单。"

PYTHONPATH="${root}/src" "${core_python}" - \
  "${config}" \
  "${derivatives}" \
  "${remaining_targets_file}" \
  "${remaining_participants_file}" <<'PY'
import sys
from pathlib import Path

from substain_features.schema import load_config, load_participants


config_path = Path(sys.argv[1]).resolve()
derivatives = Path(sys.argv[2]).resolve()
targets_file = Path(sys.argv[3]).resolve()
participants_file = Path(sys.argv[4]).resolve()

config = load_config(config_path)
project_root = Path(str(config["project_root"]))
participant_path = Path(str(config["participants"]))
if not participant_path.is_absolute():
    participant_path = project_root / participant_path
participants = load_participants(participant_path, project_root)

targets = []
participant_ids = []
for participant in participants:
    status_dir = derivatives / "sub-{}".format(participant.participant_id) / "status"
    cleanup = status_dir / "cleanup.json"
    # pass和真实fail状态都视为已处理；只有缺失状态才进入后续波次。
    if cleanup.exists():
        continue
    targets.append(str(cleanup))
    participant_ids.append(participant.participant_id)

targets_file.write_text(
    "".join("{}\n".format(target) for target in targets),
    encoding="utf-8",
)
participants_file.write_text(
    "".join("{}\n".format(participant_id) for participant_id in participant_ids),
    encoding="utf-8",
)
print("冻结积压完成后的剩余病例数: {}".format(len(targets)))
print("剩余病例名单: {}".format(participants_file))
PY

mapfile -t remaining_targets < "${remaining_targets_file}"
remaining_count=${#remaining_targets[@]}
if (( remaining_count > 0 )); then
  wave_count=$(((remaining_count + batch_size - 1) / batch_size))
else
  wave_count=0
fi
echo "200例波次阶段: batch_size=${batch_size}, remaining=${remaining_count}, waves=${wave_count}"

for ((offset = 0, wave = 1; offset < remaining_count; offset += batch_size, wave += 1)); do
  wave_targets=("${remaining_targets[@]:offset:batch_size}")
  echo "开始波次 ${wave}/${wave_count}: ${#wave_targets[@]}例；本波全部产生cleanup状态后才进入下一波。"
  PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
    --snakefile "${snakefile}" \
    --configfile "${config}" \
    --cores "${cores}" \
    --keep-going \
    --printshellcmds \
    "${wave_targets[@]}" \
    --resources \
    "gpu=1" \
    "finish_cpu=${finish_cpu_slots}" \
    "skullstrip_cpu=${skullstrip_slots}" \
    "t1_cpu=${t1_slots}" \
    --config \
    "active_config_file=${config}" \
    "selected_participant=all" \
    "profile=gpu" \
    "gpu_devices=0"
  echo "完成波次 ${wave}/${wave_count}。"
done

echo "所有固定波次均已完成，执行全队列最终聚合；已有状态不会重算。"
exec "${root}/run_pipeline.sh" run all gpu "${cores}"
