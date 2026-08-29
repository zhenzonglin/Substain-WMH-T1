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
targets_file="${root}/logs/backlog_cleanup_targets_v1.0.8.txt"
participants_file="${root}/logs/backlog_participants_v1.0.8.txt"
remaining_targets_file="${root}/logs/wave_cleanup_targets_v1.0.8.txt"
remaining_participants_file="${root}/logs/wave_participants_v1.0.8.txt"
archive_root="${root}/archive/interrupted-status-before-v1.0.8-$(date +%Y%m%d_%H%M%S)"

mkdir -p "${root}/logs"
cd "${root}"

# CPU重任务不再划分固定阶段槽；Snakemake依据总核心数、每例线程数和优先级
# 动态填充空闲核心。GPU仍由唯一资源令牌和进程锁串行化。

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

# 已有任一有效阶段状态的病例均属于积压。按缺失阶段数、最深阶段、ID排序，
# 使最接近cleanup的病例先进入固定波次。pass和真实fail都算已物化状态。
progress_stages = ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc")
stage_depth = {stage: index for index, stage in enumerate(progress_stages, start=1)}
records = []
for subject_dir in sorted(derivatives.glob("sub-*")):
    status_dir = subject_dir / "status"
    cleanup = status_dir / "cleanup.json"
    if cleanup.exists():
        continue
    materialized = {
        stage: load_status(status_dir / "{}.json".format(stage))
        for stage in progress_stages
    }
    materialized = {stage: status for stage, status in materialized.items() if status is not None}
    if not materialized:
        continue
    missing_count = len(progress_stages) - len(materialized)
    deepest_stage = max(stage_depth[stage] for stage in materialized)
    participant_id = subject_dir.name[4:]
    records.append((missing_count, -deepest_stage, participant_id, str(cleanup)))

records.sort(key=lambda record: (record[0], record[1], record[2]))
targets_file.write_text(
    "".join("{}\n".format(record[3]) for record in records),
    encoding="utf-8",
)
participants_file.write_text(
    "".join("{}\n".format(record[2]) for record in records),
    encoding="utf-8",
)
print("归档的明确中断fail状态数: {}".format(archived))
print("排序后的积压病例数: {}".format(len(records)))
print("积压名单: {}".format(participants_file))
PY

mapfile -t backlog_targets < "${targets_file}"
backlog_count=${#backlog_targets[@]}
if (( backlog_count > 0 )); then
  backlog_wave_count=$(((backlog_count + batch_size - 1) / batch_size))
else
  backlog_wave_count=0
fi

echo "严格调度资源: cores=${cores}, heavy_threads=${cpu_threads}, cpu_custom_caps=none, gpu=1"
echo "积压严格波次: batch_size=${batch_size}, backlog=${backlog_count}, waves=${backlog_wave_count}"

print_wave_plan() {
  local phase="$1"
  local total="$2"
  local wave_count="$3"
  local offset wave size
  for ((offset = 0, wave = 1; offset < total; offset += batch_size, wave += 1)); do
    size=$((total - offset))
    if (( size > batch_size )); then
      size=${batch_size}
    fi
    echo "计划${phase}波次 ${wave}/${wave_count}: ${size}例"
  done
}

if [[ "${BACKLOG_LIST_ONLY:-0}" == "1" ]]; then
  print_wave_plan "积压" "${backlog_count}" "${backlog_wave_count}"
  exit 0
fi

run_wave() {
  local phase="$1"
  local wave="$2"
  local wave_count="$3"
  shift 3
  local wave_targets=("$@")
  local missing=0
  local target

  echo "[$(date -Is)] 开始${phase}波次 ${wave}/${wave_count}: ${#wave_targets[@]}例；全部产生cleanup状态后才进入下一波。"
  PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
    --snakefile "${snakefile}" \
    --configfile "${config}" \
    --cores "${cores}" \
    --keep-going \
    --printshellcmds \
    "${wave_targets[@]}" \
    --resources \
    "gpu=1" \
    --config \
    "active_config_file=${config}" \
    "selected_participant=all" \
    "profile=gpu" \
    "gpu_devices=0"

  for target in "${wave_targets[@]}"; do
    if [[ ! -f "${target}" ]]; then
      echo "波次结束但缺少cleanup状态: ${target}" >&2
      missing=$((missing + 1))
    fi
  done
  if (( missing > 0 )); then
    echo "${phase}波次 ${wave}/${wave_count} 未完整结束，停止开放下一波（缺少${missing}例）。" >&2
    return 1
  fi
  echo "[$(date -Is)] 完成${phase}波次 ${wave}/${wave_count}。"
}

for ((offset = 0, wave = 1; offset < backlog_count; offset += batch_size, wave += 1)); do
  backlog_wave_targets=("${backlog_targets[@]:offset:batch_size}")
  run_wave "积压" "${wave}" "${backlog_wave_count}" "${backlog_wave_targets[@]}"
done

echo "全部积压波次已产生cleanup状态；生成从未启动病例的固定波次名单。"

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
    # pass和真实fail状态都视为已处理；只有缺失cleanup状态才进入后续波次。
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
print("积压完成后的未处理病例数: {}".format(len(targets)))
print("未处理病例名单: {}".format(participants_file))
PY

mapfile -t remaining_targets < "${remaining_targets_file}"
remaining_count=${#remaining_targets[@]}
if (( remaining_count > 0 )); then
  remaining_wave_count=$(((remaining_count + batch_size - 1) / batch_size))
else
  remaining_wave_count=0
fi
echo "新病例严格波次: batch_size=${batch_size}, remaining=${remaining_count}, waves=${remaining_wave_count}"

for ((offset = 0, wave = 1; offset < remaining_count; offset += batch_size, wave += 1)); do
  remaining_wave_targets=("${remaining_targets[@]:offset:batch_size}")
  run_wave "新病例" "${wave}" "${remaining_wave_count}" "${remaining_wave_targets[@]}"
done

echo "[$(date -Is)] 所有固定波次均已完成，直接聚合现有状态；不再递归启动全量DAG。"
PYTHONPATH="${root}/src" "${core_python}" -m substain_features.cli stage aggregate \
  --config-file "${config}" \
  --participant-id all
