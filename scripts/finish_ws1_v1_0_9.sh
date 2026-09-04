#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cores="${1:-96}"
cpu_threads="${CPU_THREADS_PER_JOB:-4}"
batch_size="${BATCH_SIZE:-200}"
manifest_pointer="${root}/logs/ws1_v1.0.9_rerun_manifest.path"
config="${root}/config/config.yaml"
core_python="${root}/envs/core-venv/bin/python"
snakefile="${root}/workflow/Snakefile"
derivatives="${root}/derivatives/substain_features"
plan_dir="${root}/logs/ws1_v1.0.9_queue"

for value_name in cores cpu_threads batch_size; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name}必须为正整数: ${value}" >&2
    exit 2
  fi
done
if [[ ! -s "${manifest_pointer}" ]]; then
  echo "缺少重排队清单指针: ${manifest_pointer}" >&2
  exit 2
fi
manifest="$(tr -d '[:space:]' < "${manifest_pointer}")"
if [[ ! -f "${manifest}" ]]; then
  echo "重排队清单不存在: ${manifest}" >&2
  exit 2
fi

mkdir -p "${plan_dir}"
cd "${root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpu_threads}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${cpu_threads}}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-${cpu_threads}}"

PYTHONPATH="${root}/src" "${core_python}" - \
  "${manifest}" "${config}" "${derivatives}" "${plan_dir}" <<'PY'
import json
import re
import sys
from pathlib import Path

from substain_features.schema import load_config, load_participants

manifest_path = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
derivatives = Path(sys.argv[3]).resolve()
plan_dir = Path(sys.argv[4]).resolve()
root = config_path.parent.parent.resolve()
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if Path(str(payload.get("project_root", ""))).resolve() != root:
    raise SystemExit("清单project_root与当前项目不一致")

valid_id = re.compile(r"^[A-Za-z0-9._-]+$")
grid = [str(value) for value in payload.get("grid_failure_participants", [])]
representatives = [str(value) for value in payload.get("representative_participants", [])]
interrupted = [str(value) for value in payload.get("interrupted_participants", [])]
for participant_id in grid + representatives + interrupted:
    if not valid_id.fullmatch(participant_id):
        raise SystemExit("清单含非法participant_id: {}".format(participant_id))
if not set(representatives).issubset(grid):
    raise SystemExit("代表病例不属于网格失败名单")

def cleanup_target(participant_id):
    return derivatives / "sub-{}".format(participant_id) / "status" / "cleanup.json"

def needs_run(participant_id):
    return not cleanup_target(participant_id).is_file()

stage_order = ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc")

def progress_key(participant_id):
    status_dir = derivatives / "sub-{}".format(participant_id) / "status"
    completed = []
    for index, stage in enumerate(stage_order):
        path = status_dir / "{}.json".format(stage)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise SystemExit("无法解析状态文件: {}".format(path))
        if payload.get("status") == "pass":
            completed.append(index)
    deepest = max(completed, default=-1)
    return (-deepest, -len(completed), participant_id)

def closest_first(participant_ids):
    return sorted(participant_ids, key=progress_key)

representatives = closest_first([value for value in representatives if needs_run(value)])
grid_rest = closest_first([value for value in grid if value not in representatives and needs_run(value)])
grid_ids = set(grid)
interrupted = closest_first(
    [value for value in interrupted if value not in grid_ids and needs_run(value)]
)
priority = set(grid) | set(interrupted)

config = load_config(config_path)
participant_path = Path(str(config["participants"]))
if not participant_path.is_absolute():
    participant_path = root / participant_path
all_participants = load_participants(participant_path, root)
untouched = []
skipped_historical_failures = []
for participant in all_participants:
    participant_id = participant.participant_id
    if participant_id in priority or not needs_run(participant_id):
        continue
    status_dir = derivatives / "sub-{}".format(participant_id) / "status"
    has_real_failure = False
    for status_path in status_dir.glob("*.json") if status_dir.is_dir() else ():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise SystemExit("无法解析状态文件: {}".format(status_path))
        if status.get("status") == "fail":
            has_real_failure = True
            break
    if has_real_failure:
        skipped_historical_failures.append(participant_id)
    else:
        untouched.append(participant_id)

untouched = closest_first(untouched)
rolling = []
seen = set()
for participant_id in grid_rest + interrupted + untouched:
    if participant_id not in seen:
        rolling.append(participant_id)
        seen.add(participant_id)

phases = {
    "representative": representatives,
    "grid": grid_rest,
    "interrupted": interrupted,
    "untouched": untouched,
    "rolling": rolling,
}
for name, participant_ids in phases.items():
    (plan_dir / "{}.targets".format(name)).write_text(
        "".join("{}\n".format(cleanup_target(value)) for value in participant_ids),
        encoding="utf-8",
    )
    (plan_dir / "{}.participants".format(name)).write_text(
        "".join("{}\n".format(value) for value in participant_ids), encoding="utf-8"
    )
(plan_dir / "skipped_historical_failures.participants").write_text(
    "".join("{}\n".format(value) for value in skipped_historical_failures), encoding="utf-8"
)
print("优先队列: representatives={}, grid={}, interrupted={}, untouched={}, historical_failures_skipped={}".format(
    len(representatives), len(grid_rest), len(interrupted), len(untouched), len(skipped_historical_failures)
))
print("滚动队列: total={}, order=grid->interrupted->untouched, closest_stage_first=true".format(len(rolling)))
PY

run_wave() {
  local phase="$1"
  local wave="$2"
  local wave_count="$3"
  shift 3
  local targets=("$@")
  local missing=0
  local target
  echo "[$(date -Is)] 开始${phase} ${wave}/${wave_count}: ${#targets[@]}例"
  PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
    --snakefile "${snakefile}" \
    --configfile "${config}" \
    --cores "${cores}" \
    --keep-going \
    --printshellcmds \
    "${targets[@]}" \
    --resources "gpu=1" \
    --config \
      "active_config_file=${config}" \
      "selected_participant=all" \
      "profile=gpu" \
      "gpu_devices=0"
  for target in "${targets[@]}"; do
    if [[ ! -f "${target}" ]]; then
      echo "波次结束但缺少cleanup状态: ${target}" >&2
      missing=$((missing + 1))
    fi
  done
  if (( missing > 0 )); then
    echo "${phase} ${wave}/${wave_count} 缺少${missing}例cleanup，停止开放下一波。" >&2
    return 1
  fi
  echo "[$(date -Is)] 完成${phase} ${wave}/${wave_count}"
}

run_phase() {
  local phase="$1"
  local file="$2"
  local phase_batch="$3"
  local targets=()
  mapfile -t targets < "${file}"
  local count=${#targets[@]}
  local waves=0
  if (( count > 0 )); then
    waves=$(((count + phase_batch - 1) / phase_batch))
  fi
  echo "阶段${phase}: count=${count}, batch_size=${phase_batch}, waves=${waves}"
  local offset wave
  for ((offset = 0, wave = 1; offset < count; offset += phase_batch, wave += 1)); do
    run_wave "${phase}" "${wave}" "${waves}" "${targets[@]:offset:phase_batch}"
  done
}

run_rolling() {
  local targets_file="$1"
  local participants_file="$2"
  local window="$3"
  local targets=()
  local participants=()
  mapfile -t targets < "${targets_file}"
  mapfile -t participants < "${participants_file}"
  if (( ${#targets[@]} != ${#participants[@]} )); then
    echo "滚动队列targets与participants数量不一致" >&2
    return 2
  fi
  if (( ${#targets[@]} == 0 )); then
    echo "滚动队列为空，无需启动。"
    return 0
  fi

  local run_tag token_dir completion_target missing target
  run_tag="$(date +%Y%m%d_%H%M%S)-$$"
  token_dir="${plan_dir}/admission-${run_tag}"
  "${core_python}" "${root}/scripts/rolling_admission.py" initialize \
    --order-file "${participants_file}" \
    --token-dir "${token_dir}" \
    --window "${window}"
  completion_target="${token_dir}/all_complete.json"

  echo "[$(date -Is)] 启动滚动队列: total=${#targets[@]}, window=${window}, cores=${cores}, gpu=1"
  PYTHONPATH="${root}/src" "${core_python}" -m snakemake \
    --snakefile "${snakefile}" \
    --configfile "${config}" \
    --cores "${cores}" \
    --keep-going \
    --printshellcmds \
    "${completion_target}" \
    --resources "gpu=1" \
    --config \
      "active_config_file=${config}" \
      "selected_participant=all" \
      "profile=gpu" \
      "gpu_devices=0" \
      "rolling_order_file=${participants_file}" \
      "rolling_token_dir=${token_dir}" \
      "rolling_window=${window}" \
      "rolling_poll_seconds=10"

  missing=0
  for target in "${targets[@]}"; do
    if [[ ! -f "${target}" ]]; then
      echo "滚动队列结束但缺少cleanup状态: ${target}" >&2
      missing=$((missing + 1))
    fi
  done
  if (( missing > 0 )); then
    echo "滚动队列缺少${missing}例cleanup状态。" >&2
    return 1
  fi
  echo "[$(date -Is)] 滚动队列完成: ${#targets[@]}例"
}

echo "正式资源: cores=${cores}, threads_per_job=${cpu_threads}, batch_size=${batch_size}, gpu=${CUDA_VISIBLE_DEVICES}"
run_phase "代表病例" "${plan_dir}/representative.targets" 3

PYTHONPATH="${root}/src" "${core_python}" - "${plan_dir}/representative.participants" "${derivatives}" <<'PY'
import json
import sys
from pathlib import Path

participants_file = Path(sys.argv[1])
derivatives = Path(sys.argv[2])
for participant_id in participants_file.read_text(encoding="utf-8").splitlines():
    status_dir = derivatives / "sub-{}".format(participant_id) / "status"
    wmh = json.loads((status_dir / "wmh.json").read_text(encoding="utf-8"))
    cleanup = json.loads((status_dir / "cleanup.json").read_text(encoding="utf-8"))
    if wmh.get("status") != "pass" or cleanup.get("status") != "pass":
        raise SystemExit("代表病例未通过WMH/cleanup: {}".format(participant_id))
    details = wmh.get("details", {})
    grid = details.get("wmh_atlas_grid", {})
    if not grid.get("matches") or float(grid.get("max_corner_displacement_mm", 1.0)) >= 0.05:
        raise SystemExit("代表病例网格诊断未通过: {} -> {}".format(participant_id, grid))
    runtime = details.get("runtime", {})
    if runtime.get("duration_seconds") is None:
        raise SystemExit("代表病例缺少WMH runtime: {}".format(participant_id))
    feature_path = Path(str(details.get("feature_json", "")))
    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    if len(feature.get("raw_ml", {})) != 20:
        raise SystemExit("代表病例不是20个WMH特征: {}".format(participant_id))
print("代表病例验收通过")
PY

run_rolling "${plan_dir}/rolling.targets" "${plan_dir}/rolling.participants" "${batch_size}"

echo "[$(date -Is)] 滚动队列已全部产生cleanup状态，执行一次最终聚合；未运行audit。"
PYTHONPATH="${root}/src" "${core_python}" -m substain_features.cli stage aggregate \
  --config-file "${config}" --participant-id all
