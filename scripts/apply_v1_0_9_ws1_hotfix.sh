#!/usr/bin/env bash
set -euo pipefail

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
active_arg="${1:-/data/usersdir/linzhenzong/Substain}"
if [[ ! -d "${active_arg}" ]]; then
  echo "活动项目目录不存在: ${active_arg}" >&2
  exit 2
fi
active_root="$(cd "${active_arg}" && pwd -P)"
if [[ "${active_root}" == "${source_root}" ]]; then
  echo "热修复clone不能与活动项目相同" >&2
  exit 2
fi
patch_file="${source_root}/deploy/ws1_v1_0_9_source.patch"
core_python="${active_root}/envs/core-venv/bin/python"
if [[ ! -x "${core_python}" || ! -f "${patch_file}" ]]; then
  echo "缺少活动Python或热修复补丁" >&2
  exit 2
fi
if ! grep -q 'error_records_only_v1' "${active_root}/src/substain_features/pipeline.py"; then
  echo "现场代码未检测到error_records_only_v1；为避免覆盖未知cleanup实现，停止部署。" >&2
  exit 2
fi
if ! grep -q '^def stage_skullstrip' "${active_root}/src/substain_features/pipeline.py"; then
  echo "现场pipeline.py不是预期的完整工作站1实现；停止部署。" >&2
  exit 2
fi
if grep -q '^def physical_grid_diagnostics' "${active_root}/src/substain_features/images.py"; then
  echo "现场似乎已应用v1.0.9物理网格补丁；停止重复应用。" >&2
  exit 2
fi

source_files=(
  "src/substain_features/images.py"
  "src/substain_features/status.py"
  "src/substain_features/wmh.py"
  "src/substain_features/pipeline.py"
)
new_scripts=(
  "scripts/prepare_ws1_v1_0_9_rerun.py"
  "scripts/finish_ws1_v1_0_9.sh"
  "scripts/start_ws1_v1_0_9.sh"
  "scripts/stop_ws1_v1_0_8.sh"
  "scripts/monitor_ws1.py"
)
for relative in "${source_files[@]}"; do
  target="${active_root}/${relative}"
  if [[ ! -f "${target}" || -L "${target}" ]]; then
    echo "活动源码不是普通文件: ${target}" >&2
    exit 2
  fi
done
for relative in "${new_scripts[@]}"; do
  source_file="${source_root}/${relative}"
  if [[ ! -f "${source_file}" || -L "${source_file}" ]]; then
    echo "clone脚本不是普通文件: ${source_file}" >&2
    exit 2
  fi
done

if ! (cd "${active_root}" && git apply --check "${patch_file}"); then
  echo "现场源码上下文与补丁不匹配；未修改任何文件。" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_root="${active_root}/archive/hotfix-v1.0.9-backup-${stamp}"
if [[ -e "${backup_root}" ]]; then
  echo "备份目录已存在: ${backup_root}" >&2
  exit 2
fi
mkdir -p "${backup_root}"
manifest="${backup_root}/manifest.tsv"
printf 'kind\trelative_path\tsize_bytes\tsha256\n' > "${manifest}"
source_commit="$(git -C "${source_root}" rev-parse HEAD)"
printf 'metadata\tsource_commit\t0\t%s\n' "${source_commit}" >> "${manifest}"

backup_file() {
  local relative="$1"
  local source_file="${active_root}/${relative}"
  local destination="${backup_root}/${relative}"
  mkdir -p "$(dirname "${destination}")"
  cp -p "${source_file}" "${destination}"
  printf 'backup\t%s\t%s\t%s\n' \
    "${relative}" "$(stat -c %s "${source_file}")" "$(sha256sum "${source_file}" | awk '{print $1}')" \
    >> "${manifest}"
}

for relative in "${source_files[@]}"; do
  backup_file "${relative}"
done
for relative in "${new_scripts[@]}"; do
  if [[ -e "${active_root}/${relative}" ]]; then
    if [[ ! -f "${active_root}/${relative}" || -L "${active_root}/${relative}" ]]; then
      echo "现有脚本不是普通文件: ${active_root}/${relative}" >&2
      exit 2
    fi
    backup_file "${relative}"
  fi
done

restore_backup() {
  echo "部署验证失败，正在从 ${backup_root} 恢复源码。" >&2
  local relative
  for relative in "${source_files[@]}"; do
    cp -p "${backup_root}/${relative}" "${active_root}/${relative}"
  done
  for relative in "${new_scripts[@]}"; do
    if [[ -f "${backup_root}/${relative}" ]]; then
      cp -p "${backup_root}/${relative}" "${active_root}/${relative}"
    else
      rm -f -- "${active_root}/${relative}"
    fi
  done
}

rollback_on_error() {
  local exit_code=$?
  trap - ERR
  set +e
  restore_backup
  exit "${exit_code}"
}

trap rollback_on_error ERR
(cd "${active_root}" && git apply "${patch_file}")
for relative in "${new_scripts[@]}"; do
  install -m 0755 "${source_root}/${relative}" "${active_root}/${relative}"
done

if ! "${core_python}" -m py_compile \
  "${active_root}/src/substain_features/images.py" \
  "${active_root}/src/substain_features/status.py" \
  "${active_root}/src/substain_features/wmh.py" \
  "${active_root}/src/substain_features/pipeline.py" \
  "${active_root}/scripts/prepare_ws1_v1_0_9_rerun.py" \
  "${active_root}/scripts/monitor_ws1.py"; then
  trap - ERR
  restore_backup
  exit 1
fi

for relative in "${source_files[@]}" "${new_scripts[@]}"; do
  target="${active_root}/${relative}"
  printf 'installed\t%s\t%s\t%s\n' \
    "${relative}" "$(stat -c %s "${target}")" "$(sha256sum "${target}" | awk '{print $1}')" \
    >> "${manifest}"
done

trap - ERR
echo "v1.0.9热修复已应用；备份与校验清单: ${manifest}"
echo "热修复commit: ${source_commit}"
echo "下一步先dry-run: ${core_python} ${active_root}/scripts/prepare_ws1_v1_0_9_rerun.py --project-root ${active_root}"
