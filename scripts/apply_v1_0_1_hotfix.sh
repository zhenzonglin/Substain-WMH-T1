#!/usr/bin/env bash
set -euo pipefail

# 将GitHub中的V1.0.1小型修正覆盖到已有完整离线项目；不复制或修改原始影像。
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -ne 1 ]]; then
  echo "用法: $0 /现有完整Substain项目绝对路径" >&2
  exit 2
fi

requested_target="$1"
if [[ -L "${requested_target}" ]]; then
  echo "目标不能是符号链接: ${requested_target}" >&2
  exit 2
fi
target_root="$(realpath -e -- "${requested_target}")"
if [[ ! -d "${target_root}" ]]; then
  echo "目标必须是已存在且不是符号链接的目录: ${target_root}" >&2
  exit 2
fi
if [[ "${target_root}" == "/" || "${target_root}" == "${HOME}" || "${target_root}" == "${source_root}" ]]; then
  echo "拒绝更新不安全或与补丁源相同的目标: ${target_root}" >&2
  exit 2
fi
for sentinel in run_pipeline.sh pyproject.toml envs/offline/environment_archives.sha256 resources/tools/offline-smoke-image.tar; do
  if [[ ! -e "${target_root}/${sentinel}" ]]; then
    echo "目标不是完整离线Substain项目，缺少: ${sentinel}" >&2
    exit 2
  fi
done

hotfix_files=(
  "00_make_metadata_c3.py"
  "README.md"
  "docs/OFFLINE_TRANSFER_ZH.md"
  "run_pipeline.sh"
  "scripts/apply_v1_0_1_hotfix.sh"
  "scripts/build_offline_bundle.sh"
  "scripts/build_offline_smoke_image.sh"
  "scripts/offline_smoke.sh"
  "scripts/steps/06_offline_check.sh"
  "scripts/verify_offline_package.py"
  "src/substain_features/cli.py"
  "src/substain_features/resources.py"
  "tests/test_cli.py"
  "tests/test_resources.py"
  "wheels/core/substain_features-1.0.0-py3-none-any.whl"
)
for relative in "${hotfix_files[@]}"; do
  if [[ ! -f "${source_root}/${relative}" ]]; then
    echo "补丁源缺少文件: ${relative}" >&2
    exit 2
  fi
done

expected_wheel_sha256="906dcfff2b57571bdea421b6a49894a71389ab656f3dba67147ef19310bb0453"
observed_wheel_sha256="$(sha256sum "${source_root}/wheels/core/substain_features-1.0.0-py3-none-any.whl" | awk '{print $1}')"
if [[ "${observed_wheel_sha256}" != "${expected_wheel_sha256}" ]]; then
  echo "补丁wheel的SHA256不正确: ${observed_wheel_sha256}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_root="${target_root}/archive/hotfix-v1.0.1-backup-${timestamp}-${BASHPID}"
mkdir -p "${target_root}/archive"
mkdir "${backup_root}"

for relative in "${hotfix_files[@]}"; do
  source_path="${source_root}/${relative}"
  target_path="${target_root}/${relative}"
  if [[ -e "${target_path}" ]]; then
    mkdir -p "${backup_root}/$(dirname "${relative}")"
    cp -a -- "${target_path}" "${backup_root}/${relative}"
  fi
  mkdir -p "$(dirname "${target_path}")"
  cp -- "${source_path}" "${target_path}"
done
chmod +x \
  "${target_root}/run_pipeline.sh" \
  "${target_root}/scripts/apply_v1_0_1_hotfix.sh" \
  "${target_root}/scripts/build_offline_bundle.sh" \
  "${target_root}/scripts/build_offline_smoke_image.sh" \
  "${target_root}/scripts/offline_smoke.sh" \
  "${target_root}/scripts/steps/06_offline_check.sh" \
  "${target_root}/scripts/verify_offline_package.py"

manifest="${target_root}/derivatives/substain_features/tables/resource_manifest.tsv"
if [[ -f "${manifest}" ]]; then
  mkdir -p "${backup_root}/derivatives/substain_features/tables"
  cp -a -- "${manifest}" "${backup_root}/derivatives/substain_features/tables/resource_manifest.tsv"
fi

bash -n \
  "${target_root}/run_pipeline.sh" \
  "${target_root}/scripts/apply_v1_0_1_hotfix.sh" \
  "${target_root}/scripts/build_offline_bundle.sh" \
  "${target_root}/scripts/build_offline_smoke_image.sh" \
  "${target_root}/scripts/offline_smoke.sh" \
  "${target_root}/scripts/steps/06_offline_check.sh"
(cd "${target_root}/wheels/core" && echo "${expected_wheel_sha256}  substain_features-1.0.0-py3-none-any.whl" | sha256sum -c -)

# install_offline会在core、WMH和T1环境中强制重装同一份项目wheel，不访问网络。
"${target_root}/scripts/install_offline.sh"
SUBSTAIN_PROJECT_ROOT="${target_root}" PYTHONPATH="${target_root}/src" \
  "${target_root}/envs/core-venv/bin/python" -c \
  "import os; from pathlib import Path; from substain_features.resources import build_manifest; root=Path(os.environ['SUBSTAIN_PROJECT_ROOT']); table=build_manifest(root, root/'derivatives/substain_features/tables/resource_manifest.tsv'); print('resource_manifest_rows', len(table))"
"${target_root}/run_pipeline.sh" offline

echo "V1.0.1热修复已应用。旧文件备份: ${backup_root}"
echo "下一步运行: cd '${target_root}' && sudo -v && ./run_pipeline.sh offline-smoke --container-command 'sudo safe_docker.sh'"
