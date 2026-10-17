#!/usr/bin/env bash
set -euo pipefail

# 禁网工作站：从项目 wheel 缓存重建 core，并从 conda-pack 归档恢复 WMH/T1。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
test -d "${project_root}/wheels/core"
test -f "${project_root}/envs/offline/wmh-env.tar.gz"
test -f "${project_root}/envs/offline/t1-env.tar.gz"
test -f "${project_root}/envs/offline/environment_archives.sha256"
# 解包前先验证两个大归档，避免传输损坏后得到表面可启动、实际不完整的环境。
(cd "${project_root}/envs/offline" && sha256sum -c environment_archives.sha256)
environment_root="${SUBSTAIN_OFFLINE_ENV_ROOT:-${project_root}/envs}"
mkdir -p "${environment_root}"

restore_packed_env() {
  local archive="$1"
  local target="$2"
  local wheel_dir="$3"
  if [[ -x "${target}/bin/python" ]]; then
    echo "保留已有环境：${target}"
  else
    if [[ -d "${target}" && -n "$(find "${target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "目标环境非空且不可验证，拒绝覆盖：${target}" >&2
      return 1
    fi
    mkdir -p "${target}"
    tar -xzf "${archive}" -C "${target}"
    # conda-unpack的shebang是/usr/bin/env python，需先让env找到刚恢复的解释器。
    PATH="${target}/bin:${PATH}" "${target}/bin/conda-unpack"
  fi
  PYTHONNOUSERSITE=1 "${target}/bin/python" -m pip install --no-index --no-deps --force-reinstall \
    --only-binary=:all: --find-links "${wheel_dir}" substain-features
  PYTHONNOUSERSITE=1 "${target}/bin/python" -m pip check
}

core_target="${environment_root}/core-venv"
if [[ ! -x "${core_target}/bin/python" ]]; then
  if [[ -d "${core_target}" && -n "$(find "${core_target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "core 目标非空且不可验证，拒绝覆盖：${core_target}" >&2
    exit 1
  fi
  python3 -m venv "${core_target}"
fi
# Ubuntu-20.04自带pip 20不识别manylinux_2_28标签；先用纯Python wheel离线升级安装器。
PYTHONNOUSERSITE=1 "${core_target}/bin/python" -m pip install --no-index --upgrade \
  --only-binary=:all: --find-links "${project_root}/wheels/core" pip setuptools wheel
PYTHONNOUSERSITE=1 "${core_target}/bin/python" -m pip install --no-index --force-reinstall \
  --only-binary=:all: --find-links "${project_root}/wheels/core" substain-features snakemake pytest pulp
PYTHONNOUSERSITE=1 "${core_target}/bin/python" -m pip check

# 项目 wheel 是纯 Python 包，三套缓存中的文件完全相同。统一从 core wheelhouse
# 重装可减少离线迁移包约 4 GB，同时不改变 WMH/T1 环境中的第三方依赖。
restore_packed_env "${project_root}/envs/offline/wmh-env.tar.gz" "${environment_root}/wmh" "${project_root}/wheels/core"
restore_packed_env "${project_root}/envs/offline/t1-env.tar.gz" "${environment_root}/t1" "${project_root}/wheels/core"

echo "core/WMH/T1 离线环境已恢复。运行 substain-features verify-offline --smoke-test 继续验证。"
