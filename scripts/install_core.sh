#!/usr/bin/env bash
set -euo pipefail

# 仅在项目目录创建私有 venv；不会修改 BIDS/Lesion 或系统 Python。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
test -f "${project_root}/pyproject.toml"
core_site="${project_root}/envs/core-site"
core_env="${project_root}/envs/core-venv"
if [[ ! -x "${core_env}/bin/python" ]]; then
  if [[ -d "${core_env}" && -n "$(find "${core_env}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "core 环境目录非空且缺少解释器，拒绝覆盖：${core_env}" >&2
    exit 1
  fi
  python3 -m venv "${core_env}"
fi

# 兼容本开发机早期的只读系统 Python 包装器；新安装始终使用真正 venv。
if [[ -f "${core_env}/pyvenv.cfg" ]]; then
  "${core_env}/bin/python" -m pip install --upgrade \
    "${project_root}" "pytest>=8,<9" "snakemake==7.32.4" "pulp==2.7.0"
else
  mkdir -p "${core_site}"
  python3 -m pip install --upgrade --target "${core_site}" \
    "${project_root}" "pytest>=8,<9" "snakemake==7.32.4" "pulp==2.7.0"
fi
mkdir -p "${project_root}/wheels/core"
python3 -m pip download --dest "${project_root}/wheels/core" \
  "${project_root}" "pytest>=8,<9" "snakemake==7.32.4" "pulp==2.7.0"
# Python 3.8 venv自带旧pip；将可识别新manylinux标签的安装器wheel一并缓存。
python3 -m pip download --dest "${project_root}/wheels/core" pip setuptools wheel
# 将PyPI仅提供源码的依赖预编译；离线端用--only-binary，绝不临时编译或联网取构建依赖。
for source_archive in "${project_root}/wheels/core"/*.tar.gz; do
  [[ -e "${source_archive}" ]] || continue
  python3 -m pip wheel --no-deps -w "${project_root}/wheels/core" "${source_archive}"
done
python3 -m pip wheel --no-deps -w "${project_root}/wheels/core" "${project_root}"
"${core_env}/bin/substain-features" build-mapping --config-file "${project_root}/config/config.yaml"
echo "core 私有环境已安装：${core_env}"
