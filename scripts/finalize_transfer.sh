#!/usr/bin/env bash
set -euo pipefail

# 多卷迁移包会把两个conda-pack环境直接展开到envs/wmh和envs/t1。
# 本脚本只修正迁移后的绝对前缀，并验证三个运行环境；不访问网络。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${project_root}/src"
export PIP_NO_INDEX=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

finalize_conda_env() {
  local environment="$1"
  local marker="${environment}/.substain_transfer_needs_conda_unpack"
  test -x "${environment}/bin/python"
  if [[ -f "${marker}" ]]; then
    test -x "${environment}/bin/conda-unpack"
    PATH="${environment}/bin:${PATH}" "${environment}/bin/conda-unpack"
    rm -f -- "${marker}"
  fi
  "${environment}/bin/python" -m pip check
}

test -x "${project_root}/envs/core-venv/bin/python"
test -d "${project_root}/envs/core-site"
finalize_conda_env "${project_root}/envs/wmh"
finalize_conda_env "${project_root}/envs/t1"

"${project_root}/envs/core-venv/bin/python" -c "import substain_features; print('core_ready', substain_features.__version__)"
"${project_root}/envs/wmh/bin/python" -c "import substain_features,torch,surfa,xxhash; print('wmh_ready', substain_features.__version__, torch.__version__, surfa.__version__, xxhash.VERSION)"
"${project_root}/envs/t1/bin/python" -c "import substain_features,DLICV,DLMUSE,NiChart_DLMUSE; print('t1_ready', substain_features.__version__)"

echo "transfer_finalize pass: ${project_root}"
