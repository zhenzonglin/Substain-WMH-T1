#!/usr/bin/env bash
set -euo pipefail
root="${1:-/data/usersdir/linzhenzong/Substain}"
[[ "${root}" == /data/usersdir/linzhenzong/Substain && -d "${root}" && ! -L "${root}" ]] || {
  echo '拒绝操作非预期活动项目。' >&2; exit 2;
}
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
core_python="${root}/envs/core-venv/bin/python"
# venv的Python允许合法软链接；源码和目标路径仍严格校验。
[[ -f "${core_python}" && -x "${core_python}" ]] || {
  echo "Python不可执行: ${core_python}" >&2; exit 2;
}
(cd "${bundle_dir}" && sha256sum -c SHA256SUMS)
export PYTHONUTF8=1
exec "${core_python}" "${bundle_dir}/deploy_ws1_t1_hybrid.py" "${root}"
