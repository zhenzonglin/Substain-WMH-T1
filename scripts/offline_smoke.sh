#!/usr/bin/env bash
set -euo pipefail

# 在独立目录恢复三环境，并在 Docker --network none 中验证代码、模型和可用输入。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
verification_root="${SUBSTAIN_OFFLINE_VERIFY_ROOT:-${project_root}/offline}"
environment_root="${verification_root}/envs"
log_path="${verification_root}/smoke.log"
mkdir -p "${verification_root}"

container_image="substain/offline-smoke:ubuntu20.04-cuda12.4-py38"
container_archive="${project_root}/resources/tools/offline-smoke-image.tar"
if ! docker image inspect "${container_image}" >/dev/null 2>&1; then
  test -f "${container_archive}"
  (cd "$(dirname "${container_archive}")" && sha256sum -c "$(basename "${container_archive}").sha256")
  docker load -i "${container_archive}" >/dev/null
fi
user_spec="$(id -u):$(id -g)"

docker run --rm --network none --gpus all --user "${user_spec}" \
  -v "${project_root}:${project_root}" -w "${project_root}" "${container_image}" \
  bash -lc "
    set -euo pipefail
    export SUBSTAIN_OFFLINE_ENV_ROOT='${environment_root}'
    export SUBSTAIN_MPLCONFIGDIR='${verification_root}/matplotlib-cache'
    '${project_root}/scripts/verify_transferred_project.sh'
  " >"${log_path}" 2>&1

echo "离线重建与无网络烟雾测试通过：${log_path}"
