#!/usr/bin/env bash
set -euo pipefail

# 在独立目录恢复三环境，并通过受控Docker入口以--network none验证代码、模型和可用输入。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
verification_root="${SUBSTAIN_OFFLINE_VERIFY_ROOT:-${project_root}/offline}"
environment_root="${verification_root}/envs"
log_path="${verification_root}/smoke.log"
mkdir -p "${verification_root}"

# CLI在“--”后按原始参数传入容器入口；直接运行时也可使用环境变量。
docker_command=()
if [[ "${1:-}" == "--" ]]; then
  shift
  docker_command=("$@")
elif [[ $# -gt 0 ]]; then
  echo "容器入口参数必须放在--之后" >&2
  exit 2
elif [[ -n "${SUBSTAIN_DOCKER_COMMAND:-}" ]]; then
  read -r -a docker_command <<< "${SUBSTAIN_DOCKER_COMMAND}"
elif command -v safe_docker.sh >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
  docker_command=("$(command -v sudo)" "$(command -v safe_docker.sh)")
elif command -v safe_docker >/dev/null 2>&1; then
  docker_command=("$(command -v safe_docker)")
else
  docker_command=("docker")
fi
if [[ ${#docker_command[@]} -eq 0 ]]; then
  echo "Docker调用入口为空" >&2
  exit 127
fi
if ! docker_executable="$(command -v "${docker_command[0]}")"; then
  echo "找不到Docker调用入口：${docker_command[0]}" >&2
  exit 127
fi
docker_command[0]="${docker_executable}"

container_image="substain/offline-smoke:ubuntu20.04-cuda12.4-py38"
container_archive="${project_root}/resources/tools/offline-smoke-image.tar"
if ! "${docker_command[@]}" image inspect "${container_image}" >/dev/null 2>&1; then
  test -f "${container_archive}"
  (cd "$(dirname "${container_archive}")" && sha256sum -c "$(basename "${container_archive}").sha256")
  "${docker_command[@]}" load -i "${container_archive}" >/dev/null
fi
user_spec="$(id -u):$(id -g)"

"${docker_command[@]}" run --rm --network none --gpus all --user "${user_spec}" \
  -v "${project_root}:${project_root}" -w "${project_root}" "${container_image}" \
  bash -lc "
    set -euo pipefail
    export SUBSTAIN_OFFLINE_ENV_ROOT='${environment_root}'
    export SUBSTAIN_MPLCONFIGDIR='${verification_root}/matplotlib-cache'
    '${project_root}/scripts/verify_transferred_project.sh'
  " >"${log_path}" 2>&1

echo "离线重建与无网络烟雾测试通过：${log_path}"
