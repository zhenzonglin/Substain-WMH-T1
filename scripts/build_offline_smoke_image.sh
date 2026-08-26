#!/usr/bin/env bash
set -euo pipefail

# 该脚本只在联网准备阶段运行；正式断网验收从项目内 tar 恢复镜像。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
image="substain/offline-smoke:ubuntu20.04-cuda12.4-py38"
archive="${project_root}/resources/tools/offline-smoke-image.tar"
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
"${docker_command[@]}" build -t "${image}" "${project_root}/containers/offline-smoke"
"${docker_command[@]}" save -o "${archive}" "${image}"
(cd "$(dirname "${archive}")" && sha256sum "$(basename "${archive}")" > "$(basename "${archive}").sha256")
echo "离线烟雾测试镜像已保存：${archive}"
