#!/usr/bin/env bash
set -euo pipefail

# 该脚本只在联网准备阶段运行；正式断网验收从项目内 tar 恢复镜像。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
image="substain/offline-smoke:ubuntu20.04-cuda12.4-py38"
archive="${project_root}/resources/tools/offline-smoke-image.tar"
docker build -t "${image}" "${project_root}/containers/offline-smoke"
docker save -o "${archive}" "${image}"
(cd "$(dirname "${archive}")" && sha256sum "$(basename "${archive}")" > "$(basename "${archive}").sha256")
echo "离线烟雾测试镜像已保存：${archive}"
