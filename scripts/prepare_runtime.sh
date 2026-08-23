#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Python脚本是唯一运行时生成实现，避免shell与Python维护出两套不同补丁。
exec python3 "${project_root}/scripts/prepare_runtime.py"
