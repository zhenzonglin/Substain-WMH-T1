#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
echo "[06/06] 核验V1.0输入、四图QC、人工门控和并行配置"
PYTHONPATH="${root}/src" "${root}/envs/core-venv/bin/python" "${root}/scripts/verify_outputs.py"
