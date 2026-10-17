#!/usr/bin/env bash
set -euo pipefail
ls_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ls_project_root="${SUBSTAIN_PROJECT_ROOT:-/data02/linzhenzong/Substain_GB}"
export PYTHONDONTWRITEBYTECODE=1
exec "${ls_project_root}/envs/core-venv/bin/python" \
  "${ls_source_dir}/warmup80_LS.py" start --config "${ls_project_root}/config/config_LS.yaml"
