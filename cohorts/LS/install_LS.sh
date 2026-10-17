#!/usr/bin/env bash
set -euo pipefail
ls_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ls_project_root="${SUBSTAIN_PROJECT_ROOT:-/data02/linzhenzong/Substain_GB}"
python3 "${ls_source_dir}/verify_checkout_LS.py" "${ls_source_dir}"
if [[ ! -x "${ls_project_root}/envs/core-venv/bin/python" ]]; then
  printf 'Missing existing project Python: %s\n' "${ls_project_root}/envs/core-venv/bin/python" >&2
  exit 2
fi
exec "${ls_project_root}/envs/core-venv/bin/python" \
  "${ls_source_dir}/runner_LS.py" install --root "${ls_project_root}" "$@"
