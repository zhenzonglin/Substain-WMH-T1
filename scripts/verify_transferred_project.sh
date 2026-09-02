#!/usr/bin/env bash
set -euo pipefail

# 在已禁网的Ubuntu 20.04环境中验证解包后的项目。若未携带原始影像，则明确跳过受试者级烟雾测试。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
environment_root="${SUBSTAIN_OFFLINE_ENV_ROOT:-${project_root}/envs}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PIP_NO_INDEX=1
export MPLCONFIGDIR="${SUBSTAIN_MPLCONFIGDIR:-${project_root}/offline/matplotlib-cache}"

if [[ -d "${environment_root}/wmh" && -d "${environment_root}/t1" && ! -f "${project_root}/envs/offline/wmh-env.tar.gz" ]]; then
  "${project_root}/scripts/finalize_transfer.sh"
else
  "${project_root}/scripts/install_offline.sh"
fi
if [[ "${SUBSTAIN_SKIP_NETWORK_ASSERT:-0}" == "1" ]]; then
  echo "network_block_check skipped_by_request"
else
  "${environment_root}/core-venv/bin/python" -c "import socket; s=socket.socket(); s.settimeout(1); rc=s.connect_ex(('1.1.1.1',53)); s.close(); assert rc != 0, rc; print('network_blocked',rc)"
fi
"${environment_root}/core-venv/bin/python" -c "import substain_features; print('core_installed_code', substain_features.__file__)"
"${environment_root}/wmh/bin/python" -c "import substain_features,surfa; print('wmh_installed_code', substain_features.__file__, 'surfa', surfa.__version__)"
"${environment_root}/t1/bin/python" -c "import substain_features; print('t1_installed_code', substain_features.__file__)"
(cd "${project_root}" && "${environment_root}/core-venv/bin/python" -m pytest -q -p no:cacheprovider)
"${environment_root}/core-venv/bin/substain-features" verify-offline --config-file "${project_root}/config/config.yaml"
"${environment_root}/wmh/bin/python" "${project_root}/scripts/probe_gpu.py"
SUBSTAIN_WMH_MODEL="${project_root}/resources/models/WMH-SynthSeg_v10_231110.pth" \
  "${environment_root}/wmh/bin/python" -c "import os,torch; checkpoint=torch.load(os.environ['SUBSTAIN_WMH_MODEL'],map_location='cpu',weights_only=False); print('wmh_checkpoint_loaded',type(checkpoint).__name__)"
"${environment_root}/wmh/bin/python" "${project_root}/scripts/offline_algorithm_smoke.py" \
  --project-root "${project_root}" --output "${project_root}/offline/algorithm-smoke"
"${environment_root}/t1/bin/python" -c "from pathlib import Path; import DLICV,DLMUSE,NiChart_DLMUSE; from substain_features.t1 import dlmuse_model_provenance; dlicv=Path(DLICV.__file__).parent; assert any(path.is_file() and path.stat().st_size>0 for path in dlicv.rglob('*') if path.suffix in {'.pth','.pt','.onnx','.h5'}); print('t1_packages_and_models_loaded',dlmuse_model_provenance())"
"${environment_root}/t1/bin/python" "${project_root}/scripts/verify_genmind_kde.py"

# V1.1不硬编码病例或猜测病灶空间。只有当前配置可生成严格输入契约时才运行一例。
if "${environment_root}/core-venv/bin/python" -m substain_features.cli prepare-inputs \
  --config-file "${project_root}/config/config.yaml"; then
  participant_id="$(awk -F '\t' 'NR==2 {print $1; exit}' "${project_root}/config/participants.tsv")"
  if [[ -z "${participant_id}" ]]; then
    echo "subject_smoke_test failed_empty_participant_id" >&2
    exit 1
  fi
  "${environment_root}/core-venv/bin/python" -m substain_features.cli audit \
    --config-file "${project_root}/config/config.yaml" --participant-id "${participant_id}"
  if [[ "${SUBSTAIN_RUN_SUBJECT_SMOKE:-0}" == "1" ]]; then
    "${environment_root}/core-venv/bin/python" -m substain_features.cli run \
      --config-file "${project_root}/config/config.yaml" --participant-id "${participant_id}" \
      --profile auto --cores 1 --skip-prepare
    echo "subject_smoke_test pass ${participant_id}"
  else
    echo "subject_smoke_test skipped_set_SUBSTAIN_RUN_SUBJECT_SMOKE_1"
  fi
else
  echo "subject_smoke_test skipped_no_valid_v1_inputs"
fi

echo "transferred_project_verification pass"
