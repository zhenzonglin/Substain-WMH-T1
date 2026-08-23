#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_script="${project_root}/resources/third_party/WMH-SynthSeg/WMHSynthSeg/inference.py"
runtime_dir="${project_root}/resources/runtime"
test -f "${source_script}"
mkdir -p "${runtime_dir}"
cp -f "${source_script}" "${runtime_dir}/wmh_synthseg_inference.py"
sed -i "s#model_file = os.path.join('/app/models', 'WMH-SynthSeg_v10_231110.pth')#model_file = os.environ['SUBSTAIN_WMH_MODEL']#" \
  "${runtime_dir}/wmh_synthseg_inference.py"
sed -i "s#torch.load(model_file, map_location=device)#torch.load(model_file, map_location=device, weights_only=False)#" \
  "${runtime_dir}/wmh_synthseg_inference.py"
grep -Fq "SUBSTAIN_WMH_MODEL" "${runtime_dir}/wmh_synthseg_inference.py"
grep -Fq "weights_only=False" "${runtime_dir}/wmh_synthseg_inference.py"

# DLMUSE 的模型下载位置写在 Python 包内部，因此制作工作副本，保持 pinned clone 干净。
if [[ ! -d "${runtime_dir}/DLMUSE" ]]; then
  cp -a "${project_root}/resources/third_party/DLMUSE" "${runtime_dir}/DLMUSE"
fi
echo "运行时副本已准备：${runtime_dir}"
