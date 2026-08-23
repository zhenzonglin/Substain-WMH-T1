#!/usr/bin/env bash
set -euo pipefail

# 联网准备机执行。所有 micromamba 缓存、环境和模型都保存在项目目录。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
tools_dir="${project_root}/resources/tools"
packages_dir="${project_root}/resources/packages"
micromamba_root="${project_root}/resources/micromamba"
micromamba="${tools_dir}/bin/micromamba"
wheel_root="${project_root}/wheels"
pip_cache="${wheel_root}/pip-cache"
mkdir -p "${tools_dir}" "${packages_dir}" "${micromamba_root}" "${project_root}/envs/offline"
mkdir -p "${wheel_root}/wmh" "${wheel_root}/t1" "${pip_cache}" "${project_root}/envs/locks"

if [[ ! -x "${micromamba}" ]]; then
  archive="${packages_dir}/micromamba-linux-64.tar.bz2"
  wget -c --tries=5 --timeout=30 -O "${archive}" "https://micro.mamba.pm/api/micromamba/linux-64/latest"
  tar -xjf "${archive}" -C "${tools_dir}" bin/micromamba
fi

export MAMBA_ROOT_PREFIX="${micromamba_root}"
export PIP_CACHE_DIR="${pip_cache}"
export HF_HUB_DISABLE_XET=1
cd "${project_root}"
if [[ ! -x "${project_root}/envs/wmh/bin/python" ]]; then
  "${micromamba}" create -y -p "${project_root}/envs/wmh" -f "${project_root}/envs/wmh.yaml"
fi
if [[ ! -x "${project_root}/envs/t1/bin/python" ]]; then
  "${micromamba}" create -y -p "${project_root}/envs/t1" -f "${project_root}/envs/t1.yaml"
fi

# WMH 环境采用 PyTorch 2.7.1/CUDA 12.8：这是支持 RTX 50 系 Blackwell 的首个官方预编译组合。
"${project_root}/envs/wmh/bin/pip" install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1
"${project_root}/envs/wmh/bin/pip" install \
  numpy==1.26.4 scipy==1.13.1 nibabel==5.3.2 h5py==3.12.1 \
  matplotlib==3.9.4 pandas==2.2.3 pyyaml==6.0.2 click==8.1.8 surfa==0.6.3
"${project_root}/envs/wmh/bin/pip" install --no-deps "${project_root}"

# NiChart 固定 torch<=2.2.1，因此 T1 环境保持官方约束并默认允许自动回退 CPU。
"${project_root}/envs/t1/bin/pip" install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.2.1
"${project_root}/envs/t1/bin/pip" install \
  --constraint "${project_root}/envs/t1-constraints.txt" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  numpy==1.26.4 scipy==1.13.1 nibabel==5.3.2 pandas==2.2.3 \
  matplotlib==3.9.4 pyyaml==6.0.2 click==8.1.8 huggingface_hub==0.27.1 \
  scikit-learn==1.2.2 nnunetv2==2.5.1 DLICV==1.0.5
"${project_root}/envs/t1/bin/pip" install --no-deps "${project_root}/resources/runtime/DLMUSE"
"${project_root}/envs/t1/bin/pip" install --no-deps "${project_root}/resources/runtime/NiChart_DLMUSE"
"${project_root}/envs/t1/bin/pip" install --no-deps "${project_root}"

# 只把模型下载到项目工作副本/项目环境，不污染固定提交的源码 clone。
"${project_root}/envs/t1/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nichart/DLMUSE', local_dir='${project_root}/resources/runtime/DLMUSE/DLMUSE', max_workers=1)"
"${project_root}/envs/t1/bin/python" -c "from pathlib import Path; import DLICV; from huggingface_hub import snapshot_download; snapshot_download(repo_id='nichart/DLICV', local_dir=str(Path(DLICV.__file__).parent), max_workers=1)"
# DLMUSE 在运行时把包目录作为 local_dir；显式同步项目快照，避免离线时只有源码没有 nnUNet 权重。
dlmuse_package_dir="$("${project_root}/envs/t1/bin/python" -c 'from pathlib import Path; import DLMUSE; print(Path(DLMUSE.__file__).parent)')"
dlicv_package_dir="$("${project_root}/envs/t1/bin/python" -c 'from pathlib import Path; import DLICV; print(Path(DLICV.__file__).parent)')"
test -f "${project_root}/resources/runtime/DLMUSE/DLMUSE/nnunet_results/Dataset903_Task903_DLMUSEV2/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"
test -f "${project_root}/resources/runtime/DLMUSE/DLMUSE/nnunet_results/Dataset903_Task903_DLMUSEV2/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
test -f "${project_root}/resources/runtime/DLMUSE/DLMUSE/nnunet_results/Dataset903_Task903_DLMUSEV2/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
mkdir -p "${dlmuse_package_dir}/nnunet_results"
cp -a "${project_root}/resources/runtime/DLMUSE/DLMUSE/nnunet_results/." "${dlmuse_package_dir}/nnunet_results/"
test -f "${dlmuse_package_dir}/nnunet_results/Dataset903_Task903_DLMUSEV2/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
test -f "${dlicv_package_dir}/nnunet_results/Dataset901_Task901_dlicv/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"
test -f "${dlicv_package_dir}/nnunet_results/Dataset901_Task901_dlicv/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
test -f "${dlicv_package_dir}/nnunet_results/Dataset901_Task901_dlicv/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

# 记录真实 GPU 可执行性；官方 T1 依赖固定 torch<=2.2.1，新显卡不兼容时可切换 CPU profile。
"${project_root}/envs/wmh/bin/python" "${project_root}/scripts/probe_gpu.py" \
  > "${project_root}/envs/wmh_gpu_probe.json"
"${project_root}/envs/t1/bin/python" "${project_root}/scripts/probe_gpu.py" \
  > "${project_root}/envs/t1_gpu_probe.json"

"${project_root}/envs/wmh/bin/pip" list --format=freeze | LC_ALL=C sort > "${project_root}/envs/locks/wmh-pip-lock-all.txt"
grep -Eiv '^substain-features==' \
  "${project_root}/envs/locks/wmh-pip-lock-all.txt" > "${project_root}/envs/locks/wmh-pip-lock.txt"
"${project_root}/envs/t1/bin/pip" list --format=freeze | LC_ALL=C sort > "${project_root}/envs/locks/t1-pip-lock-all.txt"
grep -Eiv '^(DLMUSE|NiChart[_-]DLMUSE|substain-features)==' \
  "${project_root}/envs/locks/t1-pip-lock-all.txt" > "${project_root}/envs/locks/t1-pip-lock.txt"
download_locked_wheels() {
  local pip_executable="$1"
  local lock_file="$2"
  local destination="$3"
  local extra_index="$4"
  local requirement
  while IFS= read -r requirement; do
    [[ -z "${requirement}" || "${requirement}" == \#* ]] && continue
    "${pip_executable}" download -d "${destination}" --no-deps \
      --retries 12 --timeout 180 --extra-index-url "${extra_index}" "${requirement}"
  done < "${lock_file}"
}

# 每个 wheel 独立提交；网络中断时不丢弃已经完成的大文件。
download_locked_wheels "${project_root}/envs/wmh/bin/pip" \
  "${project_root}/envs/locks/wmh-pip-lock.txt" "${wheel_root}/wmh" \
  https://download.pytorch.org/whl/cu128
download_locked_wheels "${project_root}/envs/t1/bin/pip" \
  "${project_root}/envs/locks/t1-pip-lock.txt" "${wheel_root}/t1" \
  https://download.pytorch.org/whl/cpu

# 把仅有源码分发的依赖预编译为 wheel，离线工作站无需编译工具链。
for source_archive in "${wheel_root}/t1"/*.tar.gz; do
  [[ -e "${source_archive}" ]] || continue
  "${project_root}/envs/t1/bin/pip" wheel --no-deps --no-build-isolation \
    -w "${wheel_root}/t1" "${source_archive}"
done
"${project_root}/envs/wmh/bin/pip" wheel --no-deps -w "${wheel_root}/wmh" "${project_root}"
"${project_root}/envs/t1/bin/pip" wheel --no-deps -w "${wheel_root}/t1" "${project_root}/resources/runtime/DLMUSE"
"${project_root}/envs/t1/bin/pip" wheel --no-deps -w "${wheel_root}/t1" "${project_root}/resources/runtime/NiChart_DLMUSE"
"${project_root}/envs/t1/bin/pip" wheel --no-deps -w "${wheel_root}/t1" "${project_root}"

"${project_root}/envs/wmh/bin/conda-pack" -f -p "${project_root}/envs/wmh" -o "${project_root}/envs/offline/wmh-env.tar.gz"
"${project_root}/envs/t1/bin/conda-pack" -f -p "${project_root}/envs/t1" -o "${project_root}/envs/offline/t1-env.tar.gz"
echo "WMH/T1 环境、模型与可迁移环境包已完成。"
