#!/usr/bin/env bash
set -euo pipefail

# 联网准备机运行一次。正式工作站只接收校验后的项目离线包。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mkdir -p "${project_root}/resources/models" "${project_root}/resources/normative" \
  "${project_root}/resources/templates" "${project_root}/resources/packages" "${project_root}/resources/tools"

wget -c --tries=5 --timeout=30 -O "${project_root}/resources/models/WMH-SynthSeg_v10_231110.pth" \
  "https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth"
wget -c --tries=5 --timeout=30 -O "${project_root}/resources/models/synthstrip.1.pt" \
  "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt"
echo "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33  ${project_root}/resources/models/synthstrip.1.pt" | sha256sum -c -
genmind_commit="d1546dc01cf44842a2b30fb71c5118154cecf2e6"
wget -c --tries=5 --timeout=30 -O "${project_root}/resources/normative/genmind_dataset.csv" \
  "https://huggingface.co/spaces/rongguangw/GenMIND/resolve/${genmind_commit}/dataset/genmind_dataset.csv?download=true"
# 保存官方生成器使用的6个种族×性别KDE和列名字典；当前特征转换只读CSV，
# 这些模型用于离线复现/审计，不会替代本版预定义技术常模。
genmind_root="${project_root}/resources/normative/genmind_upstream"
mkdir -p "${genmind_root}/model" "${genmind_root}/script"
for model_file in col_dict.npz kde_asian_female.npz kde_asian_male.npz kde_black_female.npz kde_black_male.npz kde_white_female.npz kde_white_male.npz; do
  wget -c --tries=5 --timeout=30 -O "${genmind_root}/model/${model_file}" \
    "https://huggingface.co/spaces/rongguangw/GenMIND/resolve/${genmind_commit}/model/${model_file}?download=true"
done
wget -c --tries=5 --timeout=30 -O "${genmind_root}/app.py" \
  "https://huggingface.co/spaces/rongguangw/GenMIND/resolve/${genmind_commit}/app.py?download=true"
wget -c --tries=5 --timeout=30 -O "${genmind_root}/script/synthetic_data_generation.ipynb" \
  "https://huggingface.co/spaces/rongguangw/GenMIND/resolve/${genmind_commit}/script/synthetic_data_generation.ipynb?download=true"
wget -c --tries=5 --timeout=30 -O "${genmind_root}/requirements.txt" \
  "https://huggingface.co/spaces/rongguangw/GenMIND/resolve/${genmind_commit}/requirements.txt?download=true"

cp -f "${project_root}/resources/third_party/WMH_progression_modeling/Residual_Info.mat" \
  "${project_root}/resources/normative/Residual_Info.mat"
cp -f "${project_root}/resources/third_party/WMH_progression_modeling/MNI_ch2better_WM_20ROIs.nii.gz" \
  "${project_root}/resources/templates/MNI_ch2better_WM_20ROIs.nii.gz"

ants_zip="${project_root}/resources/packages/ants-2.5.4-ubuntu-20.04-X64-gcc.zip"
wget -c --tries=5 --timeout=30 -O "${ants_zip}" \
  "https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu-20.04-X64-gcc.zip"
if [[ ! -d "${project_root}/resources/tools/ants-2.5.4" ]]; then
  unzip -q "${ants_zip}" -d "${project_root}/resources/tools"
fi

# 从发行版数据包提取与 Chung 图谱同源使用的 ch2better 模板，不安装系统软件。
if [[ ! -f "${project_root}/resources/templates/ch2better.nii.gz" ]]; then
  package_dir="${project_root}/resources/packages"
  deb_file="${package_dir}/mricron-data_1.2.20211006+dfsg-4_all.deb"
  wget -c --tries=5 --timeout=30 -O "${deb_file}" \
    "https://deb.debian.org/debian/pool/main/m/mricron/mricron-data_1.2.20211006+dfsg-4_all.deb"
  echo "d40aa0ed66cf89bf6cafa049ad3884703c9d975c2ee2737525b7196528df9d6f  ${deb_file}" | sha256sum -c -
  extract_dir="$(mktemp -d "${project_root}/resources/packages/mricron-extract.XXXXXX")"
  case "${extract_dir}" in "${project_root}/resources/packages/"*) ;; *) exit 70 ;; esac
  dpkg-deb -x "${deb_file}" "${extract_dir}"
  cp -f "${extract_dir}/usr/share/mricron/templates/ch2better.nii.gz" "${project_root}/resources/templates/ch2better.nii.gz"
  rm -rf -- "${extract_dir}"
fi

"${project_root}/scripts/prepare_runtime.sh"
echo "基础资源下载完成。DLMUSE/DLICV 权重由 scripts/install_full_envs.sh 下载到受许可的项目运行时副本。"
