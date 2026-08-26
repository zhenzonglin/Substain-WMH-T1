#!/usr/bin/env bash
set -euo pipefail

# 生成可迁移的项目运行包。原始影像、历史归档、派生结果和不可迁移的活跃环境均不进入包内。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
project_parent="$(dirname "${project_root}")"
project_name="$(basename "${project_root}")"
archive="${project_parent}/${project_name}_offline.tar.gz"
contents="${project_parent}/${project_name}_offline.contents.txt"
checksum="${archive}.sha256"
verification="${project_parent}/${project_name}_offline.verification.json"

if [[ -e "${archive}" || -e "${contents}" || -e "${checksum}" || -e "${verification}" ]]; then
  echo "目标文件已存在，拒绝覆盖：${project_parent}/${project_name}_offline.*" >&2
  exit 1
fi

# 打包前先确认当前资源与环境归档没有损坏。
"${project_root}/run_pipeline.sh" offline
(cd "${project_root}/envs/offline" && sha256sum -c environment_archives.sha256)
test -f "${project_root}/wheels/core/substain_features-1.0.0-py3-none-any.whl"
test -x "${project_root}/resources/tools/ants-2.5.4/bin/antsRegistration"

temporary_archive="$(mktemp --tmpdir="${project_parent}" ".${project_name}_offline.tar.gz.tmp.XXXXXX")"
temporary_contents="$(mktemp --tmpdir="${project_parent}" ".${project_name}_offline.contents.tmp.XXXXXX")"
cleanup() {
  rm -f -- "${temporary_archive}" "${temporary_contents}"
}
trap cleanup EXIT

compressor="gzip -1"
if command -v pigz >/dev/null 2>&1; then
  compressor="pigz -1"
fi

# envs/offline 中的 conda-pack 归档是可迁移环境；envs/wmh、envs/t1 和 offline/envs
# 已绑定本机安装前缀，因此明确排除。WMH/T1 仅需 core 中同一份纯 Python 项目 wheel。
exclude_args=(
  "--exclude=${project_name}/BIDS"
  "--exclude=${project_name}/Lesion"
  "--exclude=${project_name}/archive"
  "--exclude=${project_name}/derivatives"
  "--exclude=${project_name}/transfer"
  "--exclude=${project_name}/inputs"
  "--exclude=${project_name}/config/metadata.tsv"
  "--exclude=${project_name}/config/participants.tsv"
  "--exclude=${project_name}/.git"
  "--exclude=${project_name}/build"
  "--exclude=${project_name}/dist"
  "--exclude=${project_name}/logs"
  "--exclude=${project_name}/offline/envs"
  "--exclude=${project_name}/offline/matplotlib-cache"
  "--exclude=${project_name}/envs/wmh"
  "--exclude=${project_name}/envs/t1"
  "--exclude=${project_name}/envs/core-site"
  "--exclude=${project_name}/envs/core-venv"
  "--exclude=${project_name}/envs/core-venv.failed-*"
  "--exclude=${project_name}/envs/repair-backup"
  "--exclude=${project_name}/resources/micromamba"
  "--exclude=${project_name}/resources/packages"
  "--exclude=${project_name}/wheels/wmh"
  "--exclude=${project_name}/wheels/t1"
  "--exclude=${project_name}/wheels/pip-cache"
  "--exclude=${project_name}/wheels/final-build"
  "--exclude=${project_name}/src/substain_features.egg-info"
  "--exclude=${project_name}/offline_bundle"
  "--exclude=${project_name}/.snakemake"
  "--exclude=${project_name}/.pytest_cache"
  "--exclude=${project_name}/pipeline.log"
  "--exclude=__pycache__"
  "--exclude=*.pyc"
)

tar "${exclude_args[@]}" \
  --index-file="${temporary_contents}" --verbose \
  -I "${compressor}" -cf "${temporary_archive}" \
  -C "${project_parent}" "${project_name}"

mv -- "${temporary_archive}" "${archive}"
mv -- "${temporary_contents}" "${contents}"
(cd "${project_parent}" && sha256sum "$(basename "${archive}")" > "$(basename "${checksum}")")
"${project_root}/envs/core-venv/bin/python" "${project_root}/scripts/verify_offline_package.py" \
  --archive "${archive}" --contents "${contents}" --output "${verification}"

trap - EXIT
echo "离线项目包：${archive}"
echo "SHA256文件：${checksum}"
echo "内容清单：${contents}"
echo "结构验证：${verification}"
