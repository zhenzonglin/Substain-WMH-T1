#!/usr/bin/env bash
set -euo pipefail

# 构建可直接叠加解压的多卷迁移包；每卷必须小于3,000,000,000字节。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
project_parent="$(dirname "${project_root}")"
transfer_root="${project_root}/transfer"

if [[ -e "${transfer_root}" ]]; then
  echo "目标目录已存在，拒绝覆盖：${transfer_root}" >&2
  exit 1
fi
mkdir -p "${transfer_root}"
staging_root="$(mktemp -d --tmpdir="${transfer_root}" .staging.XXXXXX)"
list_root="$(mktemp -d --tmpdir="${transfer_root}" .lists.XXXXXX)"

cleanup() {
  for target in "${staging_root}" "${list_root}"; do
    if [[ -n "${target}" && ( "${target}" == "${transfer_root}"/.staging.* || "${target}" == "${transfer_root}"/.lists.* ) ]]; then
      if [[ -d "${target}" && ! -L "${target}" ]]; then
        rm -rf -- "${target}"
      fi
    fi
  done
}
trap cleanup EXIT

mkdir -p "${staging_root}/Substain"
rsync -a --delete \
  --exclude=/BIDS/ \
  --exclude=/Lesion/ \
  --exclude=/archive/ \
  --exclude=/derivatives/ \
  --exclude=/transfer/ \
  --exclude=/offline/envs/ \
  --exclude=/offline/matplotlib-cache/ \
  --exclude=/envs/wmh/ \
  --exclude=/envs/t1/ \
  --exclude=/envs/offline/ \
  --exclude=/envs/repair-backup/ \
  --exclude=/resources/micromamba/ \
  --exclude=/resources/packages/ \
  --exclude=/wheels/wmh/ \
  --exclude=/wheels/t1/ \
  --exclude=/wheels/pip-cache/ \
  --exclude=/wheels/final-build/ \
  --exclude=/src/substain_features.egg-info/ \
  --exclude=/offline_bundle/ \
  --exclude=/.snakemake/ \
  --exclude=/.pytest_cache/ \
  --exclude=/pipeline.log \
  --exclude=__pycache__/ \
  --exclude='*.pyc' \
  "${project_root}/" "${staging_root}/Substain/"

mkdir -p "${staging_root}/Substain/envs/wmh" "${staging_root}/Substain/envs/t1"
tar -xzf "${project_root}/envs/offline/wmh-env.tar.gz" -C "${staging_root}/Substain/envs/wmh"
tar -xzf "${project_root}/envs/offline/t1-env.tar.gz" -C "${staging_root}/Substain/envs/t1"
touch "${staging_root}/Substain/envs/wmh/.substain_transfer_needs_conda_unpack"
touch "${staging_root}/Substain/envs/t1/.substain_transfer_needs_conda_unpack"

"${project_root}/envs/core-venv/bin/python" "${project_root}/scripts/build_transfer_parts.py" \
  --staging-root "${staging_root}" --list-dir "${list_root}"

compressor="gzip -1"
if command -v pigz >/dev/null 2>&1; then
  compressor="pigz -1"
fi

"${project_root}/envs/core-venv/bin/python" - "${list_root}/groups.json" <<'PY' > "${list_root}/archive_rows.tsv"
import json, sys
for group in json.load(open(sys.argv[1], encoding="utf-8"))["groups"]:
    print(f'{group["archive"]}\t{group["list"]}')
PY

while IFS=$'\t' read -r archive_name list_name; do
  tar --no-recursion -I "${compressor}" -cf "${transfer_root}/${archive_name}" \
    -C "${staging_root}" -T "${list_root}/${list_name}"
  tar -tzf "${transfer_root}/${archive_name}" > "${transfer_root}/${archive_name%.tar.gz}.contents.txt"
done < "${list_root}/archive_rows.tsv"

(cd "${transfer_root}" && sha256sum -- *.tar.gz > SHA256SUMS)
"${project_root}/envs/core-venv/bin/python" - "${transfer_root}" "${list_root}/groups.json" <<'PY'
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
groups = json.load(open(sys.argv[2], encoding="utf-8"))["groups"]
limit = 3_000_000_000
seen = set()
rows = []
unsafe = []
for group in groups:
    archive = root / group["archive"]
    members = (root / group["archive"].replace(".tar.gz", ".contents.txt")).read_text(encoding="utf-8").splitlines()
    duplicate = sorted(set(members) & seen)
    if duplicate:
        raise SystemExit(f"跨卷重复成员: {duplicate[:5]}")
    seen.update(members)
    for member in members:
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts:
            unsafe.append(member)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    rows.append({
        "archive": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": digest.hexdigest(),
        "member_count": len(members),
        "logical_size_bytes": group["logical_size_bytes"],
    })
for row in rows:
    if row["size_bytes"] >= limit:
        raise SystemExit(f"分卷超过3 GB: {row}")
required = {
    "Substain/run_pipeline.sh",
    "Substain/scripts/finalize_transfer.sh",
    "Substain/src/substain_features/symmetry.py",
    "Substain/src/substain_features/synthstrip.py",
    "Substain/resources/models/synthstrip.1.pt",
    "Substain/resources/tools/ants-2.5.4/bin/antsRegistration",
    "Substain/envs/wmh/bin/python",
    "Substain/envs/t1/bin/python",
}
missing = sorted(required - seen)
forbidden = sorted(member for member in seen if any(
    member == prefix or member.startswith(prefix + "/")
    for prefix in ("Substain/BIDS", "Substain/Lesion", "Substain/archive", "Substain/derivatives", "Substain/transfer")
))
status = "pass" if not missing and not forbidden and not unsafe else "fail"
with (root / "transfer_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
    writer.writeheader(); writer.writerows(rows)
(root / "transfer_verification.json").write_text(json.dumps({
    "status": status,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "archive_count": len(rows),
    "total_archive_bytes": sum(row["size_bytes"] for row in rows),
    "max_archive_bytes": max(row["size_bytes"] for row in rows),
    "all_archives_below_3GB": all(row["size_bytes"] < limit for row in rows),
    "union_member_count": len(seen),
    "missing_required_members": missing,
    "forbidden_members": forbidden,
    "unsafe_members": unsafe,
}, ensure_ascii=False, indent=2), encoding="utf-8")
if status != "pass":
    raise SystemExit(1)
PY

cp -- "${project_root}/docs/README_TRANSFER_ZH.txt" "${transfer_root}/README_TRANSFER_ZH.txt"

trap - EXIT
cleanup
echo "transfer_build pass: ${transfer_root}"
