#!/usr/bin/env python3
"""核验离线项目包的成员清单，防止原始数据或本机环境误入压缩包。"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence


FORBIDDEN_PREFIXES: Sequence[str] = (
    "Substain/BIDS",
    "Substain/Lesion",
    "Substain/archive",
    "Substain/derivatives",
    "Substain/transfer",
    "Substain/offline/envs",
    "Substain/envs/wmh",
    "Substain/envs/t1",
    "Substain/envs/core-site",
    "Substain/envs/core-venv",
    "Substain/resources/micromamba",
    "Substain/resources/packages",
    "Substain/wheels/wmh",
    "Substain/wheels/t1",
)

REQUIRED_MEMBERS: Sequence[str] = (
    "Substain/README.md",
    "Substain/config/config.yaml",
    "Substain/config/participants.tsv",
    "Substain/run_pipeline.sh",
    "Substain/workflow/Snakefile",
    "Substain/scripts/install_offline.sh",
    "Substain/scripts/verify_transferred_project.sh",
    "Substain/envs/offline/wmh-env.tar.gz",
    "Substain/envs/offline/t1-env.tar.gz",
    "Substain/envs/offline/environment_archives.sha256",
    "Substain/wheels/core/substain_features-0.1.0-py3-none-any.whl",
    "Substain/resources/models/WMH-SynthSeg_v10_231110.pth",
    "Substain/resources/models/synthstrip.1.pt",
    "Substain/resources/models/SynthStrip_VERSION.txt",
    "Substain/resources/templates/ch2better.nii.gz",
    "Substain/resources/templates/MNI_ch2better_WM_20ROIs.nii.gz",
    "Substain/resources/normative/Residual_Info.mat",
    "Substain/resources/normative/genmind_dataset.csv",
    "Substain/resources/tools/ants-2.5.4/bin/antsRegistration",
    "Substain/resources/tools/synthstrip/mri_synthstrip",
    "Substain/resources/third_party/SynthStrip/mri_synthstrip/mri_synthstrip",
    "Substain/resources/licenses/SynthStrip_MODEL_MIT.txt",
    "Substain/resources/mappings/muse_macro20_v1_provisional.tsv",
    "Substain/resources/tools/offline-smoke-image.tar",
    "Substain/resources/licenses/THIRD_PARTY_LICENSES.md",
    "Substain/offline/verification.json",
)


def sha256(path: Path) -> str:
    """以流式读取计算大文件SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_members(lines: Sequence[str]) -> List[str]:
    members: List[str] = []
    for line in lines:
        member = line.strip()
        if member.startswith("./"):
            member = member[2:]
        member = member.rstrip("/")
        if member:
            members.append(member)
    return sorted(set(members))


def verify(archive: Path, contents: Path) -> Dict[str, object]:
    members = _normalise_members(contents.read_text(encoding="utf-8").splitlines())
    member_set = set(members)
    unsafe = []
    for member in members:
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts:
            unsafe.append(member)
    forbidden = [
        member
        for member in members
        if any(member == prefix or member.startswith(prefix + "/") for prefix in FORBIDDEN_PREFIXES)
    ]
    missing = [member for member in REQUIRED_MEMBERS if member not in member_set]
    status = "pass" if not unsafe and not forbidden and not missing else "fail"
    return {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "member_count": len(members),
        "unsafe_members": unsafe,
        "forbidden_members": forbidden,
        "missing_required_members": missing,
        "raw_bids_included": False if not forbidden else None,
        "lesion_included": False if not forbidden else None,
        "history_archive_included": False if not forbidden else None,
        "derivatives_included": False if not forbidden else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--contents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.archive.resolve(), args.contents.resolve())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
