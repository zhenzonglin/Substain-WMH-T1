#!/usr/bin/env python3
"""核验离线项目包的成员清单，防止原始数据或本机环境误入压缩包。"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence


FORBIDDEN_RELATIVE_PREFIXES: Sequence[str] = (
    "BIDS",
    "Lesion",
    "archive",
    "derivatives",
    "transfer",
    "inputs",
    "config/metadata.tsv",
    "config/participants.tsv",
    ".git",
    "build",
    "dist",
    "logs",
    "offline/envs",
    "envs/wmh",
    "envs/t1",
    "envs/core-site",
    "envs/core-venv",
    "resources/micromamba",
    "resources/packages",
    "wheels/wmh",
    "wheels/t1",
)

REQUIRED_RELATIVE_MEMBERS: Sequence[str] = (
    "README.md",
    "config/config.yaml",
    "config/metadata.example.tsv",
    "run_pipeline.sh",
    "workflow/Snakefile",
    "scripts/install_offline.sh",
    "scripts/verify_transferred_project.sh",
    "envs/offline/wmh-env.tar.gz",
    "envs/offline/t1-env.tar.gz",
    "envs/offline/environment_archives.sha256",
    "resources/models/WMH-SynthSeg_v10_231110.pth",
    "resources/models/synthstrip.1.pt",
    "resources/models/SynthStrip_VERSION.txt",
    "resources/templates/ch2better.nii.gz",
    "resources/templates/MNI_ch2better_WM_20ROIs.nii.gz",
    "resources/normative/Residual_Info.mat",
    "resources/normative/genmind_dataset.csv",
    "resources/tools/ants-2.5.4/bin/antsRegistration",
    "resources/tools/synthstrip/mri_synthstrip",
    "resources/third_party/SynthStrip/mri_synthstrip/mri_synthstrip",
    "resources/licenses/SynthStrip_MODEL_MIT.txt",
    "resources/mappings/muse_macro20_v1_provisional.tsv",
    "resources/tools/offline-smoke-image.tar",
    "resources/licenses/THIRD_PARTY_LICENSES.md",
    "offline/verification.json",
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


def _project_member(project_name: str, relative_path: str) -> str:
    return "{}/{}".format(project_name, relative_path)


def verify(
    archive: Path,
    contents: Path,
    project_name: str,
    project_version: str,
) -> Dict[str, object]:
    project_path = PurePosixPath(project_name)
    if project_path.name != project_name or project_name in {"", ".", ".."}:
        raise ValueError("project_name必须是单层目录名: {}".format(project_name))
    members = _normalise_members(contents.read_text(encoding="utf-8").splitlines())
    member_set = set(members)
    project_wheel = _project_member(
        project_name,
        "wheels/core/substain_features-{}-py3-none-any.whl".format(project_version),
    )
    forbidden_prefixes = tuple(
        _project_member(project_name, relative_path)
        for relative_path in FORBIDDEN_RELATIVE_PREFIXES
    )
    required_members = tuple(
        _project_member(project_name, relative_path)
        for relative_path in REQUIRED_RELATIVE_MEMBERS
    ) + (project_wheel,)
    unsafe = []
    for member in members:
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts:
            unsafe.append(member)
    forbidden = [
        member
        for member in members
        if any(member == prefix or member.startswith(prefix + "/") for prefix in forbidden_prefixes)
    ]
    missing = [member for member in required_members if member not in member_set]
    status = "pass" if not unsafe and not forbidden and not missing else "fail"
    return {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "project_name": project_name,
        "project_version": project_version,
        "project_wheel": project_wheel,
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
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--project-version", required=True)
    args = parser.parse_args()
    report = verify(
        args.archive.resolve(),
        args.contents.resolve(),
        args.project_name.strip(),
        args.project_version.strip(),
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
