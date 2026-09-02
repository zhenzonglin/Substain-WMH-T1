"""第三方资源版本、SHA256 和离线完整性。"""

import hashlib
import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd


PINNED_SOURCES = {
    "WMH_progression_modeling": "1d9c8456168fa8b3a0190c6e91ca3dfbc6c90068",
    "WMH-SynthSeg": "2bf9a421f9707142a83db575ac0f3fa9a2d2631c",
    "DLMUSE": "50c59bc1d24b24392305b454a5359eb28eea1aab",
    "NiChart_DLMUSE": "84a977b77243a64f38f2ea0f1423daabe6cfaddd",
    "SynthStrip": "7eb846079b0dc0c92e8313205a3d2387b5c7a354",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_git_head_without_executable(path: Path) -> str:
    """在工作站未安装git时，直接读取仓库元数据中的固定提交。"""

    git_entry = path / ".git"
    if git_entry.is_dir():
        git_dir = git_entry
    elif git_entry.is_file():
        marker = git_entry.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir:"):
            return ""
        candidate = Path(marker.split(":", 1)[1].strip())
        git_dir = candidate if candidate.is_absolute() else (path / candidate).resolve()
    else:
        return ""

    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return ""
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head if len(head) == 40 and all(char in "0123456789abcdef" for char in head.lower()) else ""

    reference = head[5:].strip()
    loose_reference = git_dir / reference
    if loose_reference.is_file():
        return loose_reference.read_text(encoding="utf-8").strip()
    packed_references = git_dir / "packed-refs"
    if packed_references.is_file():
        for line in packed_references.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, name = line.split(" ", 1)
            if name == reference:
                return commit
    return ""


def git_head(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return _read_git_head_without_executable(path)
    if completed.returncode == 0:
        return completed.stdout.strip()
    return _read_git_head_without_executable(path)


def build_manifest(project_root: Path, output: Path) -> pd.DataFrame:
    """记录可离线复核的文件摘要；原始 BIDS/Lesion 不进入资源包。"""

    include_roots = [
        project_root / "resources",
        project_root / "wheels",
        project_root / "envs",
        project_root / "scripts",
        project_root / "src",
        project_root / "workflow",
        project_root / "config",
        project_root / "containers",
    ]
    rows = []
    # 仅转换为绝对路径，禁止resolve符号链接。venv的bin/python通常指向系统解释器；
    # 若解析链接，路径会逃离envs目录，导致活动或失败环境被错误写入正式清单。
    output_absolute = output.absolute()
    final_build_root = (project_root / "wheels" / "final-build").absolute()
    # conda-pack 解包后会合法改写前缀；只校验环境归档/锁文件，不哈希活跃环境目录。
    active_environment_roots = [
        (project_root / "envs" / name).absolute()
        for name in ("wmh", "t1", "core-site", "core-venv", "repair-backup")
    ]
    # 失败的core venv与离线打包脚本同样排除；它们可归档但不能成为正式资源依赖。
    active_environment_roots.extend(
        path.absolute() for path in (project_root / "envs").glob("core-venv.failed-*")
    )
    for root in include_roots:
        if not root.exists():
            continue
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file()
            and not any(part in {".git", "__pycache__", ".pytest_cache"} for part in item.parts)
        ):
            path_absolute = path.absolute()
            if path_absolute == output_absolute:
                continue
            # final-build是联网构建时的临时落点；三套正式wheel目录才进入离线包和清单。
            if final_build_root in path_absolute.parents:
                continue
            if any(
                environment_root == path_absolute or environment_root in path_absolute.parents
                for environment_root in active_environment_roots
            ):
                continue
            rows.append(
                {
                    "relative_path": str(path.relative_to(project_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    table = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False)
    return table


def verify_manifest(project_root: Path, manifest_path: Path) -> Dict[str, object]:
    table = pd.read_csv(manifest_path, sep="\t")
    missing = []
    mismatched = []
    for row in table.itertuples(index=False):
        path = project_root / str(row.relative_path)
        if not path.is_file():
            missing.append(str(row.relative_path))
        elif sha256(path) != str(row.sha256):
            mismatched.append(str(row.relative_path))
    source_mismatch = {}
    for name, commit in PINNED_SOURCES.items():
        actual = git_head(project_root / "resources" / "third_party" / name)
        if actual != commit:
            source_mismatch[name] = {"expected": commit, "actual": actual}
    return {
        "status": "pass" if not missing and not mismatched and not source_mismatch else "fail",
        "network_used": False,
        "checked_files": len(table),
        "missing": missing,
        "sha256_mismatch": mismatched,
        "source_commit_mismatch": source_mismatch,
    }
