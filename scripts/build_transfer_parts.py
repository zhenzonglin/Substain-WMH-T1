#!/usr/bin/env python3
"""将迁移暂存树按功能和大小生成互不重叠的文件清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


WMH_BATCH_LIMIT = 2_400_000_000


def files_under(root: Path) -> List[Path]:
    """返回普通文件和符号链接；目录由tar解包时自动创建。"""

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def logical_size(path: Path) -> int:
    return int(path.lstat().st_size)


def batch_by_size(paths: Iterable[Path], limit: int) -> List[List[Path]]:
    """按未压缩大小分批，保证每个压缩卷明显低于3 GB。"""

    batches: List[List[Path]] = []
    current: List[Path] = []
    current_size = 0
    for path in paths:
        size = logical_size(path)
        if size > limit:
            raise ValueError(f"单文件超过分卷上限: {path} ({size})")
        if current and current_size + size > limit:
            batches.append(current)
            current = []
            current_size = 0
        current.append(path)
        current_size += size
    if current:
        batches.append(current)
    return batches


def write_list(path: Path, staging_root: Path, files: Iterable[Path]) -> Dict[str, object]:
    rows = [item.relative_to(staging_root).as_posix() for item in files]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "list": path.name,
        "file_count": len(rows),
        "logical_size_bytes": sum(logical_size(staging_root / row) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--list-dir", required=True, type=Path)
    args = parser.parse_args()
    staging_root = args.staging_root.resolve()
    project = staging_root / "Substain"
    list_dir = args.list_dir.resolve()
    list_dir.mkdir(parents=True, exist_ok=True)

    all_files = files_under(project)
    wmh_root = project / "envs" / "wmh"
    t1_root = project / "envs" / "t1"
    ants_root = project / "resources" / "tools" / "ants-2.5.4"
    resources_root = project / "resources"

    wmh_files = files_under(wmh_root)
    t1_files = files_under(t1_root)
    ants_files = files_under(ants_root)
    resource_files = [
        path
        for path in files_under(resources_root)
        if ants_root not in path.parents
    ]
    excluded = set(wmh_files + t1_files + ants_files + resource_files)
    base_files = [path for path in all_files if path not in excluded]

    groups: List[Dict[str, object]] = []
    groups.append({"archive": "01_project_core.tar.gz", **write_list(list_dir / "01.list", staging_root, base_files)})
    groups.append({"archive": "02_resources.tar.gz", **write_list(list_dir / "02.list", staging_root, resource_files)})
    groups.append({"archive": "03_ants.tar.gz", **write_list(list_dir / "03.list", staging_root, ants_files)})
    for index, batch in enumerate(batch_by_size(wmh_files, WMH_BATCH_LIMIT), start=1):
        name = f"wmh_{index:02d}.list"
        groups.append({
            "archive": f"{index + 3:02d}_wmh_env_{index:02d}.tar.gz",
            **write_list(list_dir / name, staging_root, batch),
        })
    t1_logical_size = sum(logical_size(path) for path in t1_files)
    if t1_logical_size >= 2_900_000_000:
        raise ValueError(f"T1环境超过单卷安全上限: {t1_logical_size}")
    t1_index = len(groups) + 1
    groups.append({
        "archive": f"{t1_index:02d}_t1_env.tar.gz",
        **write_list(list_dir / "t1.list", staging_root, t1_files),
    })

    assigned = sum(int(group["file_count"]) for group in groups)
    if assigned != len(all_files):
        raise RuntimeError(f"文件分组不守恒: assigned={assigned}, all={len(all_files)}")
    (list_dir / "groups.json").write_text(
        json.dumps({"groups": groups, "file_count": len(all_files)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
