#!/usr/bin/env python3
"""冻结工作站1重跑名单；默认仅预览，--apply才归档并清理。"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


GRID_ERROR_SIGNATURE = "WMH 与 20区图谱网格不一致"
INTERRUPTION_MARKERS = (
    "exit=-15",
    "exit = -15",
    "sigterm",
    "keyboardinterrupt",
    "terminated",
    "signal 15",
    "returncode -15",
)
REPRESENTATIVE_IDS = ("1303105005", "1349853324", "1480140859")


def _load_status(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("状态文件无法解析: {}".format(path)) from exc
    if payload.get("status") not in {"pass", "fail"}:
        raise RuntimeError("未知状态值: {} -> {}".format(path, payload.get("status")))
    if payload.get("stage") not in {None, path.stem}:
        raise RuntimeError("状态阶段与文件名不一致: {}".format(path))
    return payload


def _timestamp(payload: Dict[str, object], path: Path) -> datetime:
    value = payload.get("timestamp_utc")
    if not isinstance(value, str):
        raise RuntimeError("状态缺少timestamp_utc: {}".format(path))
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("状态时间无法解析: {}".format(path)) from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def _status_text(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_managed(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise RuntimeError("拒绝软链接: {}".format(path))
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("拒绝项目外路径: {}".format(path)) from exc
    return resolved


def _assert_tree_safe(path: Path, root: Path) -> None:
    _assert_managed(path, root)
    if path.is_dir():
        for child in path.rglob("*"):
            _assert_managed(child, root)


def _tree_summary(path: Path) -> Dict[str, object]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()] if path.is_dir() else [path]
    return {
        "path": str(path),
        "file_count": len(files),
        "size_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def _status_records(subject_dir: Path) -> List[Tuple[Path, Dict[str, object]]]:
    status_dir = subject_dir / "status"
    if not status_dir.is_dir():
        return []
    records = []
    for path in sorted(status_dir.glob("*.json")):
        records.append((path, _load_status(path)))
    return records


def discover(derivatives: Path) -> Tuple[List[str], Dict[str, List[Path]]]:
    grid_cases: List[str] = []
    interruption_files: Dict[str, List[Path]] = {}
    for subject_dir in sorted(derivatives.glob("sub-*")):
        participant_id = subject_dir.name[4:]
        records = _status_records(subject_dir)
        wmh = next((payload for path, payload in records if path.stem == "wmh"), None)
        if wmh and wmh.get("status") == "fail" and GRID_ERROR_SIGNATURE.lower() in _status_text(wmh):
            grid_cases.append(participant_id)
            continue
        marked = [
            (path, payload)
            for path, payload in records
            if payload.get("status") == "fail"
            and any(marker in _status_text(payload) for marker in INTERRUPTION_MARKERS)
        ]
        if not marked:
            continue
        first_interruption = min(_timestamp(payload, path) for path, payload in marked)
        archive_paths = []
        for path, payload in records:
            if payload.get("status") == "fail" and _timestamp(payload, path) >= first_interruption:
                archive_paths.append(path)
            elif path.stem == "cleanup":
                archive_paths.append(path)
        interruption_files[participant_id] = sorted(set(archive_paths), key=lambda value: value.name)
    return sorted(grid_cases), interruption_files


def _representatives(grid_cases: Sequence[str]) -> List[str]:
    preferred = [participant for participant in REPRESENTATIVE_IDS if participant in grid_cases]
    fallback = [participant for participant in grid_cases if participant not in preferred]
    return (preferred + fallback)[:3]


def _archive_file(source: Path, destination: Path, archive_records: List[Dict[str, object]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_records.append(
        {
            "source": str(source),
            "archive": str(destination),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    )
    shutil.move(str(source), str(destination))


def apply_plan(
    root: Path,
    derivatives: Path,
    grid_cases: Sequence[str],
    interruption_files: Dict[str, List[Path]],
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = root / "archive" / "ws1-v1.0.9-rerun-{}".format(stamp)
    if archive_root.exists():
        raise RuntimeError("归档目录已存在: {}".format(archive_root))
    archive_root.mkdir(parents=True)
    archived_files: List[Dict[str, object]] = []
    removed_entries: List[Dict[str, object]] = []

    for participant_id in grid_cases:
        subject_dir = derivatives / "sub-{}".format(participant_id)
        for name in ("status", "logs"):
            source = subject_dir / name
            if source.is_dir():
                destination = archive_root / "grid" / subject_dir.name / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
                    archived_files.append(
                        {
                            "source": str(file_path),
                            "archive": str(destination / file_path.relative_to(source)),
                            "size_bytes": file_path.stat().st_size,
                            "sha256": _sha256(file_path),
                        }
                    )
                shutil.move(str(source), str(destination))
        for candidate in sorted(subject_dir.iterdir(), key=lambda value: value.name):
            _assert_tree_safe(candidate, root)
            removed_entries.append(_tree_summary(candidate))
            if candidate.is_dir():
                shutil.rmtree(str(candidate))
            elif candidate.is_file():
                candidate.unlink()
            else:
                raise RuntimeError("拒绝未知文件类型: {}".format(candidate))
        qc_dir = derivatives / "qc"
        if qc_dir.is_dir():
            for qc_path in sorted(qc_dir.glob("{}_*.png".format(participant_id))):
                _assert_managed(qc_path, root)
                _archive_file(qc_path, archive_root / "grid" / "qc" / qc_path.name, archived_files)

    for participant_id, paths in sorted(interruption_files.items()):
        for source in paths:
            if not source.is_file():
                continue
            destination = archive_root / "interrupted" / "sub-{}".format(participant_id) / "status" / source.name
            _archive_file(source, destination, archived_files)

    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    representatives = _representatives(grid_cases)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "grid_error_signature": GRID_ERROR_SIGNATURE,
        "grid_failure_participants": list(grid_cases),
        "representative_participants": representatives,
        "interrupted_participants": sorted(interruption_files),
        "archived_files": archived_files,
        "removed_intermediate_entries": removed_entries,
        "archive_root": str(archive_root),
    }
    manifest_path = archive_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (logs / "ws1_v1.0.9_grid_failures.txt").write_text(
        "".join("{}\n".format(value) for value in grid_cases), encoding="utf-8"
    )
    (logs / "ws1_v1.0.9_interrupted.txt").write_text(
        "".join("{}\n".format(value) for value in sorted(interruption_files)), encoding="utf-8"
    )
    (logs / "ws1_v1.0.9_rerun_manifest.path").write_text(str(manifest_path) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="执行归档和清理；缺省只预览")
    args = parser.parse_args()
    root = args.project_root.resolve()
    derivatives = root / "derivatives" / "substain_features"
    if args.project_root.is_symlink() or not derivatives.is_dir():
        raise SystemExit("无效项目或衍生结果目录: {}".format(root))
    _assert_managed(derivatives, root)
    grid_cases, interruption_files = discover(derivatives)

    # 在任何写操作前验证全部目标树，确保不会穿过软链接。
    for participant_id in grid_cases:
        _assert_tree_safe(derivatives / "sub-{}".format(participant_id), root)
    for paths in interruption_files.values():
        for path in paths:
            _assert_managed(path, root)

    print("模式: {}".format("APPLY" if args.apply else "DRY-RUN"))
    print("WMH网格失败病例: {}".format(len(grid_cases)))
    print("代表病例: {}".format(",".join(_representatives(grid_cases)) or "none"))
    print("TERM中断病例: {}".format(len(interruption_files)))
    if not args.apply:
        print("未修改任何文件；确认停止进程后加 --apply 执行。")
        return 0
    manifest_path = apply_plan(root, derivatives, grid_cases, interruption_files)
    print("重排队清单: {}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
