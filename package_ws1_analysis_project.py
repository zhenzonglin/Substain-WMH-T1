#!/usr/bin/env python3
"""Package the live WS1 analysis workflow for a second workstation.

The source project is read-only. Raw images, participant manifests, results,
logs, run state, archives, caches and host-bound active environments are
excluded. Portable offline environment archives, code, models, templates,
third-party tools and tests are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_NAME = "Substain"
DEFAULT_PROJECT_ROOT = Path("/data/usersdir/linzhenzong/Substain")
DEFAULT_OUTPUT_PARENT = DEFAULT_PROJECT_ROOT.parent
BUNDLE_NAME = "Substain_GB"
ONE_GIB = 1024 ** 3

# These contain source data, first-queue identifiers, analysis results, logs or
# mutable run state. They must never enter the migration archive.
EXCLUDED_PREFIXES: Sequence[str] = (
    "BIDS",
    "Lesion",
    "archive",
    "derivatives",
    "inputs",
    "logs",
    "transfer",
    ".git",
    ".snakemake",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "offline_bundle",
    "offline/envs",
    "offline/matplotlib-cache",
    "envs/core-venv",
    "envs/core-site",
    "envs/wmh",
    "envs/t1",
    "envs/repair-backup",
    "resources/micromamba",
    "resources/packages",
    "resources/tools/offline-smoke-image.tar",
    "resources/tools/offline-smoke-image.tar.sha256",
    "wheels/wmh",
    "wheels/t1",
    "wheels/pip-cache",
    "wheels/final-build",
    "src/substain_features.egg-info",
)

EXCLUDED_EXACT: Sequence[str] = (
    "config/participants.tsv",
    "config/metadata.tsv",
    "pipeline.log",
)

EXCLUDED_DIRECTORY_NAMES = {"__pycache__", ".mypy_cache", ".cache"}
EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo", ".pid", ".log", ".logpath")

REQUIRED_FILES: Sequence[str] = (
    "README.md",
    "pyproject.toml",
    "run_pipeline.sh",
    "workflow/Snakefile",
    "config/config.yaml",
    "src/substain_features/pipeline.py",
    "src/substain_features/gpu_pool.py",
    "scripts/start_ws1_v1_0_9.sh",
    "scripts/finish_ws1_v1_0_9.sh",
    "scripts/install_offline.sh",
    "scripts/verify_transferred_project.sh",
    "envs/offline/wmh-env.tar.gz",
    "envs/offline/t1-env.tar.gz",
    "envs/offline/environment_archives.sha256",
    "wheels/core/substain_features-1.0.0-py3-none-any.whl",
    "resources/models/WMH-SynthSeg_v10_231110.pth",
    "resources/models/synthstrip.1.pt",
    "resources/templates/ch2better.nii.gz",
    "resources/templates/MNI_ch2better_WM_20ROIs.nii.gz",
    "resources/normative/Residual_Info.mat",
    "resources/normative/genmind_dataset.csv",
    "resources/tools/ants-2.5.4/bin/antsRegistration",
    "resources/tools/synthstrip/mri_synthstrip",
)

REQUIRED_DIRECTORIES: Sequence[str] = (
    "resources/third_party/WMH-SynthSeg",
    "resources/third_party/DLMUSE",
    "resources/third_party/NiChart_DLMUSE",
    "resources/mappings",
    "tests",
)


class PackagingError(RuntimeError):
    """Raised when packaging cannot prove a safe and complete bundle."""


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _matches_prefix(relative: str, prefix: str) -> bool:
    return relative == prefix or relative.startswith(prefix + "/")


def exclusion_reason(relative: str, is_directory: bool) -> str:
    """Return the matched exclusion rule, or an empty string."""

    if relative in EXCLUDED_EXACT:
        return "exact:" + relative
    for prefix in EXCLUDED_PREFIXES:
        if _matches_prefix(relative, prefix):
            return "prefix:" + prefix
    if relative.startswith("envs/core-venv.failed-"):
        return "runtime-env:envs/core-venv.failed-*"
    name = PurePosixPath(relative).name
    if is_directory and name in EXCLUDED_DIRECTORY_NAMES:
        return "directory-name:" + name
    if not is_directory:
        if name == ".DS_Store":
            return "file-name:.DS_Store"
        if name.endswith(EXCLUDED_FILE_SUFFIXES):
            return "runtime-suffix:" + next(suffix for suffix in EXCLUDED_FILE_SUFFIXES if name.endswith(suffix))
        if name.endswith("_gpu_probe.json") and relative.startswith("envs/"):
            return "runtime-probe:envs/*_gpu_probe.json"
    return ""


def _assert_safe_symlink(path: Path, root: Path) -> str:
    target = os.readlink(str(path))
    if os.path.isabs(target):
        raise PackagingError(f"拒绝绝对符号链接: {path} -> {target}")
    resolved = (path.parent / target).resolve(strict=False)
    try:
        target_relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise PackagingError(f"拒绝越出项目目录的符号链接: {path} -> {target}") from exc
    reason = exclusion_reason(target_relative, resolved.is_dir())
    if reason:
        raise PackagingError(f"符号链接指向被排除内容: {path} -> {target} ({reason})")
    return target


def validate_project(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise PackagingError(f"项目根目录不存在、不是目录或是符号链接: {root}")
    if root.name != PROJECT_NAME:
        raise PackagingError(f"项目根目录名称必须为 {PROJECT_NAME}: {root}")
    missing_files = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    empty_or_missing_directories = [
        relative
        for relative in REQUIRED_DIRECTORIES
        if not (root / relative).is_dir() or not any((root / relative).iterdir())
    ]
    if missing_files or empty_or_missing_directories:
        raise PackagingError(
            f"迁移所需内容不完整；缺少文件={missing_files}；缺少或为空的目录={empty_or_missing_directories}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_environment_archives(root: Path) -> Dict[str, str]:
    checksum_path = root / "envs/offline/environment_archives.sha256"
    expected: Dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise PackagingError(f"环境校验文件格式无效: {line}")
        name = parts[1].lstrip("*").strip()
        candidate = (checksum_path.parent / name).resolve()
        try:
            candidate.relative_to(checksum_path.parent.resolve())
        except ValueError as exc:
            raise PackagingError(f"环境校验路径越界: {name}") from exc
        if not candidate.is_file():
            raise PackagingError(f"环境归档不存在: {candidate}")
        actual = _sha256(candidate)
        if actual != parts[0].lower():
            raise PackagingError(f"环境归档SHA256不匹配: {candidate}")
        expected[name] = actual
    for name in ("wmh-env.tar.gz", "t1-env.tar.gz"):
        if name not in expected:
            raise PackagingError(f"环境校验文件未覆盖 {name}")
    return expected


def collect_entries(root: Path) -> Tuple[List[Dict[str, object]], Mapping[str, int]]:
    """Collect regular files and safe internal symlinks without following links."""

    entries: List[Dict[str, object]] = []
    excluded: Counter[str] = Counter()
    for current_text, directory_names, file_names in os.walk(str(root), topdown=True, followlinks=False):
        current = Path(current_text)
        kept_directories: List[str] = []
        for name in sorted(directory_names):
            path = current / name
            relative = _relative_posix(path, root)
            reason = exclusion_reason(relative, True)
            if reason:
                excluded[reason] += 1
                continue
            if path.is_symlink():
                target = _assert_safe_symlink(path, root)
                entries.append(_entry_record(path, root, "symlink", target))
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            path = current / name
            relative = _relative_posix(path, root)
            reason = exclusion_reason(relative, False)
            if reason:
                excluded[reason] += 1
                continue
            if "\n" in relative or "\r" in relative:
                raise PackagingError(f"文件名含换行，拒绝打包: {path}")
            if path.is_symlink():
                target = _assert_safe_symlink(path, root)
                entries.append(_entry_record(path, root, "symlink", target))
            elif path.is_file():
                entries.append(_entry_record(path, root, "file", ""))
            else:
                raise PackagingError(f"拒绝非常规文件: {path}")

    entries.sort(key=lambda item: str(item["relative_path"]))
    selected = {str(item["relative_path"]) for item in entries}
    missing = [relative for relative in REQUIRED_FILES if relative not in selected]
    if missing:
        raise PackagingError(f"必需文件被意外排除: {missing}")
    return entries, dict(sorted(excluded.items()))


def _entry_record(path: Path, root: Path, kind: str, link_target: str) -> Dict[str, object]:
    info = path.lstat()
    return {
        "relative_path": _relative_posix(path, root),
        "archive_member": PROJECT_NAME + "/" + _relative_posix(path, root),
        "kind": kind,
        "size_bytes": int(info.st_size) if kind == "file" else 0,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": int(info.st_mtime_ns),
        "link_target": link_target,
        "source_path": path,
    }


def _git_snapshot(root: Path) -> Dict[str, object]:
    git = shutil.which("git")
    if not git or not (root / ".git").exists():
        return {"available": False, "commit": "", "dirty": None, "changed_path_count": None}
    try:
        commit = subprocess.run(
            [git, "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_text = subprocess.run(
            [git, "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        changed = [line for line in status_text.splitlines() if line.strip()]
        return {"available": True, "commit": commit, "dirty": bool(changed), "changed_path_count": len(changed)}
    except subprocess.CalledProcessError:
        return {"available": False, "commit": "", "dirty": None, "changed_path_count": None}


def _write_source_manifest(path: Path, entries: Iterable[Mapping[str, object]]) -> None:
    fields = ("archive_member", "kind", "size_bytes", "mode", "mtime_ns", "link_target")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for entry in entries:
            writer.writerow({field: entry[field] for field in fields})


def _write_file_hashes(path: Path, entries: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            if entry["kind"] != "file":
                continue
            member = str(entry["archive_member"])
            value = _sha256(Path(entry["source_path"]))
            hashes[member] = value
            handle.write(f"{value}  {member}\n")
    return hashes


def _write_null_file_list(path: Path, entries: Sequence[Mapping[str, object]]) -> None:
    with path.open("wb") as handle:
        for entry in entries:
            handle.write(os.fsencode(str(entry["archive_member"])))
            handle.write(b"\0")


def _run_tar_create(root: Path, file_list: Path, archive: Path, log: Path) -> str:
    tar = shutil.which("tar")
    if not tar:
        raise PackagingError("找不到tar")
    if shutil.which("pigz"):
        compressor = "pigz -1"
    elif shutil.which("gzip"):
        compressor = "gzip -1"
    else:
        raise PackagingError("找不到pigz或gzip")
    command = [
        tar,
        "--format=posix",
        "--numeric-owner",
        "--owner=0",
        "--group=0",
        "--hard-dereference",
        "--no-recursion",
        "--null",
        "--verbatim-files-from",
        "-I",
        compressor,
        "-cf",
        str(archive),
        "-C",
        str(root.parent),
        "-T",
        str(file_list),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise PackagingError(f"tar失败，exit={completed.returncode}；见 {log}")
    return compressor


def _inspect_archive(
    archive: Path,
    contents_path: Path,
    expected_hashes: Mapping[str, str],
    expected_links: Mapping[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """Read every archived payload once and verify file/link contents."""

    members: List[str] = []
    hash_mismatches: List[str] = []
    link_mismatches: List[str] = []
    try:
        with tarfile.open(str(archive), mode="r:gz") as handle:
            for member in handle:
                name = member.name.rstrip("/")
                if not name:
                    continue
                members.append(name)
                if member.isreg():
                    expected = expected_hashes.get(name)
                    extracted = handle.extractfile(member)
                    if expected is None or extracted is None:
                        hash_mismatches.append(name)
                        continue
                    digest = hashlib.sha256()
                    while True:
                        chunk = extracted.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                    if digest.hexdigest() != expected:
                        hash_mismatches.append(name)
                elif member.issym():
                    if expected_links.get(name) != member.linkname:
                        link_mismatches.append(name)
                else:
                    link_mismatches.append(name + ":unexpected_type")
    except (tarfile.TarError, OSError) as exc:
        raise PackagingError(f"归档完整性检查失败: {exc}") from exc
    contents_path.write_text("\n".join(members) + "\n", encoding="utf-8")
    return members, hash_mismatches, link_mismatches


def _source_changes(entries: Sequence[Mapping[str, object]]) -> List[str]:
    changed: List[str] = []
    for entry in entries:
        path = Path(entry["source_path"])
        try:
            info = path.lstat()
        except OSError:
            changed.append(str(entry["relative_path"]) + ":missing")
            continue
        if stat.S_IMODE(info.st_mode) != int(entry["mode"]) or int(info.st_mtime_ns) != int(entry["mtime_ns"]):
            changed.append(str(entry["relative_path"]) + ":metadata")
            continue
        if entry["kind"] == "file" and int(info.st_size) != int(entry["size_bytes"]):
            changed.append(str(entry["relative_path"]) + ":size")
        elif entry["kind"] == "symlink":
            try:
                if os.readlink(str(path)) != str(entry["link_target"]):
                    changed.append(str(entry["relative_path"]) + ":link")
            except OSError:
                changed.append(str(entry["relative_path"]) + ":link-missing")
    return changed


def _verify_members(actual: Sequence[str], expected: Sequence[str]) -> Dict[str, object]:
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    unsafe = []
    forbidden = []
    forbidden_prefixes = tuple(PROJECT_NAME + "/" + prefix for prefix in EXCLUDED_PREFIXES)
    forbidden_exact = {PROJECT_NAME + "/" + value for value in EXCLUDED_EXACT}
    for member in actual:
        parsed = PurePosixPath(member)
        if parsed.is_absolute() or ".." in parsed.parts:
            unsafe.append(member)
        if member in forbidden_exact or any(_matches_prefix(member, prefix) for prefix in forbidden_prefixes):
            forbidden.append(member)
    return {
        "missing_members": missing,
        "unexpected_members": unexpected,
        "unsafe_members": sorted(unsafe),
        "forbidden_members": sorted(forbidden),
    }


def _write_readme(path: Path, archive_name: str) -> None:
    path.write_text(
        f"""Substain_GB 分析流程迁移包

本包只包含分析流程、配置模板、测试、模型、模板、第三方工具、wheel和可迁移环境归档。
不包含原始BIDS/Lesion、第一队列participants/metadata、derivatives结果、QC、日志、状态、历史archive或活动虚拟环境。

本次只执行压缩与校验。迁移完成后再确定新工作站的解压位置、路径配置和第二队列输入。
压缩包内部顶层目录仍为Substain，现有源码和配置内容按原样保留。

需要转移的文件位于本说明目录的上一级：
  {archive_name}
  {archive_name}.sha256
同时转移本说明所在的Substain_GB_manifest目录，以保留逐文件清单和验证报告。

在压缩包所在目录校验：
  sha256sum -c {archive_name}.sha256
""",
        encoding="utf-8",
    )


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def build_bundle(project_root: Path, output_parent: Path, dry_run: bool = False) -> Dict[str, object]:
    root = project_root.resolve()
    output = output_parent.resolve()
    validate_project(root)
    if output_parent.is_symlink() or not output.is_dir():
        raise PackagingError(f"输出父目录不存在、不是目录或是符号链接: {output_parent}")
    if _is_within(output, root) or output == root:
        raise PackagingError(f"输出目录不得位于源项目内部: {output}")

    archive_name = BUNDLE_NAME + ".tar.gz"
    final_archive = output / archive_name
    final_checksum = output / (archive_name + ".sha256")
    final_bundle = output / (BUNDLE_NAME + "_manifest")
    for target in (final_archive, final_checksum, final_bundle):
        if target.exists() or target.is_symlink():
            raise PackagingError(f"目标已存在，拒绝覆盖: {target}")

    environment_archives = verify_environment_archives(root)
    entries, excluded_counts = collect_entries(root)
    logical_size = sum(int(entry["size_bytes"]) for entry in entries if entry["kind"] == "file")
    plan = {
        "status": "dry-run" if dry_run else "building",
        "project_root": str(root),
        "output_parent": str(output),
        "archive_path": str(final_archive),
        "checksum_path": str(final_checksum),
        "bundle_path": str(final_bundle),
        "entry_count": len(entries),
        "regular_file_count": sum(entry["kind"] == "file" for entry in entries),
        "symlink_count": sum(entry["kind"] == "symlink" for entry in entries),
        "logical_size_bytes": logical_size,
        "excluded_rule_counts": excluded_counts,
        "environment_archives": environment_archives,
        "git": _git_snapshot(root),
    }
    if dry_run:
        return plan

    free_bytes = shutil.disk_usage(str(output)).free
    required_free = logical_size + ONE_GIB
    if free_bytes < required_free:
        raise PackagingError(
            f"输出磁盘空间不足；可用={free_bytes}，至少需要={required_free}（入选内容+1GiB余量）"
        )

    temporary_bundle = Path(tempfile.mkdtemp(prefix="." + BUNDLE_NAME + ".tmp-", dir=str(output)))
    archive = temporary_bundle / archive_name
    try:
        file_list = temporary_bundle / ".archive-members.null"
        _write_null_file_list(file_list, entries)
        _write_source_manifest(temporary_bundle / "SOURCE_MANIFEST.tsv", entries)
        print(f"入选条目={len(entries)}，逻辑大小={logical_size / ONE_GIB:.2f} GiB；开始计算逐文件SHA256。", flush=True)
        file_hashes = _write_file_hashes(temporary_bundle / "FILES.sha256", entries)
        print("开始生成压缩包；源项目保持只读。", flush=True)
        compressor = _run_tar_create(root, file_list, archive, temporary_bundle / "tar.log")
        contents_path = temporary_bundle / "CONTENTS.txt"
        expected_links = {
            str(entry["archive_member"]): str(entry["link_target"])
            for entry in entries
            if entry["kind"] == "symlink"
        }
        actual_members, payload_hash_mismatches, archived_link_mismatches = _inspect_archive(
            archive, contents_path, file_hashes, expected_links
        )
        verification = _verify_members(actual_members, [str(entry["archive_member"]) for entry in entries])
        source_changes = _source_changes(entries)
        archive_sha256 = _sha256(archive)
        status = "pass" if not any(verification.values()) and not payload_hash_mismatches and not archived_link_mismatches and not source_changes else "fail"
        report = dict(plan)
        report.update(
            {
                "status": status,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "bundle_name": BUNDLE_NAME,
                "archive": archive_name,
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": archive_sha256,
                "compressor": compressor,
                "actual_member_count": len(actual_members),
                "raw_inputs_included": False if not verification["forbidden_members"] else None,
                "analysis_results_included": False if not verification["forbidden_members"] else None,
                "first_queue_manifests_included": False if not verification["forbidden_members"] else None,
                "payload_hash_mismatches": payload_hash_mismatches,
                "archived_link_mismatches": archived_link_mismatches,
                "source_changes_during_packaging": source_changes,
                **verification,
            }
        )
        (temporary_bundle / "VERIFICATION.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        checksum_text = f"{archive_sha256}  {archive_name}\n"
        temporary_checksum = temporary_bundle / (archive_name + ".sha256")
        temporary_checksum.write_text(checksum_text, encoding="utf-8")
        (temporary_bundle / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")
        (temporary_bundle / "EXCLUDED_RULES.txt").write_text(
            "\n".join(
                ["排除目录/前缀:"]
                + ["  " + value for value in EXCLUDED_PREFIXES]
                + ["", "排除文件:"]
                + ["  " + value for value in EXCLUDED_EXACT]
                + ["", "实际命中计数:"]
                + [f"  {key}\t{value}" for key, value in excluded_counts.items()]
            )
            + "\n",
            encoding="utf-8",
        )
        _write_readme(temporary_bundle / "README_ZH.txt", archive_name)
        shutil.copy2(Path(__file__).resolve(), temporary_bundle / Path(__file__).name)
        file_list.unlink()
        if status != "pass":
            raise PackagingError("归档成员核验失败；见 {}".format(temporary_bundle / "VERIFICATION.json"))
        # Temporary and final files share a filesystem. Hard-link publication
        # fails if a target exists, so another invocation cannot be overwritten.
        os.link(str(archive), str(final_archive))
        os.link(str(temporary_checksum), str(final_checksum))
        archive.unlink()
        temporary_checksum.unlink()
        temporary_bundle.rename(final_bundle)
        return report
    except BaseException:
        print(f"打包失败；源项目未修改。诊断目录保留在 {temporary_bundle}", file=sys.stderr)
        raise


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--dry-run", action="store_true", help="只检查范围、依赖和预计大小，不创建文件")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = build_bundle(args.project_root, args.output_parent, dry_run=args.dry_run)
    except (PackagingError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "pass":
        print("迁移包生成并验证通过: {}".format(report["archive_path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
