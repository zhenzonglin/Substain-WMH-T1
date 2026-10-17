"""把BIDS或普通递归目录统一成无session的只读软链接输入视图。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from .resources import sha256
from .schema import ALLOWED_SEX, PARTICIPANT_COLUMNS


METADATA_COLUMNS = ["participant_id", "age", "sex", "site_id"]
_BIDS_IMAGE = re.compile(
    r"^sub-(?P<subject>[A-Za-z0-9]+)(?:_ses-(?P<session>[A-Za-z0-9]+))?"
    r"(?:_[A-Za-z0-9]+-[A-Za-z0-9]+)*_(?P<suffix>T1w|FLAIR)\.nii(?:\.gz)?$"
)
_BIDS_LABEL = re.compile(r"^[A-Za-z0-9]+$")


@dataclass(frozen=True)
class InputCase:
    """完成一一匹配、但尚未创建软链接的一例输入。"""

    participant_id: str
    t1w: Path
    flair: Path
    lesion: Path


def _absolute(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _normalise_id(value: object) -> str:
    """内部ID不带BIDS的``sub-``前缀，防止生成``sub-sub-*``。"""

    participant_id = str(value).strip()
    if participant_id.startswith("sub-"):
        participant_id = participant_id[4:]
    if not participant_id or not _BIDS_LABEL.fullmatch(participant_id):
        raise ValueError("participant_id 必须是非空BIDS字母数字标签: {!r}".format(value))
    return participant_id


def _read_metadata(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError("缺少输入元数据: {}".format(path))
    table = pd.read_csv(path, sep="\t", dtype=str)
    missing = [column for column in METADATA_COLUMNS if column not in table.columns]
    extra = [column for column in table.columns if column not in METADATA_COLUMNS]
    if missing or extra:
        raise ValueError("metadata.tsv 字段不符合契约；missing={} extra={}".format(missing, extra))
    if table[METADATA_COLUMNS].isna().any().any():
        raise ValueError("metadata.tsv 不允许空字段")
    records: Dict[str, Dict[str, object]] = {}
    for row in table.to_dict(orient="records"):
        participant_id = _normalise_id(row["participant_id"])
        if participant_id in records:
            raise ValueError("metadata.tsv participant_id 重复: {}".format(participant_id))
        sex = str(row["sex"]).strip().lower()
        if sex not in ALLOWED_SEX:
            raise ValueError("sex 只允许 female/male，收到 {!r}".format(row["sex"]))
        records[participant_id] = {
            "participant_id": participant_id,
            "age": float(str(row["age"]).strip()),
            "sex": sex,
            "site_id": str(row["site_id"]).strip(),
        }
    return records


def _scan_suffix(root: Path, suffix: str, modality: str) -> Dict[str, Path]:
    """递归扫描精确文件名后缀；重复ID必须显式失败。"""

    if not root.is_dir():
        raise FileNotFoundError("{} 根目录不存在: {}".format(modality, root))
    if not suffix or "/" in suffix or "\\" in suffix:
        raise ValueError("{} suffix 必须是非空文件名后缀".format(modality))
    matches: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name.endswith(suffix)):
        raw_id = path.name[: -len(suffix)]
        participant_id = _normalise_id(raw_id)
        if participant_id in matches:
            duplicates.setdefault(participant_id, [str(matches[participant_id])]).append(str(path.resolve()))
        else:
            matches[participant_id] = path.resolve()
    if duplicates:
        raise ValueError("{} 出现重复ID: {}".format(modality, duplicates))
    return matches


def _scan_bids(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """解析最小BIDS实体并拒绝多session或同模态多文件。"""

    if not root.is_dir():
        raise FileNotFoundError("BIDS根目录不存在: {}".format(root))
    grouped: Dict[str, Dict[str, List[Tuple[str, Path]]]] = {}
    for path in sorted(root.rglob("*.nii*")):
        match = _BIDS_IMAGE.fullmatch(path.name)
        if match is None:
            continue
        participant_id = _normalise_id(match.group("subject"))
        grouped.setdefault(participant_id, {}).setdefault(match.group("suffix"), []).append(
            (match.group("session") or "", path.resolve())
        )
    t1: Dict[str, Path] = {}
    flair: Dict[str, Path] = {}
    for participant_id, modalities in grouped.items():
        all_sessions = {session for values in modalities.values() for session, _ in values}
        if len(all_sessions) > 1:
            raise ValueError("{} 含多个BIDS session，不能展平: {}".format(participant_id, sorted(all_sessions)))
        for suffix, destination in (("T1w", t1), ("FLAIR", flair)):
            values = modalities.get(suffix, [])
            if len(values) > 1:
                raise ValueError("{} 的{}文件不唯一: {}".format(participant_id, suffix, [str(row[1]) for row in values]))
            if values:
                destination[participant_id] = values[0][1]
    return t1, flair


def _require_same_ids(collections: Mapping[str, Mapping[str, Path]], metadata_ids: Iterable[str]) -> List[str]:
    expected = set(metadata_ids)
    messages: List[str] = []
    for name, values in collections.items():
        observed = set(values)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            messages.append("{} missing={} extra={}".format(name, missing, extra))
    if messages:
        raise ValueError("输入ID无法严格一一匹配；" + "；".join(messages))
    return sorted(expected)


def _replace_symlink(link: Path, target: Path) -> None:
    """只替换本工具管理的软链接，绝不删除普通文件或目录。"""

    link.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(str(link)):
        if not link.is_symlink():
            raise RuntimeError("软链接目标位置已被普通文件占用: {}".format(link))
        if link.resolve() == target.resolve():
            return
        link.unlink()
    link.symlink_to(target.resolve())


def _write_tsv_if_changed(table: pd.DataFrame, output: Path) -> bool:
    """内容未变化时不触碰mtime，保证Snakemake可恢复且不会无故全量重跑。"""

    buffer = StringIO()
    table.to_csv(buffer, sep="\t", index=False)
    content = buffer.getvalue()
    if output.is_file() and output.read_text(encoding="utf-8") == content:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output)
    return True


def prepare_inputs(config: Mapping[str, object]) -> Dict[str, object]:
    """生成规范化软链接、participants.tsv和可追溯输入清单。"""

    project_root = Path(str(config["project_root"])).resolve()
    input_config = config.get("input")
    if not isinstance(input_config, Mapping):
        raise ValueError("config.yaml 缺少 input 配置")
    mode = str(input_config.get("mode", "")).strip().lower()
    metadata_path = _absolute(project_root, input_config.get("metadata_tsv", "config/metadata.tsv"))
    metadata = _read_metadata(metadata_path)
    lesion = _scan_suffix(
        _absolute(project_root, input_config["lesion_root"]),
        str(input_config["lesion_suffix"]),
        "lesion",
    )
    if mode == "bids":
        t1, flair = _scan_bids(_absolute(project_root, input_config["bids_root"]))
    elif mode == "folders":
        t1 = _scan_suffix(_absolute(project_root, input_config["t1_root"]), str(input_config["t1_suffix"]), "T1")
        flair = _scan_suffix(
            _absolute(project_root, input_config["flair_root"]), str(input_config["flair_suffix"]), "FLAIR"
        )
    else:
        raise ValueError("input.mode 只允许 bids/folders，收到 {!r}".format(mode))

    participant_ids = _require_same_ids({"T1": t1, "FLAIR": flair, "lesion": lesion}, metadata)
    view_root = _absolute(project_root, input_config.get("bids_links", "inputs/bids_links"))
    participants_path = _absolute(project_root, config["participants"])
    manifest_path = view_root / "input_manifest.tsv"
    participant_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    for participant_id in participant_ids:
        bids_id = "sub-{}".format(participant_id)
        anat = view_root / bids_id / "anat"
        t1_link = anat / "{}_T1w.nii.gz".format(bids_id)
        flair_link = anat / "{}_FLAIR.nii.gz".format(bids_id)
        _replace_symlink(t1_link, t1[participant_id])
        _replace_symlink(flair_link, flair[participant_id])
        row = metadata[participant_id]
        participant_rows.append(
            {
                **row,
                "t1w": str(t1_link.relative_to(project_root)),
                "flair": str(flair_link.relative_to(project_root)),
                "lesion_mask": str(lesion[participant_id]),
            }
        )
        for modality, source, link in (("T1w", t1[participant_id], t1_link), ("FLAIR", flair[participant_id], flair_link)):
            manifest_rows.append(
                {
                    "participant_id": participant_id,
                    "modality": modality,
                    "source": str(source),
                    "source_sha256": sha256(source),
                    "link": str(link),
                    "link_target": str(link.resolve()),
                }
            )
        manifest_rows.append(
            {
                "participant_id": participant_id,
                "modality": "lesion_MNI152",
                "source": str(lesion[participant_id]),
                "source_sha256": sha256(lesion[participant_id]),
                "link": "",
                "link_target": "",
            }
        )

    participants_changed = _write_tsv_if_changed(
        pd.DataFrame(participant_rows, columns=PARTICIPANT_COLUMNS), participants_path
    )
    manifest_changed = _write_tsv_if_changed(pd.DataFrame(manifest_rows), manifest_path)
    return {
        "status": "pass",
        "mode": mode,
        "participant_count": len(participant_ids),
        "participants": participant_ids,
        "participants_tsv": str(participants_path),
        "bids_links": str(view_root),
        "input_manifest": str(manifest_path),
        "participants_tsv_changed": participants_changed,
        "input_manifest_changed": manifest_changed,
        "raw_inputs_modified": False,
        "session_entities_removed_from_links": True,
    }
