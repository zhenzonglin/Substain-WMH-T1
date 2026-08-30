"""把BIDS或普通递归目录统一成无session的只读软链接输入视图。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .resources import sha256
from .schema import ALLOWED_SEX, PARTICIPANT_COLUMNS


METADATA_COLUMNS = ["participant_id", "age", "sex", "site_id"]
_BIDS_IMAGE = re.compile(
    r"^sub-(?P<subject>[A-Za-z0-9]+)(?:_ses-(?P<session>[A-Za-z0-9]+))?"
    r"(?:_[A-Za-z0-9]+-[A-Za-z0-9]+)*_(?P<suffix>T1w|FLAIR)\.nii(?:\.gz)?$"
)
_BIDS_LABEL = re.compile(r"^[A-Za-z0-9]+$")
_RUN_ENTITY = re.compile(r"(?:^|_)run-(?P<run>[0-9]+)(?:_|$)", re.IGNORECASE)
_PLANE_NAMES = {0: "sagittal", 1: "coronal", 2: "axial"}
_PLANE_PRIORITY = {"axial": 0, "sagittal": 1, "coronal": 2, "oblique": 3, "unknown": 4}
BIDS_SELECTION_COLUMNS = [
    "participant_id",
    "modality",
    "selected",
    "rank",
    "path",
    "plane",
    "orientation_confidence",
    "orientation_source",
    "isotropic_3d",
    "shape",
    "zooms_mm",
    "voxel_volume_mm3",
    "minimum_fov_mm",
    "run_number",
]


@dataclass(frozen=True)
class InputCase:
    """完成一一匹配、但尚未创建软链接的一例输入。"""

    participant_id: str
    t1w: Path
    flair: Path
    lesion: Path


@dataclass(frozen=True)
class BidsCandidate:
    """只依据头信息得到的BIDS候选几何质量。"""

    path: Path
    plane: str
    orientation_confidence: float
    orientation_source: str
    isotropic_3d: bool
    shape: Tuple[int, int, int]
    zooms: Tuple[float, float, float]
    voxel_volume: float
    minimum_fov: float
    voxel_count: int
    run_number: int


def _absolute(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _normalise_id(value: object) -> str:
    """GB内部ID统一小写且不带``sub-``前缀。"""

    participant_id = str(value).strip()
    if participant_id.lower().startswith("sub-"):
        participant_id = participant_id[4:]
    participant_id = participant_id.lower()
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
    normalised_suffix = suffix.lower()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name.lower().endswith(normalised_suffix)
    ):
        raw_id = path.name[: -len(suffix)]
        participant_id = _normalise_id(raw_id)
        if participant_id in matches:
            duplicates.setdefault(participant_id, [str(matches[participant_id])]).append(str(path.resolve()))
        else:
            matches[participant_id] = path.resolve()
    if duplicates:
        raise ValueError("{} 出现重复ID: {}".format(modality, duplicates))
    return matches


def _json_sidecar(nifti: Path) -> Path:
    text = str(nifti)
    return Path(text[:-7] + ".json") if text.lower().endswith(".nii.gz") else nifti.with_suffix(".json")


def _plane_from_normal(normal: np.ndarray) -> Tuple[str, float]:
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm == 0:
        return "unknown", 0.0
    direction = np.abs(normal / norm)
    dominant_axis = int(np.argmax(direction))
    confidence = float(direction[dominant_axis])
    return (_PLANE_NAMES[dominant_axis] if confidence >= 0.8 else "oblique", confidence)


def _candidate_geometry(path: Path) -> BidsCandidate:
    """优先用DICOM方向；缺失时由NIfTI仿射和层厚估计采集平面。"""

    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ValueError("无法读取BIDS候选NIfTI头信息: {} ({})".format(path, exc)) from exc
    if len(image.shape) < 3:
        raise ValueError("BIDS候选不是三维影像: {} shape={}".format(path, image.shape))
    shape = tuple(int(value) for value in image.shape[:3])
    zooms_array = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if zooms_array.size != 3 or not np.all(np.isfinite(zooms_array)) or np.any(zooms_array <= 0):
        raise ValueError("BIDS候选体素尺寸非法: {} zooms={}".format(path, zooms_array.tolist()))
    zooms = tuple(float(value) for value in zooms_array)
    voxel_volume = float(np.prod(zooms_array))
    fov = np.asarray(shape, dtype=float) * zooms_array
    anisotropy = float(np.max(zooms_array) / np.min(zooms_array))
    isotropic_3d = bool(np.max(zooms_array) <= 2.0 and anisotropy <= 1.5 and min(shape) >= 32)

    metadata = {}
    sidecar = _json_sidecar(path)
    if sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("BIDS JSON无法读取: {} ({})".format(sidecar, exc)) from exc

    iop = metadata.get("ImageOrientationPatientDICOM")
    if isinstance(iop, list) and len(iop) == 6:
        try:
            row = np.asarray(iop[:3], dtype=float)
            column = np.asarray(iop[3:], dtype=float)
            plane, confidence = _plane_from_normal(np.cross(row, column))
            orientation_source = "json_iop"
        except (TypeError, ValueError):
            plane, confidence = "unknown", 0.0
            orientation_source = "invalid_json_iop"
    elif isotropic_3d:
        plane, confidence = "unknown", 0.0
        orientation_source = "isotropic_header"
    else:
        slice_axis = int(np.argmax(zooms_array))
        plane, confidence = _plane_from_normal(np.asarray(image.affine[:3, slice_axis], dtype=float))
        orientation_source = "nifti_affine"

    run_match = _RUN_ENTITY.search(path.name)
    run_number = int(run_match.group("run")) if run_match else 0
    return BidsCandidate(
        path=path.resolve(),
        plane=plane,
        orientation_confidence=confidence,
        orientation_source=orientation_source,
        isotropic_3d=isotropic_3d,
        shape=shape,
        zooms=zooms,
        voxel_volume=voxel_volume,
        minimum_fov=float(np.min(fov)),
        voxel_count=int(np.prod(np.asarray(shape, dtype=np.int64))),
        run_number=run_number,
    )


def _candidate_quality(candidate: BidsCandidate) -> Tuple[object, ...]:
    """3D近等体素优先；其余优先轴位；几何同分时默认run-2。"""

    return (
        0 if candidate.isotropic_3d else 1,
        0 if candidate.isotropic_3d else _PLANE_PRIORITY.get(candidate.plane, 4),
        round(candidate.voxel_volume, 8),
        -round(candidate.minimum_fov, 4),
        -candidate.voxel_count,
        0 if candidate.run_number == 2 else 1,
        candidate.run_number,
        str(candidate.path),
    )


def _choose_bids_candidate(
    participant_id: str,
    modality: str,
    paths: List[Path],
) -> Tuple[Path, List[Dict[str, object]]]:
    candidates = sorted((_candidate_geometry(path) for path in paths), key=_candidate_quality)
    selected = candidates[0]
    records: List[Dict[str, object]] = []
    for rank, candidate in enumerate(candidates, start=1):
        records.append(
            {
                "participant_id": participant_id,
                "modality": modality,
                "selected": candidate.path == selected.path,
                "rank": rank,
                "path": str(candidate.path),
                "plane": candidate.plane,
                "orientation_confidence": round(candidate.orientation_confidence, 6),
                "orientation_source": candidate.orientation_source,
                "isotropic_3d": candidate.isotropic_3d,
                "shape": "x".join(str(value) for value in candidate.shape),
                "zooms_mm": "x".join("{:g}".format(value) for value in candidate.zooms),
                "voxel_volume_mm3": round(candidate.voxel_volume, 6),
                "minimum_fov_mm": round(candidate.minimum_fov, 4),
                "run_number": candidate.run_number,
            }
        )
    return selected.path, records


def _scan_bids(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path], List[Dict[str, object]]]:
    """解析BIDS并对同session、同模态的多run进行可追溯几何择优。"""

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
    selection_rows: List[Dict[str, object]] = []
    for participant_id, modalities in grouped.items():
        all_sessions = {session for values in modalities.values() for session, _ in values}
        if len(all_sessions) > 1:
            raise ValueError("{} 含多个BIDS session，不能展平: {}".format(participant_id, sorted(all_sessions)))
        for suffix, destination in (("T1w", t1), ("FLAIR", flair)):
            values = modalities.get(suffix, [])
            if values:
                selected, records = _choose_bids_candidate(
                    participant_id,
                    suffix,
                    [row[1] for row in values],
                )
                destination[participant_id] = selected
                selection_rows.extend(records)
    return t1, flair, selection_rows


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
    selection_rows: List[Dict[str, object]] = []
    if mode == "bids":
        t1, flair, selection_rows = _scan_bids(_absolute(project_root, input_config["bids_root"]))
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
    selection_manifest_path = view_root / "bids_selection.tsv"
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
    selection_manifest_changed = _write_tsv_if_changed(
        pd.DataFrame(selection_rows, columns=BIDS_SELECTION_COLUMNS),
        selection_manifest_path,
    )
    return {
        "status": "pass",
        "mode": mode,
        "participant_count": len(participant_ids),
        "participants": participant_ids,
        "participants_tsv": str(participants_path),
        "bids_links": str(view_root),
        "input_manifest": str(manifest_path),
        "bids_selection": str(selection_manifest_path),
        "participants_tsv_changed": participants_changed,
        "input_manifest_changed": manifest_changed,
        "bids_selection_changed": selection_manifest_changed,
        "raw_inputs_modified": False,
        "session_entities_removed_from_links": True,
    }
