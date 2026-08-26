#!/usr/bin/env python3
"""从C3临床表和三个影像汇总目录生成metadata.tsv与participants.tsv。"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict

import pandas as pd


# ======================== 只需要填写下面4个路径 ========================
SAS7BDAT_PATH = ""
T1_FOLDER_PATH = ""
FLAIR_FOLDER_PATH = ""
LESION_FOLDER_PATH = ""
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_OUTPUT = PROJECT_ROOT / "config" / "metadata.tsv"
PARTICIPANTS_OUTPUT = PROJECT_ROOT / "config" / "participants.tsv"

SOURCE_COLUMNS = ["D_DIAG", "H_DEMEN", "H_DYSPHR", "H_EP", "code_n", "GENDER", "AGE", "site_code"]
METADATA_COLUMNS = ["participant_id", "age", "sex", "site_id"]
PARTICIPANT_COLUMNS = METADATA_COLUMNS + ["t1w", "flair", "lesion_mask"]
_BIDS_LABEL = re.compile(r"^[A-Za-z0-9]+$")


def _text(value: object) -> str:
    """兼容pandas读取SAS后可能得到的字节字符串。"""

    if isinstance(value, bytes):
        for encoding in ("utf-8", "gb18030", "latin1"):
            try:
                return value.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
    return str(value).strip()


def _plain_value(value: object) -> str:
    """把SAS中的整数型15.0写成15。"""

    text = _text(value)
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return str(int(float(text)))
    return text


def _participant_id(value: object, source: str) -> str:
    """项目内部ID不保留可选的sub-前缀，且只允许BIDS字母数字标签。"""

    if pd.isna(value):
        raise ValueError("{}中存在缺失participant_id".format(source))
    participant_id = _plain_value(value)
    if participant_id.startswith("sub-"):
        participant_id = participant_id[4:]
    if not participant_id or not _BIDS_LABEL.fullmatch(participant_id):
        raise ValueError("{}中的ID不是BIDS字母数字标签: {!r}".format(source, value))
    return participant_id


def _configured_path(name: str, value: str, expect_file: bool) -> Path:
    """解析脚本顶部的路径；空路径必须直接阻断，不能落到当前目录。"""

    if not value.strip():
        raise ValueError("请先填写脚本顶部的{}".format(name))
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    valid = path.is_file() if expect_file else path.is_dir()
    if not valid:
        expected = "文件" if expect_file else "文件夹"
        raise FileNotFoundError("{}不是有效{}: {}".format(name, expected, path))
    return path


def _scan_images(folder: Path, modality: str) -> Dict[str, Path]:
    """递归扫描NIfTI，并用文件名第一个下划线前的内容作为ID。"""

    images = sorted(
        path.resolve()
        for path in folder.rglob("*")
        if path.is_file() and (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz"))
    )
    if not images:
        raise ValueError("{}文件夹中没有NIfTI文件: {}".format(modality, folder))

    matched: Dict[str, Path] = {}
    duplicates: Dict[str, list] = {}
    invalid_names = []
    for path in images:
        if "_" not in path.name:
            invalid_names.append(str(path))
            continue
        raw_id = path.name.split("_", 1)[0]
        participant_id = _participant_id(raw_id, "{}文件名".format(modality))
        if participant_id in matched:
            duplicates.setdefault(participant_id, [str(matched[participant_id])]).append(str(path))
        else:
            matched[participant_id] = path

    if invalid_names:
        raise ValueError("{}文件名缺少下划线，无法提取ID: {}".format(modality, invalid_names))
    if duplicates:
        raise ValueError("{}中同一ID对应多个NIfTI文件: {}".format(modality, duplicates))
    return matched


def _clinical_metadata(source: pd.DataFrame) -> pd.DataFrame:
    """筛选脑梗死且无指定既往非血管性神经系统疾病的患者。"""

    missing = [column for column in SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("SAS文件缺少字段: {}".format(missing))

    numeric = source[["D_DIAG", "H_DEMEN", "H_DYSPHR", "H_EP"]].apply(
        pd.to_numeric, errors="coerce"
    )
    selected = source.loc[
        numeric["D_DIAG"].eq(1)
        & numeric["H_DEMEN"].eq(0)
        & numeric["H_DYSPHR"].eq(0)
        & numeric["H_EP"].eq(0),
        ["code_n", "GENDER", "AGE", "site_code"],
    ].copy()
    if selected.empty:
        raise ValueError("没有患者满足指定临床筛选条件")

    selected["participant_id"] = selected["code_n"].map(lambda value: _participant_id(value, "code_n"))

    gender = pd.to_numeric(selected["GENDER"], errors="coerce")
    invalid_gender = selected.loc[~gender.isin([1, 2]), "GENDER"].map(_text).unique().tolist()
    if invalid_gender:
        raise ValueError("筛选后患者的GENDER只允许1/2，发现: {}".format(invalid_gender))
    selected["sex"] = gender.map({1: "male", 2: "female"})

    age = pd.to_numeric(selected["AGE"], errors="coerce")
    invalid_age = ~age.map(lambda value: math.isfinite(value) and 0 < value <= 120)
    if invalid_age.any():
        raise ValueError("筛选后患者存在缺失或非法AGE: {}".format(selected.loc[invalid_age, "AGE"].tolist()))
    selected["age"] = age.map(
        lambda value: str(int(value)) if float(value).is_integer() else "{:g}".format(value)
    )

    selected["site_id"] = selected["site_code"].map(_plain_value)
    if selected["site_code"].isna().any() or selected["site_id"].eq("").any():
        raise ValueError("筛选后患者存在缺失site_code")

    duplicate = selected["participant_id"].duplicated(keep=False)
    if duplicate.any():
        raise ValueError(
            "筛选后code_n重复，需先确认临床表的一人一行: {}".format(
                sorted(selected.loc[duplicate, "participant_id"].unique().tolist())
            )
        )
    return selected[METADATA_COLUMNS].sort_values("participant_id").reset_index(drop=True)


def build_outputs(
    clinical: pd.DataFrame,
    t1_files: Dict[str, Path],
    flair_files: Dict[str, Path],
    lesion_files: Dict[str, Path],
) -> tuple:
    """只保留临床条件、T1、FLAIR和lesion五方均满足的患者。"""

    clinical_ids = set(clinical["participant_id"])
    included_ids = sorted(clinical_ids & set(t1_files) & set(flair_files) & set(lesion_files))
    if not included_ids:
        raise ValueError("临床合格患者与T1、FLAIR、lesion没有共同ID")

    metadata = clinical.loc[clinical["participant_id"].isin(included_ids), METADATA_COLUMNS].copy()
    metadata = metadata.sort_values("participant_id").reset_index(drop=True)
    participants = metadata.copy()
    participants["t1w"] = participants["participant_id"].map(lambda value: str(t1_files[value]))
    participants["flair"] = participants["participant_id"].map(lambda value: str(flair_files[value]))
    participants["lesion_mask"] = participants["participant_id"].map(lambda value: str(lesion_files[value]))
    return metadata, participants[PARTICIPANT_COLUMNS]


def _write_tsv(table: pd.DataFrame, output: Path) -> None:
    """使用同目录临时文件写入，完成后原子替换旧表。"""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(output)


def main() -> None:
    sas_path = _configured_path("SAS7BDAT_PATH", SAS7BDAT_PATH, expect_file=True)
    t1_folder = _configured_path("T1_FOLDER_PATH", T1_FOLDER_PATH, expect_file=False)
    flair_folder = _configured_path("FLAIR_FOLDER_PATH", FLAIR_FOLDER_PATH, expect_file=False)
    lesion_folder = _configured_path("LESION_FOLDER_PATH", LESION_FOLDER_PATH, expect_file=False)

    source = pd.read_sas(sas_path, format="sas7bdat")
    clinical = _clinical_metadata(source)
    t1_files = _scan_images(t1_folder, "T1")
    flair_files = _scan_images(flair_folder, "FLAIR")
    lesion_files = _scan_images(lesion_folder, "lesion")
    metadata, participants = build_outputs(clinical, t1_files, flair_files, lesion_files)

    _write_tsv(metadata, METADATA_OUTPUT)
    _write_tsv(participants, PARTICIPANTS_OUTPUT)

    clinical_ids = set(clinical["participant_id"])
    print("metadata.tsv: {}".format(METADATA_OUTPUT))
    print("participants.tsv: {}".format(PARTICIPANTS_OUTPUT))
    print("SAS总记录: {}例；临床条件合格: {}例".format(len(source), len(clinical)))
    print("T1: {}例；FLAIR: {}例；lesion: {}例".format(len(t1_files), len(flair_files), len(lesion_files)))
    print("五方均满足并最终保留: {}例".format(len(participants)))
    print("临床合格但缺T1: {}例".format(len(clinical_ids - set(t1_files))))
    print("临床合格但缺FLAIR: {}例".format(len(clinical_ids - set(flair_files))))
    print("临床合格但缺lesion: {}例".format(len(clinical_ids - set(lesion_files))))


if __name__ == "__main__":
    main()
