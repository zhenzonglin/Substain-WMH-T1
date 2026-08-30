#!/usr/bin/env python3
"""从GB SAS临床表生成不含诊断排除条件的metadata.tsv。"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


REQUIRED_COLUMNS = ("C_code", "P_code", "AGE", "GENDER")
OUTPUT_COLUMNS = ("participant_id", "age", "sex", "site_id")


def _text(value: object) -> str:
    """把SAS字节串和普通标量转换为无首尾空格的文本。"""

    if isinstance(value, bytes):
        for encoding in ("utf-8", "gb18030", "latin1"):
            try:
                return value.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
    return str(value).strip()


def _id_component(value: object, prefix: str, width: int, column: str) -> str:
    """标准化C_code/P_code，保留并补齐数字部分的前导零。"""

    if pd.isna(value):
        raise ValueError("{}存在缺失值".format(column))
    text = _text(value).lower()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = str(int(float(text)))
    if text.startswith(prefix):
        text = text[1:]
    if not text.isdigit():
        raise ValueError("{}不是{}加数字的编码: {!r}".format(column, prefix.upper(), value))
    return prefix + text.zfill(width)


def build_metadata(source: pd.DataFrame) -> pd.DataFrame:
    """保留所有编码、年龄和性别均有效的GB记录，不应用诊断排除。"""

    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("SAS文件缺少字段: {}".format(missing))

    selected = source[list(REQUIRED_COLUMNS)].copy()
    selected["site_id"] = selected["C_code"].map(
        lambda value: _id_component(value, "c", 3, "C_code")
    )
    patient_code = selected["P_code"].map(
        lambda value: _id_component(value, "p", 5, "P_code")
    )
    selected["participant_id"] = selected["site_id"] + patient_code

    gender = pd.to_numeric(selected["GENDER"], errors="coerce")
    invalid_gender = ~gender.isin([1, 2])
    if invalid_gender.any():
        values = selected.loc[invalid_gender, "GENDER"].map(_text).unique().tolist()
        raise ValueError("GENDER只允许1/2，发现: {}".format(values))
    selected["sex"] = gender.map({1: "male", 2: "female"})

    age = pd.to_numeric(selected["AGE"], errors="coerce")
    invalid_age = ~age.map(lambda value: math.isfinite(value) and 0 < value <= 120)
    if invalid_age.any():
        values = selected.loc[invalid_age, "AGE"].map(_text).unique().tolist()
        raise ValueError("AGE存在缺失或非法值: {}".format(values))
    selected["age"] = age.astype(float)

    duplicate = selected["participant_id"].duplicated(keep=False)
    if duplicate.any():
        values = sorted(selected.loc[duplicate, "participant_id"].unique().tolist())
        raise ValueError("C_code与P_code组合后存在重复ID: {}".format(values))

    return selected[list(OUTPUT_COLUMNS)].sort_values("participant_id").reset_index(drop=True)


def _write_if_changed(table: pd.DataFrame, output: Path) -> bool:
    """原子写入metadata；内容未变化时保留mtime以支持断点续跑。"""

    content = table.to_csv(sep="\t", index=False)
    if output.is_file() and output.read_text(encoding="utf-8") == content:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output)
    return True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成GB队列metadata.tsv，不应用诊断排除")
    parser.add_argument("--sas-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "metadata.tsv",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sas_file = args.sas_file.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not sas_file.is_file():
        raise FileNotFoundError("SAS文件不存在: {}".format(sas_file))

    source = pd.read_sas(sas_file, format="sas7bdat")
    metadata = build_metadata(source)
    changed = _write_if_changed(metadata, output)
    print("GB SAS总记录: {}例".format(len(source)))
    print("GB有效且写入metadata: {}例".format(len(metadata)))
    print("metadata.tsv: {} ({})".format(output, "updated" if changed else "unchanged"))


if __name__ == "__main__":
    main()
