"""MUSE 原生标签到 20 个双侧宏区的版本化映射。"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd


MAPPING_VERSION = "muse_macro20_v1_provisional"
CORTICAL_DERIVED_IDS = list(range(301, 316))
DEEP_GROUPS: Sequence[Tuple[str, str, str, Sequence[int]]] = [
    ("thalamus", "丘脑", "Thalamus", [59, 60]),
    ("striatopallidal", "纹状体-苍白球", "Striatum-pallidum", [23, 30, 36, 37, 55, 56, 57, 58]),
    ("hippocampus_amygdala", "海马-杏仁核", "Hippocampus-amygdala", [31, 32, 47, 48]),
    (
        "basal_forebrain_ventraldc_brainstem",
        "基底前脑-腹侧间脑-脑干",
        "Basal forebrain-ventral diencephalon-brainstem",
        [35, 61, 62, 75, 76],
    ),
    ("cerebellar_cortex_vermis", "小脑皮层-蚓部", "Cerebellar cortex-vermis", [38, 39, 71, 72, 73]),
]

CORTICAL_NAMES: Sequence[Tuple[str, str, str]] = [
    ("frontal_inferior", "额下", "Inferior frontal"),
    ("frontal_insular", "额-岛", "Fronto-insular"),
    ("frontal_lateral", "额外侧", "Lateral frontal"),
    ("frontal_medial", "额内侧", "Medial frontal"),
    ("frontal_opercular", "额盖", "Frontal opercular"),
    ("cingulate", "扣带", "Cingulate"),
    ("medial_temporal", "内侧颞", "Medial temporal"),
    ("occipital_inferior", "枕下", "Inferior occipital"),
    ("occipital_lateral", "枕外侧", "Lateral occipital"),
    ("occipital_medial", "枕内侧", "Medial occipital"),
    ("parietal_lateral", "顶外侧", "Lateral parietal"),
    ("parietal_medial", "顶内侧", "Medial parietal"),
    ("temporal_inferior", "颞下", "Inferior temporal"),
    ("temporal_lateral", "颞外侧", "Lateral temporal"),
    ("temporal_superior", "上颞", "Superior temporal"),
]


def read_derived_mapping(path: Path) -> Dict[int, List[int]]:
    """读取 NiChart 官方派生 ROI CSV；该文件没有表头。"""

    result: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            result[int(row[0])] = [int(value) for value in row[2:]]
    return result


def read_native_names(path: Path) -> Dict[int, str]:
    table = pd.read_csv(path)
    return {int(row.IndexMUSE): str(row.ROINameMUSE) for row in table.itertuples()}


def official_gm119(path: Path) -> Set[int]:
    """NiChart 官方 `GM` 派生行（ID 601）定义的 119 个原生标签。"""

    labels = set(read_derived_mapping(path)[601])
    if len(labels) != 119:
        raise ValueError("NiChart GM 映射应含 119 个标签，实际 {}".format(len(labels)))
    return labels


def build_macro_mapping(native_path: Path, derived_path: Path) -> pd.DataFrame:
    """从固定 NiChart 映射构建 label-level TSV，避免人工改写皮层标签。"""

    native_names = read_native_names(native_path)
    derived = read_derived_mapping(derived_path)
    gm_labels = official_gm119(derived_path)
    macro_specs: List[Tuple[str, str, str, Sequence[int], str]] = []
    for derived_id, names in zip(CORTICAL_DERIVED_IDS, CORTICAL_NAMES):
        macro_specs.append((names[0], names[1], names[2], derived[derived_id], "NiChart_DLMUSE official derived ROI {}".format(derived_id)))
    for macro_id, name_cn, name_en, labels in DEEP_GROUPS:
        macro_specs.append((macro_id, name_cn, name_en, labels, "project-defined noncortical group v1 provisional"))

    rows = []
    for macro_index, (macro_id, name_cn, name_en, labels, source) in enumerate(macro_specs, start=1):
        for label in labels:
            if label not in native_names:
                raise ValueError("映射引用未知 MUSE 标签 {}".format(label))
            rows.append(
                {
                    "mapping_version": MAPPING_VERSION,
                    "macro_index": macro_index,
                    "macro_id": macro_id,
                    "macro_name_cn": name_cn,
                    "macro_name_en": name_en,
                    "native_label": label,
                    "native_name": native_names[label],
                    "in_official_gm119": label in gm_labels,
                    "mapping_source": source,
                    "mapping_tier": "official_NiChart_cortical" if macro_index <= 15 else "project_defined_noncortical",
                    "mapping_note": (
                        "heterogeneous full-coverage group; brainstem and bilateral ventral diencephalon are outside official GM119"
                        if macro_id == "basal_forebrain_ventraldc_brainstem"
                        else ""
                    ),
                }
            )
    table = pd.DataFrame(rows)
    validate_macro_mapping(table, gm_labels)
    return table


def validate_macro_mapping(table: pd.DataFrame, gm_labels: Iterable[int]) -> Dict[str, object]:
    """检查 GM119 无遗漏/无重复以及 20 组顺序。"""

    duplicated = sorted(table.loc[table["native_label"].duplicated(), "native_label"].astype(int).unique().tolist())
    mapped = set(table["native_label"].astype(int))
    gm = set(int(value) for value in gm_labels)
    missing = sorted(gm - mapped)
    macro_indices = sorted(table["macro_index"].astype(int).unique().tolist())
    if duplicated or missing or macro_indices != list(range(1, 21)):
        raise ValueError(
            "MUSE→20 映射失败：duplicates={} missing_gm={} macro_indices={}".format(duplicated, missing, macro_indices)
        )
    return {
        "mapping_version": str(table["mapping_version"].iloc[0]),
        "gm119_covered": len(gm),
        "mapped_native_labels": len(mapped),
        "extra_non_gm_labels": sorted(mapped - gm),
        "macro_count": len(macro_indices),
        "status": "pass",
    }


def aggregate_macro20(native_volumes: Mapping[int, float], mapping_table: pd.DataFrame) -> Dict[str, float]:
    """先求和原始体积；禁止平均原生 ROI 的 z 分数。"""

    output: Dict[str, float] = {}
    for macro_id, group in mapping_table.sort_values("macro_index").groupby("macro_id", sort=False):
        labels = group["native_label"].astype(int).tolist()
        absent = [label for label in labels if label not in native_volumes]
        if absent:
            raise ValueError("{} 缺少原生标签 {}".format(macro_id, absent))
        output[str(macro_id)] = float(sum(float(native_volumes[label]) for label in labels))
    if len(output) != 20:
        raise ValueError("宏区输出不是 20 维")
    return output


def assert_volume_conservation(native_volumes: Mapping[int, float], mapping_table: pd.DataFrame, atol: float = 1e-6) -> None:
    """宏区总和必须等于映射所覆盖原生标签的总和。"""

    macro = aggregate_macro20(native_volumes, mapping_table)
    labels = mapping_table["native_label"].astype(int).tolist()
    expected = sum(float(native_volumes[label]) for label in labels)
    if not np.isclose(sum(macro.values()), expected, rtol=0.0, atol=atol):
        raise ValueError("宏区体积不守恒")


def write_macro_mapping(native_path: Path, derived_path: Path, output: Path) -> Dict[str, object]:
    table = build_macro_mapping(native_path, derived_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False)
    return validate_macro_mapping(table, official_gm119(derived_path))
