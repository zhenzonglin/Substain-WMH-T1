#!/usr/bin/env python3
"""生成版本化 MUSE→20 映射；供安装脚本和手工复核调用。"""

from pathlib import Path

from substain_features.mapping import write_macro_mapping


ROOT = Path(__file__).resolve().parents[1]
report = write_macro_mapping(
    ROOT / "resources/third_party/NiChart_DLMUSE/NiChart_DLMUSE/shared/dicts/MUSE_mapping_consecutive_indices.csv",
    ROOT / "resources/third_party/NiChart_DLMUSE/NiChart_DLMUSE/shared/dicts/MUSE_mapping_derived_rois.csv",
    ROOT / "resources/mappings/muse_macro20_v1_provisional.tsv",
)
print(report)
