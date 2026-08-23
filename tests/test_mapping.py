from pathlib import Path

import numpy as np

from substain_features.mapping import (
    aggregate_macro20,
    assert_volume_conservation,
    build_macro_mapping,
    official_gm119,
    validate_macro_mapping,
)


def test_muse_macro20_has_full_unique_gm119_coverage(project_root: Path) -> None:
    dictionaries = project_root / "resources/third_party/NiChart_DLMUSE/NiChart_DLMUSE/shared/dicts"
    table = build_macro_mapping(
        dictionaries / "MUSE_mapping_consecutive_indices.csv",
        dictionaries / "MUSE_mapping_derived_rois.csv",
    )
    report = validate_macro_mapping(table, official_gm119(dictionaries / "MUSE_mapping_derived_rois.csv"))
    assert report["status"] == "pass"
    assert report["gm119_covered"] == 119
    assert report["macro_count"] == 20
    assert report["mapping_version"] == "muse_macro20_v1_provisional"
    assert not table["native_label"].duplicated().any()


def test_macro20_volume_conservation(project_root: Path) -> None:
    dictionaries = project_root / "resources/third_party/NiChart_DLMUSE/NiChart_DLMUSE/shared/dicts"
    table = build_macro_mapping(
        dictionaries / "MUSE_mapping_consecutive_indices.csv",
        dictionaries / "MUSE_mapping_derived_rois.csv",
    )
    volumes = {int(label): float(label) + 0.25 for label in table["native_label"]}
    assert_volume_conservation(volumes, table)
    macro = aggregate_macro20(volumes, table)
    assert len(macro) == 20
    assert np.isclose(sum(macro.values()), sum(volumes.values()))
