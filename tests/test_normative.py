from pathlib import Path

import numpy as np
import pandas as pd

from substain_features.normative import GenMINDGlobalV1Provider


def test_t1_atrophy_direction_is_monotonic(project_root: Path) -> None:
    mapping_path = project_root / "resources/mappings/muse_macro20_v1_provisional.tsv"
    provider = GenMINDGlobalV1Provider(
        project_root / "resources/normative/genmind_dataset.csv",
        project_root / "resources/third_party/NiChart_DLMUSE/NiChart_DLMUSE/shared/dicts/MUSE_mapping_derived_rois.csv",
        mapping_path,
    )
    mapping = pd.read_csv(mapping_path, sep="\t")
    macro_ids = mapping.sort_values("macro_index")["macro_id"].drop_duplicates().astype(str).tolist()
    rng = np.random.default_rng(7)
    n = 400
    reference = pd.DataFrame(
        {
            "PTID": ["R{}".format(i) for i in range(n)],
            "Sex": ["F"] * n,
            "Race": ["White", "Black", "Asian", "Other"] * (n // 4),
            "Age": rng.uniform(55.0, 59.0, n),
            "sex_normalized": ["female"] * n,
        }
    )
    for macro_id in macro_ids:
        reference[macro_id] = rng.normal(-2.3, 0.08, n)
    provider._reference = reference
    baseline = {name: 100.0 for name in macro_ids}
    smaller = dict(baseline)
    smaller[macro_ids[0]] = 80.0
    z1 = provider.transform(baseline, 1000.0, 57.0, "female")
    z2 = provider.transform(smaller, 1000.0, 57.0, "female")
    feature = "t1_{}_atrophy_z".format(macro_ids[0])
    assert z1.eligible and z2.eligible
    assert z2.zscores[feature] > z1.zscores[feature]


def test_t1_normative_blocks_age_outside_contract(project_root: Path) -> None:
    provider = GenMINDGlobalV1Provider(
        project_root / "resources/normative/genmind_dataset.csv",
        project_root / "resources/third_party/NiChart_DLMUSE/NiChart_DLMUSE/shared/dicts/MUSE_mapping_derived_rois.csv",
        project_root / "resources/mappings/muse_macro20_v1_provisional.tsv",
    )
    result = provider.transform({}, 1.0, 91.0, "female")
    assert not result.eligible
    assert result.failure_reason == "age_outside_22_90"
