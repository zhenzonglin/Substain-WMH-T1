from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from substain_features.registration import apply_transforms


def test_known_rigid_translation_preserves_binary_lesion(project_root: Path, tmp_path: Path) -> None:
    """已知T1→FLAIR刚体位移后，病灶位置正确且仍为二值标签。"""

    ants_bin = project_root / "resources/tools/ants-2.5.4/bin"
    if not (ants_bin / "antsApplyTransforms").is_file():
        pytest.skip("项目ANTs副本不存在")
    source = tmp_path / "lesion_space-T1w.nii.gz"
    reference = tmp_path / "FLAIR.nii.gz"
    output = tmp_path / "lesion_space-FLAIR.nii.gz"
    transform = tmp_path / "known_t1_to_flair_affine.txt"
    data = np.zeros((7, 7, 7), dtype=np.uint8)
    data[2, 3, 3] = 1
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.zeros_like(data), np.eye(4)), reference)
    # antsApplyTransforms按该正向矩阵把体素中心沿输出x方向平移1 mm。
    transform.write_text(
        "#Insight Transform File V1.0\n"
        "#Transform 0\n"
        "Transform: MatrixOffsetTransformBase_double_3_3\n"
        "Parameters: 1 0 0 0 1 0 0 0 1 1 0 0\n"
        "FixedParameters: 0 0 0\n",
        encoding="utf-8",
    )
    apply_transforms(
        source,
        reference,
        output,
        [str(transform)],
        ants_bin,
        tmp_path / "ants.log",
        "NearestNeighbor",
    )
    observed = np.rint(nib.load(str(output)).get_fdata()).astype(np.uint8)
    assert set(np.unique(observed)).issubset({0, 1})
    assert observed.sum() == 1
    assert observed[3, 3, 3] == 1
