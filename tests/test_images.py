from pathlib import Path

import nibabel as nib
import numpy as np

from substain_features.images import (
    _orthogonal_qc_views,
    resample_continuous_to_reference,
    resample_label_to_reference,
    same_grid,
)


def test_lesion_resampling_uses_world_coordinates(tmp_path: Path) -> None:
    source_data = np.zeros((5, 5, 5), dtype=np.uint8)
    source_data[2, 2, 2] = 1
    source_affine = np.eye(4)
    source_affine[:3, 3] = [10.0, 20.0, 30.0]
    reference_affine = np.diag([0.5, 0.5, 0.5, 1.0])
    reference_affine[:3, 3] = [10.0, 20.0, 30.0]
    source = tmp_path / "lesion.nii.gz"
    reference = tmp_path / "t1.nii.gz"
    output = tmp_path / "lesion_t1.nii.gz"
    nib.save(nib.Nifti1Image(source_data, source_affine), source)
    nib.save(nib.Nifti1Image(np.zeros((9, 9, 9)), reference_affine), reference)
    resample_label_to_reference(source, reference, output)
    assert same_grid(output, reference)
    assert np.count_nonzero(nib.load(output).get_fdata()) > 0


def test_continuous_resampling_restores_reference_grid(tmp_path: Path) -> None:
    source = tmp_path / "prob_crop.nii.gz"
    reference = tmp_path / "flair.nii.gz"
    output = tmp_path / "prob_native.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), reference)
    resample_continuous_to_reference(source, reference, output)
    assert same_grid(output, reference)
    values = nib.load(output).get_fdata()
    assert values.min() >= 0.0 and values.max() <= 1.0



def test_qc_views_use_radiological_orientation_and_physical_spacing() -> None:
    source = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    views = _orthogonal_qc_views(source, (1, 1, 2), (1.0, 2.0, 5.0))
    assert [view[0] for view in views] == ["Coronal", "Sagittal", "Axial"]
    np.testing.assert_array_equal(views[0][1], np.rot90(np.fliplr(source[:, 1, :].T)))
    np.testing.assert_array_equal(views[1][1], np.rot90(np.fliplr(source[1, :, :].T)))
    np.testing.assert_array_equal(views[2][1], np.rot90(np.fliplr(source[:, :, 2].T)))
    assert [(view[2], view[3]) for view in views] == [(1.0, 5.0), (2.0, 5.0), (1.0, 2.0)]
    assert [view[4] for view in views] == [("S", "I", "L", "R"), ("S", "I", "P", "A"), ("A", "P", "L", "R")]
