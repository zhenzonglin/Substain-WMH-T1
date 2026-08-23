from pathlib import Path

import nibabel as nib
import numpy as np

from substain_features.images import (
    _max_overlay_center,
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
    np.testing.assert_array_equal(views[0][1], np.fliplr(source[:, 1, :].T))
    np.testing.assert_array_equal(views[1][1], np.fliplr(source[1, :, :].T))
    np.testing.assert_array_equal(views[2][1], np.fliplr(source[:, :, 2].T))
    assert [(view[2], view[3]) for view in views] == [(5.0, 1.0), (5.0, 2.0), (2.0, 1.0)]
    assert [view[4] for view in views] == [("R", "L", "S", "I"), ("A", "P", "S", "I"), ("R", "L", "A", "P")]


def test_qc_slice_selection_uses_maximum_mask_plane() -> None:
    mask = np.zeros((7, 8, 9), dtype=np.uint8)
    mask[4, 1:7, 2:8] = 1
    mask[1:6, 5, 2:8] = 1
    mask[1:6, 1:7, 6] = 1
    x, y, z = _max_overlay_center(mask)
    assert x == int(np.argmax(mask.sum(axis=(1, 2))))
    assert y == int(np.argmax(mask.sum(axis=(0, 2))))
    assert z == int(np.argmax(mask.sum(axis=(0, 1))))
