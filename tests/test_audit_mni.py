from pathlib import Path

import nibabel as nib
import numpy as np

from substain_features.audit import _audit_participant, _mni152_grid
from substain_features.schema import Participant


def test_fsl_mni152_one_and_two_mm_grids_are_accepted(project_root: Path, tmp_path: Path) -> None:
    for resolution in ("1mm", "2mm"):
        reference = nib.load(str(project_root / "resources" / "templates" / "fsl" / "MNI152_T1_{}.nii.gz".format(resolution)))
        data = np.zeros(reference.shape, dtype=np.uint8)
        data[tuple(value // 2 for value in reference.shape)] = 1
        lesion = tmp_path / "lesion_{}.nii.gz".format(resolution)
        image = nib.Nifti1Image(data, reference.affine, reference.header)
        image.set_qform(reference.affine, code=4)
        image.set_sform(reference.affine, code=4)
        nib.save(image, str(lesion))
        result = _mni152_grid(lesion, project_root)
        assert result["errors"] == []
        assert result["resolution"] == resolution


def test_nonbinary_mni_lesion_is_rejected(project_root: Path, tmp_path: Path) -> None:
    reference = nib.load(str(project_root / "resources" / "templates" / "fsl" / "MNI152_T1_2mm.nii.gz"))
    data = np.zeros(reference.shape, dtype=np.float32)
    data[45, 54, 45] = 0.5
    lesion = tmp_path / "lesion.nii.gz"
    image = nib.Nifti1Image(data, reference.affine, reference.header)
    image.set_qform(reference.affine, code=4)
    image.set_sform(reference.affine, code=4)
    nib.save(image, str(lesion))
    t1 = tmp_path / "t1.nii.gz"
    flair = tmp_path / "flair.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), str(t1))
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), str(flair))
    participant = Participant("A01", 60.0, "female", "X", t1, flair, lesion)
    report = _audit_participant(participant, project_root)
    assert report["status"] == "fail"
    assert any("严格0/1" in value for value in report["errors"])
