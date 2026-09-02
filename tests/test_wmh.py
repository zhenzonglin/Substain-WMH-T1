from pathlib import Path
from types import SimpleNamespace
import sys

import nibabel as nib
import numpy as np
import pytest

from substain_features.wmh import WMH_FEATURES, _find_residual_arrays, chung_zscore, extract_wmh20_ml, run_wmh_synthseg


def test_chung_formula_matches_matlab(project_root: Path) -> None:
    residual = project_root / "resources/normative/Residual_Info.mat"
    if not residual.is_file():
        pytest.skip("源码发布不包含受限制Residual_Info.mat")
    male, female = _find_residual_arrays(residual)
    volumes = {name: float(index + 1) / 10.0 for index, name in enumerate(WMH_FEATURES)}
    observed = chung_zscore(volumes, "female", residual)
    expected = (np.asarray(list(volumes.values())) - female[:, 0]) / female[:, 1]
    assert np.allclose(list(observed.values()), expected)


def test_wmh_order_and_ml_are_native_voxel_based(tmp_path: Path) -> None:
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    atlas = np.zeros((20, 1, 1), dtype=np.int16)
    wmh = np.zeros_like(atlas, dtype=np.uint8)
    for index in range(20):
        atlas[index, 0, 0] = index + 1
        wmh[index, 0, 0] = 1
    atlas_path = tmp_path / "atlas.nii.gz"
    wmh_path = tmp_path / "wmh.nii.gz"
    nib.save(nib.Nifti1Image(atlas, affine), atlas_path)
    nib.save(nib.Nifti1Image(wmh, affine), wmh_path)
    features = extract_wmh20_ml(wmh_path, atlas_path)
    assert list(features) == WMH_FEATURES
    assert all(np.isclose(value, 0.005) for value in features.values())


def test_wmh_volume_is_unchanged_for_subthreshold_header_rounding(tmp_path: Path) -> None:
    affine = np.diag([0.4688, 0.4688, 7.0, 1.0])
    rounded_affine = affine.copy()
    rounded_affine[0, 3] += 0.049
    atlas = np.arange(1, 21, dtype=np.int16).reshape((20, 1, 1))
    wmh = np.ones_like(atlas, dtype=np.uint8)
    exact_atlas = tmp_path / "atlas_exact.nii.gz"
    rounded_atlas = tmp_path / "atlas_rounded.nii.gz"
    wmh_path = tmp_path / "wmh.nii.gz"
    nib.save(nib.Nifti1Image(atlas, affine), exact_atlas)
    nib.save(nib.Nifti1Image(atlas, rounded_affine), rounded_atlas)
    nib.save(nib.Nifti1Image(wmh, affine), wmh_path)

    exact = extract_wmh20_ml(wmh_path, exact_atlas)
    diagnostics = {}
    tolerated = extract_wmh20_ml(wmh_path, rounded_atlas, grid_details=diagnostics)

    assert tolerated == exact
    assert diagnostics["matches"] is True
    assert diagnostics["max_corner_displacement_mm"] < 0.05


def test_wmh_grid_error_reports_physical_diagnostics(tmp_path: Path) -> None:
    affine = np.eye(4)
    shifted = affine.copy()
    shifted[0, 3] = 0.051
    atlas = np.arange(1, 21, dtype=np.int16).reshape((20, 1, 1))
    wmh = np.ones_like(atlas, dtype=np.uint8)
    atlas_path = tmp_path / "atlas.nii.gz"
    wmh_path = tmp_path / "wmh.nii.gz"
    nib.save(nib.Nifti1Image(atlas, shifted), atlas_path)
    nib.save(nib.Nifti1Image(wmh, affine), wmh_path)

    with np.testing.assert_raises_regex(ValueError, "max_corner_displacement_mm"):
        extract_wmh20_ml(wmh_path, atlas_path)


def test_wmh_subprocess_inherits_current_interpreter(tmp_path: Path, monkeypatch) -> None:
    """防止 core/WMH 隔离环境因 PATH 顺序串用 Python。"""

    source_root = tmp_path / "third_party" / "repo"
    runtime = tmp_path / "runtime" / "wmh_synthseg_inference.py"
    model = tmp_path / "model.pth"
    flair = tmp_path / "flair.nii.gz"
    output = tmp_path / "seg.nii.gz"
    for path in (source_root, runtime.parent):
        path.mkdir(parents=True, exist_ok=True)
    for path in (runtime, model, flair):
        path.write_bytes(b"test")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output.write_bytes(b"seg")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("substain_features.wmh.subprocess.run", fake_run)
    run_wmh_synthseg(flair, output, tmp_path / "vol.csv", model, source_root, "cpu", tmp_path / "run.log")
    assert captured["command"][0] == sys.executable


def test_wmh_cuda_uses_crop_and_fp16(tmp_path: Path, monkeypatch) -> None:
    """16 GB GPU 路径必须显式启用裁剪与半精度，避免回退到超显存 FP32。"""

    source_root = tmp_path / "third_party" / "repo"
    runtime = tmp_path / "runtime" / "wmh_synthseg_inference.py"
    model = tmp_path / "model.pth"
    flair = tmp_path / "flair.nii.gz"
    output = tmp_path / "seg.nii.gz"
    for path in (source_root, runtime.parent):
        path.mkdir(parents=True, exist_ok=True)
    for path in (runtime, model, flair):
        path.write_bytes(b"test")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output.write_bytes(b"seg")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("substain_features.wmh.subprocess.run", fake_run)
    run_wmh_synthseg(flair, output, tmp_path / "vol.csv", model, source_root, "cuda", tmp_path / "run.log")
    assert "--crop" in captured["command"]
    assert "--gpu_fp16" in captured["command"]
