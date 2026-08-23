"""ANTs 配准与 MNI Chung 图谱反向变换。"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from .resources import sha256


def _tool(name: str, ants_bin: Optional[Path]) -> str:
    if ants_bin is not None and (ants_bin / name).is_file():
        return str(ants_bin / name)
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError("找不到 ANTs 工具 {}".format(name))
    return resolved


def _run(command: List[str], log_handle: object, env: Optional[Dict[str, str]] = None) -> None:
    completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError("命令失败 exit={}: {}".format(completed.returncode, " ".join(command)))


def _harmonize_atlas_xforms(atlas: Path, template: Path, output: Path) -> Path:
    """Chung 图谱 qform_code=0；补齐与 ch2better 一致的 qform，避免 ITK 误读物理空间。"""

    atlas_image = nib.load(str(atlas))
    template_image = nib.load(str(template))
    if atlas_image.shape[:3] != template_image.shape[:3] or not np.allclose(atlas_image.affine, template_image.affine, atol=1e-5):
        raise ValueError("Chung 图谱与 ch2better 数组网格/仿射不一致")
    labels = sorted(int(value) for value in np.unique(np.asanyarray(atlas_image.dataobj)) if value > 0)
    if labels != list(range(1, 21)):
        raise ValueError("Chung 图谱标签不是 1..20")
    header = atlas_image.header.copy()
    result = nib.Nifti1Image(np.asanyarray(atlas_image.dataobj), template_image.affine, header)
    result.set_qform(template_image.affine, code=1)
    result.set_sform(template_image.affine, code=1)
    nib.save(result, str(output))
    return output


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, value: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _brain_mask_qc(fixed_mask: Path, warped_mask: Path) -> Dict[str, float]:
    """用固定图像脑掩膜和变换后的移动脑掩膜计算自动QC。"""

    template_image = nib.load(str(fixed_mask))
    mask_image = nib.load(str(warped_mask))
    template_mask = np.asanyarray(template_image.dataobj) > 0
    moving_mask = np.asanyarray(mask_image.dataobj) > 0
    denominator = int(template_mask.sum()) + int(moving_mask.sum())
    dice = 0.0 if denominator == 0 else float(2 * np.logical_and(template_mask, moving_mask).sum() / denominator)
    if not template_mask.any() or not moving_mask.any():
        center_distance = float("inf")
    else:
        target_center = nib.affines.apply_affine(template_image.affine, np.argwhere(template_mask).mean(axis=0))
        moving_center = nib.affines.apply_affine(mask_image.affine, np.argwhere(moving_mask).mean(axis=0))
        center_distance = float(np.linalg.norm(target_center - moving_center))
    return {
        "brain_mask_dice": dice,
        "brain_mask_center_distance_mm": center_distance,
    }


def _ants_environment(ants_bin: Optional[Path]) -> Dict[str, str]:
    """构造离线ANTs运行环境，不依赖工作站的全局安装。"""

    runtime_env = os.environ.copy()
    if ants_bin is not None:
        runtime_env["PATH"] = "{}{}{}".format(ants_bin, os.pathsep, runtime_env.get("PATH", ""))
        ants_lib = ants_bin.parent / "lib"
        runtime_env["LD_LIBRARY_PATH"] = "{}{}{}".format(
            ants_lib, os.pathsep, runtime_env.get("LD_LIBRARY_PATH", "")
        )
    return runtime_env


def apply_transforms(
    input_image: Path,
    reference: Path,
    output: Path,
    transforms: List[str],
    ants_bin: Optional[Path],
    log_path: Path,
    interpolation: str = "NearestNeighbor",
) -> Path:
    """按明确的物理变换链重采样；标签默认使用最近邻。"""

    for path in (input_image, reference):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _tool("antsApplyTransforms", ants_bin),
        "-d", "3",
        "-i", str(input_image),
        "-r", str(reference),
        "-o", str(output),
        "-n", interpolation,
        "--verbose", "1",
    ]
    for transform in transforms:
        command.extend(["-t", str(transform)])
    with log_path.open("a", encoding="utf-8") as log_handle:
        _run(command, log_handle, _ants_environment(ants_bin))
    if not output.is_file():
        raise RuntimeError("ANTs未生成变换后图像: {}".format(output))
    return output


def register_and_warp_atlas(
    t1_brain: Path,
    t1_mask: Path,
    flair_brain: Path,
    flair_mask: Path,
    flair_reference: Path,
    template: Path,
    atlas_mni: Path,
    output_dir: Path,
    ants_bin: Optional[Path],
    log_path: Path,
    brain_mask_dice_min: float = 0.70,
    brain_mask_center_distance_max_mm: float = 15.0,
) -> Dict[str, object]:
    """始终执行T1→FLAIR刚体配准，再建立T1↔ch2better变换链。"""

    for path in (t1_brain, t1_mask, flair_brain, flair_mask, flair_reference, template, atlas_mni):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    synquick = _tool("antsRegistrationSyNQuick.sh", ants_bin)
    apply = _tool("antsApplyTransforms", ants_bin)
    runtime_env = _ants_environment(ants_bin)

    # 即使两个输入网格相同，也必须估计真实的T1→FLAIR六自由度刚体变换。
    t1_flair_prefix = output_dir / "t1_to_flair_"
    t1_flair = output_dir / "t1_to_flair_Warped.nii.gz"
    t1_to_flair_affine = output_dir / "t1_to_flair_0GenericAffine.mat"
    t1_flair_provenance_path = output_dir / "t1_to_flair_provenance.json"
    expected_t1_flair_provenance: Dict[str, object] = {
        "method": "antsRegistrationSyNQuick_rigid_6dof",
        "moving_t1_brain_sha256": sha256(t1_brain),
        "fixed_flair_brain_sha256": sha256(flair_brain),
    }
    existing_t1_flair = _read_json(t1_flair_provenance_path)
    reuse_t1_flair = bool(
        t1_flair.is_file()
        and t1_to_flair_affine.is_file()
        and all(existing_t1_flair.get(key) == value for key, value in expected_t1_flair_provenance.items())
    )

    t1_prefix = output_dir / "t1_to_ch2better_"
    t1_mni = output_dir / "t1_to_ch2better_Warped.nii.gz"
    atlas_native = output_dir / "MNI_ch2better_WM_20ROIs_space-FLAIR.nii.gz"
    atlas_for_ants = output_dir / "MNI_ch2better_WM_20ROIs_desc-xformsHarmonized.nii.gz"
    _harmonize_atlas_xforms(atlas_mni, template, atlas_for_ants)
    inverse_warp = output_dir / "t1_to_ch2better_1InverseWarp.nii.gz"
    forward_warp = output_dir / "t1_to_ch2better_1Warp.nii.gz"
    affine_transform = output_dir / "t1_to_ch2better_0GenericAffine.mat"
    provenance_path = output_dir / "registration_provenance.json"
    expected_provenance: Dict[str, object] = {
        "method": "antsRegistrationSyNQuick_s",
        "moving_t1_sha256": sha256(t1_brain),
        "fixed_template_sha256": sha256(template),
    }
    existing_provenance = _read_json(provenance_path)
    reuse_registration = bool(
        t1_mni.is_file()
        and inverse_warp.is_file()
        and forward_warp.is_file()
        and affine_transform.is_file()
        and all(existing_provenance.get(key) == value for key, value in expected_provenance.items())
    )

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            "\n=== registration invocation; reuse_t1_flair={}; reuse_t1_mni={} ===\n".format(
                reuse_t1_flair, reuse_registration
            )
        )
        if not reuse_t1_flair:
            _run(
                [
                    synquick, "-d", "3",
                    "-f", str(flair_brain),
                    "-m", str(t1_brain),
                    "-o", str(t1_flair_prefix),
                    "-t", "r",
                ],
                log_handle,
                runtime_env,
            )
            _write_json(t1_flair_provenance_path, expected_t1_flair_provenance)
        if not reuse_registration:
            _run([synquick, "-d", "3", "-f", str(template), "-m", str(t1_brain), "-o", str(t1_prefix), "-t", "s"], log_handle, runtime_env)
            _write_json(provenance_path, expected_provenance)

        # T1掩膜分别送往FLAIR和模板空间，用于两个独立的自动配准门控。
        t1_mask_flair = output_dir / "T1w_desc-SynthStrip_mask_space-FLAIR.nii.gz"
        _run(
            [
                apply, "-d", "3", "-i", str(t1_mask), "-r", str(flair_reference),
                "-o", str(t1_mask_flair), "-n", "NearestNeighbor",
                "-t", str(t1_to_flair_affine),
            ],
            log_handle,
            runtime_env,
        )
        warped_mask = output_dir / "T1w_desc-registrationInput_brain_mask_space-ch2better.nii.gz"
        _run(
            [
                apply,
                "-d", "3",
                "-i", str(t1_mask),
                "-r", str(template),
                "-o", str(warped_mask),
                "-n", "NearestNeighbor",
                "-t", str(forward_warp),
                "-t", str(affine_transform),
            ],
            log_handle,
            runtime_env,
        )
        # MNI→T1逆变换后，再用显式T1→FLAIR刚体矩阵进入原生FLAIR。
        transforms_mni_to_t1 = [
            "[{},1]".format(affine_transform),
            str(inverse_warp),
        ]
        transforms_mni_to_flair = [str(t1_to_flair_affine)] + transforms_mni_to_t1
        command = [apply, "-d", "3", "-i", str(atlas_for_ants), "-r", str(flair_reference), "-o", str(atlas_native), "-n", "NearestNeighbor", "--verbose", "1"]
        for transform in transforms_mni_to_flair:
            command.extend(["-t", transform])
        _run(command, log_handle, runtime_env)
    if not all(path.is_file() for path in (t1_flair, t1_to_flair_affine, t1_mni, atlas_native)):
        raise RuntimeError("ANTs未生成预期T1/FLAIR、T1/MNI或原生图谱输出")
    t1_flair_qc = _brain_mask_qc(flair_mask, t1_mask_flair)
    t1_mni_qc = _brain_mask_qc(template, warped_mask)
    t1_flair_qc_pass = bool(
        t1_flair_qc["brain_mask_dice"] >= brain_mask_dice_min
        and t1_flair_qc["brain_mask_center_distance_mm"] <= brain_mask_center_distance_max_mm
    )
    t1_mni_qc_pass = bool(
        t1_mni_qc["brain_mask_dice"] >= brain_mask_dice_min
        and t1_mni_qc["brain_mask_center_distance_mm"] <= brain_mask_center_distance_max_mm
    )
    transforms_flair_to_mni = [
        str(forward_warp),
        str(affine_transform),
        "[{},1]".format(t1_to_flair_affine),
    ]
    details = {
        "t1_flair": str(t1_flair),
        "t1_to_flair_affine": str(t1_to_flair_affine),
        "t1_mask_flair": str(t1_mask_flair),
        "t1_mni": str(t1_mni),
        "atlas_native_flair": str(atlas_native),
        "atlas_xforms_harmonized": str(atlas_for_ants),
        "registration_provenance": str(provenance_path),
        "t1_flair_registration_provenance": str(t1_flair_provenance_path),
        "t1_flair_registration_reused": reuse_t1_flair,
        "t1_mni_registration_reused": reuse_registration,
        "atlas_header_harmonization": "qform/sform copied from ch2better; voxel labels unchanged",
        "t1_flair_brain_mask_qc": t1_flair_qc,
        "t1_flair_brain_mask_qc_pass": t1_flair_qc_pass,
        "t1_mni_brain_mask_qc": t1_mni_qc,
        "t1_mni_brain_mask_qc_pass": t1_mni_qc_pass,
        "registration_brain_mask_dice_min": brain_mask_dice_min,
        "registration_brain_mask_center_distance_max_mm": brain_mask_center_distance_max_mm,
        "transforms_t1_to_flair": [str(t1_to_flair_affine)],
        "transforms_mni_to_t1": transforms_mni_to_t1,
        "transforms_flair_to_mni": transforms_flair_to_mni,
        "transforms_mni_to_flair": transforms_mni_to_flair,
        "transform_sha256": {
            str(path): sha256(path)
            for path in (t1_to_flair_affine, affine_transform, forward_warp, inverse_warp)
        },
    }
    if not t1_flair_qc_pass or not t1_mni_qc_pass:
        raise RuntimeError(
            "配准脑掩膜QC失败: T1-FLAIR dice={:.4f}, distance={:.2f}mm; "
            "T1-MNI dice={:.4f}, distance={:.2f}mm".format(
                t1_flair_qc["brain_mask_dice"],
                t1_flair_qc["brain_mask_center_distance_mm"],
                t1_mni_qc["brain_mask_dice"],
                t1_mni_qc["brain_mask_center_distance_mm"],
            )
        )
    return details
