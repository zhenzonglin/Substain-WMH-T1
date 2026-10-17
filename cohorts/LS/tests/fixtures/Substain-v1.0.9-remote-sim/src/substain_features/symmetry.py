"""在ch2better物理空间执行确定性的对侧WMH替代。"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

from .registration import apply_transforms
from .resources import sha256


def label_lesion_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """按3×3×3全邻域（26邻域）标记独立急性病灶成分。"""

    return ndimage.label(
        np.asarray(mask) > 0,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )


def mirror_world_x_zero(data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按世界坐标x=0反射，不依赖数组轴方向或数组下标翻转。"""

    if data.ndim != 3:
        raise ValueError("对侧反射只支持3D数组")
    reflection = np.eye(4, dtype=float)
    reflection[0, 0] = -1.0
    voxel_mapping = np.linalg.inv(affine) @ reflection @ affine
    mirrored = ndimage.affine_transform(
        data.astype(np.uint8),
        matrix=voxel_mapping[:3, :3],
        offset=voxel_mapping[:3, 3],
        output_shape=data.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    valid = ndimage.affine_transform(
        np.ones(data.shape, dtype=np.uint8),
        matrix=voxel_mapping[:3, :3],
        offset=voxel_mapping[:3, 3],
        output_shape=data.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return mirrored > 0, valid > 0


def contralateral_rule_in_common_space(
    wmh: np.ndarray,
    acute_lesion: np.ndarray,
    lesion_components: np.ndarray,
    brain_mask: np.ndarray,
    affine: np.ndarray,
    triggered_component_ids: List[int],
) -> Dict[str, np.ndarray]:
    """在共同模板空间计算供体、冲突和替代掩膜。"""

    wmh_bool = np.asarray(wmh) > 0
    lesion_bool = np.asarray(acute_lesion) > 0
    brain_bool = np.asarray(brain_mask) > 0
    triggered = np.isin(np.rint(lesion_components).astype(np.int32), triggered_component_ids)
    mirror_wmh, mirror_valid = mirror_world_x_zero(wmh_bool, affine)
    mirror_lesion, _ = mirror_world_x_zero(lesion_bool, affine)
    mirror_brain, _ = mirror_world_x_zero(brain_bool, affine)
    reliable_space = mirror_valid & mirror_brain & brain_bool
    donor = triggered & mirror_wmh
    conflict = triggered & mirror_lesion
    out_of_brain = triggered & ~reliable_space
    replacement = donor & ~conflict & reliable_space
    return {
        "triggered": triggered,
        "mirror_wmh": mirror_wmh,
        "mirror_lesion": mirror_lesion,
        "donor": donor,
        "conflict": conflict,
        "out_of_brain": out_of_brain,
        "replacement": replacement,
    }


def compose_native_wmh(
    original_wmh: np.ndarray,
    triggered_lesion: np.ndarray,
    replacement: np.ndarray,
) -> np.ndarray:
    """只允许改写触发的病灶成分，病灶外保持逐体素完全一致。"""

    original = np.asarray(original_wmh) > 0
    triggered = np.asarray(triggered_lesion) > 0
    replacement_bool = (np.asarray(replacement) > 0) & triggered
    return (original & ~triggered) | replacement_bool


def _save_array(reference: Path, output: Path, data: np.ndarray, dtype: np.dtype = np.uint8) -> Path:
    image = nib.load(str(reference))
    header = image.header.copy()
    header.set_data_dtype(dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = nib.Nifti1Image(data.astype(dtype), image.affine, header)
    result.set_qform(image.affine, code=1)
    result.set_sform(image.affine, code=1)
    nib.save(result, str(output))
    return output


def _load_bool(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj) > 0


def _volume_ml(mask: np.ndarray, reference: Path) -> float:
    image = nib.load(str(reference))
    voxel_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0
    return float(np.count_nonzero(mask) * voxel_ml)


def run_contralateral_replacement(
    wmh_segmentation: Path,
    lesion_flair: Path,
    flair_brain_mask: Path,
    flair_reference: Path,
    template: Path,
    lesion_label: int,
    transforms_flair_to_mni: List[str],
    transforms_mni_to_flair: List[str],
    ants_bin: Optional[Path],
    output_dir: Path,
    log_path: Path,
) -> Dict[str, object]:
    """执行全自动替代，并始终在原生FLAIR体素中计算最终体积。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (wmh_segmentation, lesion_flair, flair_brain_mask, flair_reference, template):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    seg_image = nib.load(str(wmh_segmentation))
    original_wmh = np.rint(np.asanyarray(seg_image.dataobj)).astype(np.int32) == int(lesion_label)
    lesion_native = _load_bool(lesion_flair)
    if original_wmh.shape != lesion_native.shape:
        raise ValueError("WMH与FLAIR病灶不在同一原生网格")

    component_labels, component_count = label_lesion_components(lesion_native)
    triggered_ids = sorted(
        int(value)
        for value in np.unique(component_labels[original_wmh & lesion_native])
        if int(value) > 0
    )
    triggered_native = np.isin(component_labels, triggered_ids)

    original_native_path = _save_array(
        flair_reference, output_dir / "wmh_original_mask.nii.gz", original_wmh
    )
    components_native_path = _save_array(
        flair_reference,
        output_dir / "lesion_components.nii.gz",
        component_labels,
        np.int32,
    )
    triggered_native_path = _save_array(
        flair_reference, output_dir / "lesion_triggered_mask.nii.gz", triggered_native
    )

    native_inputs = {
        "wmh_mni": original_native_path,
        "lesion_mni": lesion_flair,
        "components_mni": components_native_path,
        "brain_mni": flair_brain_mask,
    }
    mni_paths: Dict[str, Path] = {}
    for key, source in native_inputs.items():
        target = output_dir / "{}_space-ch2better.nii.gz".format(key.replace("_mni", ""))
        apply_transforms(
            source,
            template,
            target,
            transforms_flair_to_mni,
            ants_bin,
            log_path,
            "NearestNeighbor",
        )
        mni_paths[key] = target

    template_image = nib.load(str(template))
    common = contralateral_rule_in_common_space(
        _load_bool(mni_paths["wmh_mni"]),
        _load_bool(mni_paths["lesion_mni"]),
        np.asanyarray(nib.load(str(mni_paths["components_mni"])).dataobj),
        _load_bool(mni_paths["brain_mni"]) & (np.asanyarray(template_image.dataobj) > 0),
        template_image.affine,
        triggered_ids,
    )

    common_paths: Dict[str, Path] = {}
    for key in ("mirror_wmh", "mirror_lesion", "donor", "conflict", "out_of_brain", "replacement"):
        common_paths[key] = _save_array(
            template,
            output_dir / "{}_space-ch2better_mask.nii.gz".format(key),
            common[key],
        )

    native_return: Dict[str, Path] = {}
    for key in ("donor", "conflict", "out_of_brain", "replacement"):
        target = output_dir / "{}_space-FLAIR_mask.nii.gz".format(key)
        apply_transforms(
            common_paths[key],
            flair_reference,
            target,
            transforms_mni_to_flair,
            ants_bin,
            log_path,
            "NearestNeighbor",
        )
        native_return[key] = target

    donor_native = _load_bool(native_return["donor"]) & triggered_native
    conflict_native = _load_bool(native_return["conflict"]) & triggered_native
    out_of_brain_native = _load_bool(native_return["out_of_brain"]) & triggered_native
    replacement_native = _load_bool(native_return["replacement"]) & triggered_native
    # 反向重采样后再次施加原生触发掩膜，保证病灶外WMH逐体素不变。
    final_wmh = compose_native_wmh(original_wmh, triggered_native, replacement_native)
    donor_native_path = _save_array(flair_reference, native_return["donor"], donor_native)
    conflict_native_path = _save_array(flair_reference, native_return["conflict"], conflict_native)
    out_of_brain_native_path = _save_array(
        flair_reference, native_return["out_of_brain"], out_of_brain_native
    )
    replacement_native_path = _save_array(
        flair_reference, native_return["replacement"], replacement_native
    )
    final_path = _save_array(
        flair_reference, output_dir / "wmh_corrected_mask.nii.gz", final_wmh
    )

    overlap = original_wmh & lesion_native
    original_removed = original_wmh & triggered_native & ~replacement_native
    metrics: Dict[str, object] = {
        "wmh_lesion_overlap_ml": _volume_ml(overlap, flair_reference),
        "wmh_contralateral_donor_ml": _volume_ml(donor_native, flair_reference),
        "wmh_replacement_added_ml": _volume_ml(replacement_native & ~original_wmh, flair_reference),
        "wmh_original_overlap_removed_ml": _volume_ml(original_removed, flair_reference),
        "wmh_bilateral_lesion_conflict_removed_ml": _volume_ml(
            donor_native & conflict_native, flair_reference
        ),
        "wmh_out_of_brain_donor_removed_ml": _volume_ml(
            original_wmh & triggered_native & out_of_brain_native, flair_reference
        ),
        "wmh_volume_before_correction_ml": _volume_ml(original_wmh, flair_reference),
        "wmh_volume_after_correction_ml": _volume_ml(final_wmh, flair_reference),
        "lesion_component_count": int(component_count),
        "triggered_lesion_component_count": len(triggered_ids),
        "triggered_lesion_component_ids": triggered_ids,
        "symmetry_space": "ch2better",
        "symmetry_plane_world_x_mm": 0.0,
        "manual_adjudication_applied": False,
        "fixed_dilation_applied": False,
    }
    output_paths = {
        "original_wmh": str(original_native_path),
        "lesion_components": str(components_native_path),
        "triggered_lesion": str(triggered_native_path),
        "wmh_template": str(mni_paths["wmh_mni"]),
        "lesion_template": str(mni_paths["lesion_mni"]),
        "mirror_wmh_template": str(common_paths["mirror_wmh"]),
        "mirror_lesion_template": str(common_paths["mirror_lesion"]),
        "donor_template": str(common_paths["donor"]),
        "conflict_template": str(common_paths["conflict"]),
        "out_of_brain_template": str(common_paths["out_of_brain"]),
        "replacement_template": str(common_paths["replacement"]),
        "donor_native_flair": str(donor_native_path),
        "conflict_native_flair": str(conflict_native_path),
        "out_of_brain_native_flair": str(out_of_brain_native_path),
        "replacement_native_flair": str(replacement_native_path),
        "final_wmh": str(final_path),
    }
    hashes = {key: sha256(Path(value)) for key, value in output_paths.items()}
    hashes.update(
        {
            "flair_reference": sha256(flair_reference),
            "t1_to_flair_and_mni_transform_chain": sha256_transforms(
                transforms_flair_to_mni + transforms_mni_to_flair
            ),
        }
    )
    return dict(output_paths, output_sha256=hashes, **metrics)


def sha256_transforms(transforms: List[str]) -> Dict[str, str]:
    """记录变换文件哈希；ANTs逆矩阵语法中的方括号不影响文件定位。"""

    result: Dict[str, str] = {}
    for value in transforms:
        cleaned = value.strip("[]")
        if cleaned.endswith(",1"):
            cleaned = cleaned[:-2]
        path = Path(cleaned)
        if path.is_file():
            result[str(path)] = sha256(path)
    return result
