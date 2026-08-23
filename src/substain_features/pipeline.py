"""各处理阶段及最终表格导出；Snakemake 只负责依赖与重跑。"""

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .images import (
    resample_continuous_to_reference,
    resample_label_to_reference,
    same_grid,
    save_dual_overlay,
    save_overlay,
)
from .lowres import run_lowres_validation
from .mapping import write_macro_mapping
from .normative import GenMINDGlobalV1Provider
from .qc_review import load_review_table
from .registration import apply_transforms, register_and_warp_atlas
from .resources import sha256
from .schema import Participant
from .symmetry import run_contralateral_replacement
from .synthstrip import run_synthstrip
from .t1 import display_name_map, extract_t1_features, run_nichart_dlmuse, write_macro20_segmentation
from .wmh import WMH_FEATURES, chung_zscore, extract_wmh20_ml, run_wmh_synthseg


def _absolute(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def participant_dir(config: Mapping[str, object], participant: Participant) -> Path:
    root = Path(str(config["project_root"]))
    return _absolute(root, config["derivatives"]) / participant.bids_id


def qc_dir(config: Mapping[str, object]) -> Path:
    """返回全项目唯一QC目录；只展示当前正式输入分辨率。"""

    root = Path(str(config["project_root"]))
    default = _absolute(root, config["derivatives"]) / "qc"
    configured = config.get("qc_dir", default)
    return _absolute(root, configured)


def qc_path(
    config: Mapping[str, object], participant: Participant, index: int, label: str
) -> Path:
    """生成简洁且不会冲突的QC文件名。"""

    parts = [participant.participant_id, "{:02d}".format(index), label]
    return qc_dir(config) / ("_".join(parts) + ".png")


def status_path(config: Mapping[str, object], participant: Participant, stage: str) -> Path:
    return participant_dir(config, participant) / "status" / "{}.json".format(stage)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _force_binary(path: Path) -> Path:
    """把最近邻重采样后的任意正标签统一为0/1。"""

    image = nib.load(str(path))
    data = (np.asanyarray(image.dataobj) > 0).astype(np.uint8)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(data, image.affine, header)
    result.set_qform(image.affine, code=1)
    result.set_sform(image.affine, code=1)
    nib.save(result, str(path))
    return path


def stage_lesion(config: Mapping[str, object], participant: Participant) -> Dict[str, object]:
    """FSL MNI152病灶经ch2better物理坐标进入T1，再进入FLAIR。"""

    root = Path(str(config["project_root"]))
    output = participant_dir(config, participant) / "lesion"
    lesion_ch2better = output / "lesion_space-ch2better.nii.gz"
    lesion_t1 = output / "lesion_space-T1w.nii.gz"
    lesion_flair = output / "lesion_space-FLAIR.nii.gz"
    registration_status = _read_json(status_path(config, participant, "registration"))
    if registration_status.get("status") != "pass":
        raise RuntimeError("依赖节点registration失败，禁止转换病灶")
    source_image = nib.load(str(participant.lesion_mask))
    source_data = np.asanyarray(source_image.dataobj)
    if not np.isfinite(source_data).all() or not set(float(value) for value in np.unique(source_data)).issubset({0.0, 1.0}):
        raise ValueError("病灶输入必须是有限的严格0/1二值掩膜")
    fsl_references = [
        _absolute(root, config["wmh"]["fsl_mni152_1mm"]),  # type: ignore[index]
        _absolute(root, config["wmh"]["fsl_mni152_2mm"]),  # type: ignore[index]
    ]
    matches = [path for path in fsl_references if same_grid(participant.lesion_mask, path)]
    if len(matches) != 1:
        raise ValueError("病灶输入不匹配FSL MNI152标准1mm或2mm网格")
    template = _absolute(root, config["wmh"]["template"])  # type: ignore[index]
    resample_label_to_reference(participant.lesion_mask, template, lesion_ch2better)
    _force_binary(lesion_ch2better)
    source_nonzero = bool(np.any(source_data > 0))
    if source_nonzero and not np.any(np.asanyarray(nib.load(str(lesion_ch2better)).dataobj) > 0):
        raise RuntimeError("非空MNI152病灶位于ch2better可靠覆盖范围外")
    transforms_mni_to_t1 = list(registration_status["details"]["transforms_mni_to_t1"])
    apply_transforms(
        lesion_ch2better,
        participant.t1w,
        lesion_t1,
        transforms_mni_to_t1,
        _absolute(root, config["execution"]["ants_bin"]),  # type: ignore[index]
        participant_dir(config, participant) / "logs" / "lesion_transform.log",
        "NearestNeighbor",
    )
    _force_binary(lesion_t1)
    if source_nonzero and not np.any(np.asanyarray(nib.load(str(lesion_t1)).dataobj) > 0):
        raise RuntimeError("非空MNI152病灶经模板逆变换后在T1中为空")
    transforms_t1_to_flair = list(registration_status["details"]["transforms_t1_to_flair"])
    apply_transforms(
        lesion_t1,
        participant.flair,
        lesion_flair,
        transforms_t1_to_flair,
        _absolute(root, config["execution"]["ants_bin"]),  # type: ignore[index]
        participant_dir(config, participant) / "logs" / "lesion_transform.log",
        "NearestNeighbor",
    )
    _force_binary(lesion_flair)
    target_nonzero = bool(np.any(np.asanyarray(nib.load(str(lesion_flair)).dataobj) > 0))
    if source_nonzero and not target_nonzero:
        raise RuntimeError("非空MNI152病灶经MNI→T1→FLAIR变换后为空")
    transform_files = []
    for value in transforms_mni_to_t1 + transforms_t1_to_flair:
        cleaned = str(value).strip("[]")
        if cleaned.endswith(",1"):
            cleaned = cleaned[:-2]
        path = Path(cleaned)
        if path.is_file() and path not in transform_files:
            transform_files.append(path)
    return {
        "lesion_source_mni152": str(participant.lesion_mask),
        "lesion_ch2better": str(lesion_ch2better),
        "lesion_t1": str(lesion_t1),
        "lesion_flair": str(lesion_flair),
        "interpolation": "NearestNeighbor",
        "source_space": "FSL_MNI152",
        "source_reference": str(matches[0]),
        "mni152_to_ch2better": "identity_world_coordinates",
        "transform_chain_mni_to_t1": transforms_mni_to_t1,
        "transform_chain_t1_to_flair": transforms_t1_to_flair,
        "transform_sha256": {str(path): sha256(path) for path in transform_files},
        "source_sha256": sha256(participant.lesion_mask),
        "fixed_dilation_applied": False,
        "lesion_used_for_t1_volume_subtraction": False,
    }


def stage_registration(config: Mapping[str, object], participant: Participant, profile: str) -> Dict[str, object]:
    """使用固定SynthStrip处理T1/FLAIR，随后始终估计T1→FLAIR刚体变换。"""

    root = Path(str(config["project_root"]))
    output = participant_dir(config, participant) / "registration"
    brain_mask = output / "T1w_desc-SynthStrip_brain_mask.nii.gz"
    skullstripped_t1 = output / "T1w_desc-SynthStrip_brain.nii.gz"
    flair_mask = output / "FLAIR_desc-SynthStrip_brain_mask.nii.gz"
    skullstripped_flair = output / "FLAIR_desc-SynthStrip_brain.nii.gz"
    synthstrip = config["synthstrip"]  # type: ignore[index]
    device = str(config["execution"]["device_gpu" if profile == "gpu" else "device_cpu"])  # type: ignore[index]
    runtime = _absolute(root, synthstrip["runtime"])  # type: ignore[index]
    model = _absolute(root, synthstrip["model"])  # type: ignore[index]
    synthstrip_provenance = output / "synthstrip_provenance.json"
    expected = {
        "method": "FreeSurfer_SynthStrip_v7.4.1_model_v1",
        "t1w_sha256": sha256(participant.t1w),
        "runtime_sha256": sha256(runtime),
        "model_sha256": sha256(model),
        "device": device,
        "border_mm": float(synthstrip["border_mm"]),  # type: ignore[index]
    }
    existing = _read_json(synthstrip_provenance) if synthstrip_provenance.is_file() else {}
    synthstrip_reused = bool(
        brain_mask.is_file()
        and skullstripped_t1.is_file()
        and all(existing.get(key) == value for key, value in expected.items())
    )
    if not synthstrip_reused:
        run_synthstrip(
            participant.t1w,
            skullstripped_t1,
            brain_mask,
            runtime,
            model,
            _absolute(root, config["execution"]["wmh_python"]),  # type: ignore[index]
            device,
            float(synthstrip["border_mm"]),  # type: ignore[index]
            str(synthstrip["model_sha256"]),  # type: ignore[index]
            participant_dir(config, participant) / "logs" / "synthstrip.log",
        )
        _write_json(synthstrip_provenance, expected)
    flair_synthstrip_provenance = output / "flair_synthstrip_provenance.json"
    flair_expected = {
        "method": "FreeSurfer_SynthStrip_v7.4.1_model_v1",
        "flair_sha256": sha256(participant.flair),
        "runtime_sha256": sha256(runtime),
        "model_sha256": sha256(model),
        "device": device,
        "border_mm": float(synthstrip["border_mm"]),  # type: ignore[index]
    }
    flair_existing = _read_json(flair_synthstrip_provenance) if flair_synthstrip_provenance.is_file() else {}
    flair_synthstrip_reused = bool(
        flair_mask.is_file()
        and skullstripped_flair.is_file()
        and all(flair_existing.get(key) == value for key, value in flair_expected.items())
    )
    if not flair_synthstrip_reused:
        run_synthstrip(
            participant.flair,
            skullstripped_flair,
            flair_mask,
            runtime,
            model,
            _absolute(root, config["execution"]["wmh_python"]),  # type: ignore[index]
            device,
            float(synthstrip["border_mm"]),  # type: ignore[index]
            str(synthstrip["model_sha256"]),  # type: ignore[index]
            participant_dir(config, participant) / "logs" / "flair_synthstrip.log",
        )
        _write_json(flair_synthstrip_provenance, flair_expected)
    details = register_and_warp_atlas(
        skullstripped_t1,
        brain_mask,
        skullstripped_flair,
        flair_mask,
        participant.flair,
        _absolute(root, config["wmh"]["template"]),  # type: ignore[index]
        _absolute(root, config["wmh"]["atlas"]),  # type: ignore[index]
        output,
        _absolute(root, config["execution"]["ants_bin"]),  # type: ignore[index]
        participant_dir(config, participant) / "logs" / "registration.log",
        float(config["wmh"].get("registration_brain_mask_dice_min", 0.70)),  # type: ignore[index]
        float(config["wmh"].get("registration_brain_mask_center_distance_max_mm", 15.0)),  # type: ignore[index]
    )
    details["registration_input_t1"] = str(skullstripped_t1)
    details["t1_brain_mask"] = str(brain_mask)
    details["flair_brain_mask"] = str(flair_mask)
    details["skullstrip_method"] = "FreeSurfer_SynthStrip_v7.4.1_model_v1"
    details["flair_skullstrip_method"] = "FreeSurfer_SynthStrip_v7.4.1_model_v1"
    details["synthstrip_reused"] = synthstrip_reused
    details["flair_synthstrip_reused"] = flair_synthstrip_reused
    details["synthstrip_provenance"] = str(synthstrip_provenance)
    details["flair_synthstrip_provenance"] = str(flair_synthstrip_provenance)
    details["synthstrip_source_commit"] = str(synthstrip["source_commit"])  # type: ignore[index]
    details["synthstrip_model_sha256"] = str(synthstrip["model_sha256"])  # type: ignore[index]
    details["synthstrip_border_mm"] = float(synthstrip["border_mm"])  # type: ignore[index]
    return details


def stage_wmh_segmentation(config: Mapping[str, object], participant: Participant, profile: str) -> Dict[str, object]:
    """先生成 WMH/脑解剖标签，供后续去颅骨配准与 WMH 定量共同使用。"""

    root = Path(str(config["project_root"]))
    output = participant_dir(config, participant) / "wmh"
    # 官方 --crop 输出覆盖脑区但不恢复输入数组尺寸；先保留模型网格，再按世界坐标回到原生 FLAIR。
    model_grid_seg = output / "FLAIR_desc-WMHSynthSeg_modelGrid_dseg.nii.gz"
    model_grid_probability = Path(str(model_grid_seg).replace(".nii.gz", ".lesion_probs.nii.gz"))
    seg = output / "FLAIR_desc-WMHSynthSeg_dseg.nii.gz"
    probability = output / "FLAIR_desc-WMHSynthSeg_probability.nii.gz"
    volumes_csv = output / "WMHSynthSeg_volumes.csv"
    device = str(config["execution"]["device_gpu" if profile == "gpu" else "device_cpu"])  # type: ignore[index]
    model = _absolute(root, config["wmh"]["model"])  # type: ignore[index]
    runtime = root / "resources" / "runtime" / "wmh_synthseg_inference.py"
    provenance_path = output / "wmh_synthseg_provenance.json"
    expected_provenance = {
        "schema_version": "1.0",
        "flair_sha256": sha256(participant.flair),
        "model_sha256": sha256(model),
        "runtime_sha256": sha256(runtime),
        "device": device,
        "crop_enabled": device == "cuda",
    }
    existing_provenance = _read_json(provenance_path) if provenance_path.is_file() else {}
    reusable = (
        all(path.is_file() for path in (model_grid_seg, model_grid_probability, volumes_csv))
        and all(existing_provenance.get(key) == value for key, value in expected_provenance.items())
    )
    if not reusable:
        run_wmh_synthseg(
            participant.flair,
            model_grid_seg,
            volumes_csv,
            model,
            root / "resources" / "third_party" / "WMH-SynthSeg",
            device,
            participant_dir(config, participant) / "logs" / "wmh_synthseg.log",
        )
        _write_json(provenance_path, expected_provenance)
    resample_label_to_reference(model_grid_seg, participant.flair, seg)
    resample_continuous_to_reference(model_grid_probability, participant.flair, probability)
    return {
        "segmentation": str(seg),
        "probability_map": str(probability),
        "model_grid_segmentation": str(model_grid_seg),
        "model_grid_probability_map": str(model_grid_probability),
        "volumes_csv": str(volumes_csv),
        "device": device,
        "inference_reused": reusable,
        "native_flair_grid_restored": True,
        "segmentation_interpolation": "NearestNeighbor",
        "probability_interpolation": "Linear",
        "provenance": str(provenance_path),
    }


def stage_wmh(config: Mapping[str, object], participant: Participant, profile: str) -> Dict[str, object]:
    """在ch2better中自动对侧替代，再在原生FLAIR计数并计算20区。"""

    root = Path(str(config["project_root"]))
    output = participant_dir(config, participant) / "wmh"
    wmh_seg_status = _read_json(status_path(config, participant, "wmh_seg"))
    if wmh_seg_status["status"] != "pass":
        raise RuntimeError("依赖节点 wmh_seg 失败")
    seg = Path(wmh_seg_status["details"]["segmentation"])
    lesion_status = _read_json(status_path(config, participant, "lesion"))
    registration_status = _read_json(status_path(config, participant, "registration"))
    if lesion_status["status"] != "pass":
        raise RuntimeError("依赖节点 lesion 失败")
    if registration_status["status"] != "pass":
        raise RuntimeError("依赖节点 registration 失败")
    replacement = run_contralateral_replacement(
        seg,
        Path(lesion_status["details"]["lesion_flair"]),
        Path(registration_status["details"]["flair_brain_mask"]),
        participant.flair,
        _absolute(root, config["wmh"]["template"]),  # type: ignore[index]
        int(config["wmh"]["lesion_label"]),  # type: ignore[index]
        list(registration_status["details"]["transforms_flair_to_mni"]),
        list(registration_status["details"]["transforms_mni_to_flair"]),
        _absolute(root, config["execution"]["ants_bin"]),  # type: ignore[index]
        output / "contralateral",
        participant_dir(config, participant) / "logs" / "contralateral_replacement.log",
    )
    corrected = Path(str(replacement["final_wmh"]))
    atlas_native = Path(registration_status["details"]["atlas_native_flair"])
    raw = extract_wmh20_ml(corrected, atlas_native)
    z = chung_zscore(raw, participant.sex, _absolute(root, config["wmh"]["residual_info"]))  # type: ignore[index]
    feature_path = output / "wmh_features.json"
    _write_json(
        feature_path,
        {
            "participant_id": participant.participant_id,
            "sex": participant.sex,
            "age": participant.age,
            "raw_ml": raw,
            "z_chung": z,
            "age_adjustment_applied": False,
            "contralateral_correction": replacement,
        },
    )
    details: Dict[str, object] = {
        "segmentation": str(seg),
        "probability_map": str(wmh_seg_status["details"]["probability_map"]),
        "corrected_wmh": str(corrected),
        "feature_json": str(feature_path),
        "age_adjustment_applied": False,
    }
    details.update(replacement)
    return details


def _extract_icv_values(path: str) -> Dict[str, object]:
    if not path or not Path(path).is_file():
        return {"available": False, "values": {}}
    table = pd.read_csv(path)
    numeric = table.select_dtypes(include=[np.number])
    values = numeric.iloc[0].to_dict() if len(numeric) else {}
    output: Dict[str, object] = {"available": "702" in values, "source": path}
    if "702" in values:
        output["icv_label702_mm3"] = float(values["702"])
        output["icv_label702_ml"] = float(values["702"]) / 1000.0
    return output


def stage_t1(config: Mapping[str, object], participant: Participant, profile: str) -> Dict[str, object]:
    root = Path(str(config["project_root"]))
    output = participant_dir(config, participant) / "t1"
    tool_output = output / "nichart_tool_output"
    device = str(config["execution"]["device_gpu" if profile == "gpu" else "device_cpu"])  # type: ignore[index]
    tool = run_nichart_dlmuse(
        participant.t1w,
        tool_output,
        device,
        participant_dir(config, participant) / "logs" / "nichart_dlmuse.log",
    )
    segmentation = Path(tool["segmentation"])
    native, gm119, macro, denominator = extract_t1_features(
        segmentation,
        _absolute(root, config["t1"]["genmind_csv"]),  # type: ignore[index]
        _absolute(root, config["t1"]["derived_mapping"]),  # type: ignore[index]
        _absolute(root, config["t1"]["macro_mapping"]),  # type: ignore[index]
    )
    provider = GenMINDGlobalV1Provider(
        _absolute(root, config["t1"]["genmind_csv"]),  # type: ignore[index]
        _absolute(root, config["t1"]["derived_mapping"]),  # type: ignore[index]
        _absolute(root, config["t1"]["macro_mapping"]),  # type: ignore[index]
        int(config["t1"]["min_reference_n"]),  # type: ignore[index]
        float(config["t1"]["narrow_age_half_window"]),  # type: ignore[index]
        float(config["t1"]["expanded_age_half_window"]),  # type: ignore[index]
    )
    norm = provider.transform(macro, denominator, participant.age, participant.sex)
    if not norm.eligible:
        raise RuntimeError("T1 常模转换被阻断: {}".format(norm.failure_reason))
    icv = _extract_icv_values(tool["icv_csv"])
    feature_path = output / "t1_features.json"
    _write_json(
        feature_path,
        {
            "participant_id": participant.participant_id,
            "native145_ml": {str(key): value for key, value in native.items()},
            "gm119_ml": {str(key): value for key, value in gm119.items()},
            "macro20_ml": macro,
            "macro20_atrophy_z": norm.zscores,
            "nonventricular_muse_tissue_ml": denominator,
            "dlicv_icv": icv,
            "normative_profile": norm.profile,
            "macro_mapping_version": "muse_macro20_v1_provisional",
            "macro_mapping_scope": "NiChart official 15 cortical + project-defined 5 noncortical",
            "reference_n": norm.reference_n,
            "age_half_window": norm.age_half_window,
            "requested_device": tool["requested_device"],
            "effective_device": tool["effective_device"],
            "device_fallback_reason": tool["device_fallback_reason"],
            "inference_reused": tool["inference_reused"],
            "initial_execution_resumed_from_dlicv_checkpoint": tool["initial_execution_resumed_from_dlicv_checkpoint"],
            "provenance": tool["provenance"],
            "is_true_icv_residual_model": False,
            "lesion_subtracted_or_mirrored": False,
        },
    )
    macro_segmentation = write_macro20_segmentation(
        segmentation,
        _absolute(root, config["t1"]["macro_mapping"]),  # type: ignore[index]
        output / "T1w_desc-MUSEMacro20_dseg.nii.gz",
    )
    return {
        "segmentation": str(segmentation),
        "macro20_segmentation": str(macro_segmentation),
        "feature_json": str(feature_path),
        "dlicv_icv": icv,
        "normative_profile": norm.profile,
        "macro_mapping_version": "muse_macro20_v1_provisional",
        "reference_n": norm.reference_n,
        "age_half_window": norm.age_half_window,
        "requested_device": tool["requested_device"],
        "effective_device": tool["effective_device"],
        "device_fallback_reason": tool["device_fallback_reason"],
        "inference_reused": tool["inference_reused"],
        "initial_execution_resumed_from_dlicv_checkpoint": tool["initial_execution_resumed_from_dlicv_checkpoint"],
        "provenance": tool["provenance"],
    }


def stage_qc(config: Mapping[str, object], participant: Participant) -> Dict[str, object]:
    """从已通过节点重建固定四张中央QC图；清理本例旧图以免版本混杂。"""

    states = {stage: _read_json(status_path(config, participant, stage)) for stage in ("lesion", "registration", "wmh", "t1")}
    failed = [stage for stage, state in states.items() if state.get("status") != "pass"]
    if failed:
        raise RuntimeError("QC依赖节点未通过: {}".format("|".join(failed)))

    lesion = states["lesion"]["details"]
    wmh = states["wmh"]["details"]
    t1 = states["t1"]["details"]
    central_qc = qc_dir(config)
    central_qc.mkdir(parents=True, exist_ok=True)
    central_qc_resolved = central_qc.resolve()
    for candidate in sorted(central_qc.glob("{}_*.png".format(participant.participant_id))):
        if candidate.parent.resolve() != central_qc_resolved or candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("拒绝清理非中央QC普通文件: {}".format(candidate))
        candidate.unlink()
    figures = [
        save_overlay(participant.t1w, Path(lesion["lesion_t1"]), qc_path(config, participant, 1, "lesion_on_T1"), "{} lesion→T1".format(participant.participant_id)),
        save_overlay(participant.flair, Path(lesion["lesion_flair"]), qc_path(config, participant, 2, "lesion_on_FLAIR"), "{} lesion→FLAIR via T1 rigid transform".format(participant.participant_id)),
        save_dual_overlay(participant.flair, Path(wmh["original_wmh"]), Path(lesion["lesion_flair"]), qc_path(config, participant, 3, "WMH_lesion_overlap"), "{} original WMH and acute lesion".format(participant.participant_id)),
        save_overlay(participant.t1w, Path(t1["macro20_segmentation"]), qc_path(config, participant, 4, "T1_macro20"), "{} MUSE macro20".format(participant.participant_id)),
    ]
    return {
        "qc_dir": str(qc_dir(config)),
        "figures": [str(path) for path in figures],
        "figure_count": len(figures),
        "display_convention": "radiological_RAS_canonical_rotated_ccw_90",
        "panel_order": ["Coronal", "Sagittal", "Axial"],
        "physical_aspect_from_voxel_spacing": True,
        "subpanel_rotation_degrees": 90,
    }


def _status(config: Mapping[str, object], participant: Participant, stage: str) -> Dict[str, object]:
    path = status_path(config, participant, stage)
    return _read_json(path) if path.is_file() else {"status": "missing", "details": {"error": "status file missing"}}


def _records_to_table(records: List[Dict[str, object]], columns: Sequence[str], output: Path) -> None:
    table = pd.DataFrame(records)
    fixed = ["participant_id"] + list(columns)
    if table.empty:
        table = pd.DataFrame(columns=fixed)
    else:
        table = table.reindex(columns=fixed)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False)


def aggregate_outputs(config: Mapping[str, object], participants: Sequence[Participant]) -> Dict[str, object]:
    """分别保留单模态输出，并生成尚未经过人工QC的计算完成40维表。"""

    root = Path(str(config["project_root"]))
    tables = _absolute(root, config["derivatives"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(_absolute(root, config["t1"]["macro_mapping"]), sep="\t")  # type: ignore[index]
    macro_ids = mapping.sort_values("macro_index")["macro_id"].drop_duplicates().astype(str).tolist()
    display_names = display_name_map(_absolute(root, config["t1"]["derived_mapping"]))  # type: ignore[index]

    wmh_raw_rows: List[Dict[str, object]] = []
    wmh_z_rows: List[Dict[str, object]] = []
    native_rows: List[Dict[str, object]] = []
    gm_rows: List[Dict[str, object]] = []
    macro_rows: List[Dict[str, object]] = []
    t1_z_rows: List[Dict[str, object]] = []
    computed_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []
    native_columns: List[str] = []
    gm_columns: List[str] = []

    for participant in participants:
        stage_states = {
            stage: _status(config, participant, stage)
            for stage in ("registration", "lesion", "wmh_seg", "wmh", "t1", "qc", "cleanup", "lowres")
        }
        wmh_ok = stage_states["wmh"]["status"] == "pass"
        t1_ok = stage_states["t1"]["status"] == "pass"
        wmh_data: Dict[str, object] = {}
        t1_data: Dict[str, object] = {}
        if wmh_ok:
            wmh_data = _read_json(Path(stage_states["wmh"]["details"]["feature_json"]))
            prefix = {"participant_id": participant.participant_id}
            wmh_raw_rows.append(dict(prefix, **wmh_data["raw_ml"]))
            wmh_z_rows.append(dict(prefix, **wmh_data["z_chung"]))
        if t1_ok:
            t1_data = _read_json(Path(stage_states["t1"]["details"]["feature_json"]))
            prefix = {"participant_id": participant.participant_id}
            native_named = {"muse_{:03d}_ml".format(int(key)): value for key, value in t1_data["native145_ml"].items()}
            gm_named = {"muse_{:03d}_ml".format(int(key)): value for key, value in t1_data["gm119_ml"].items()}
            native_columns = list(native_named)
            gm_columns = list(gm_named)
            native_rows.append(dict(prefix, **native_named))
            gm_rows.append(dict(prefix, **gm_named))
            macro_rows.append(dict(prefix, **{"t1_{}_ml".format(key): value for key, value in t1_data["macro20_ml"].items()}))
            t1_z_rows.append(dict(prefix, **t1_data["macro20_atrophy_z"]))
        multimodal_eligible = bool(wmh_ok and t1_ok)
        if multimodal_eligible:
            combined = {"participant_id": participant.participant_id}
            combined.update(wmh_data["z_chung"])
            combined.update(t1_data["macro20_atrophy_z"])
            if len(combined) != 41 or any(not np.isfinite(float(value)) for key, value in combined.items() if key != "participant_id"):
                raise ValueError("{} primary40 不完整，禁止导出".format(participant.participant_id))
            computed_rows.append(combined)
        failed_nodes = [stage for stage, state in stage_states.items() if state["status"] not in {"pass", "missing"}]
        qc_rows.append(
            {
                "participant_id": participant.participant_id,
                "wmh_eligible": wmh_ok,
                "t1_eligible": t1_ok,
                "multimodal_ineligible": not multimodal_eligible,
                "failed_nodes": "|".join(failed_nodes),
                "wmh_age_adjustment_applied": False,
                "wmh_lesion_overlap_ml": wmh_data.get("contralateral_correction", {}).get("wmh_lesion_overlap_ml", "") if wmh_ok else "",
                "wmh_contralateral_donor_ml": wmh_data.get("contralateral_correction", {}).get("wmh_contralateral_donor_ml", "") if wmh_ok else "",
                "wmh_replacement_added_ml": wmh_data.get("contralateral_correction", {}).get("wmh_replacement_added_ml", "") if wmh_ok else "",
                "wmh_original_overlap_removed_ml": wmh_data.get("contralateral_correction", {}).get("wmh_original_overlap_removed_ml", "") if wmh_ok else "",
                "wmh_bilateral_lesion_conflict_removed_ml": wmh_data.get("contralateral_correction", {}).get("wmh_bilateral_lesion_conflict_removed_ml", "") if wmh_ok else "",
                "wmh_out_of_brain_donor_removed_ml": wmh_data.get("contralateral_correction", {}).get("wmh_out_of_brain_donor_removed_ml", "") if wmh_ok else "",
                "symmetry_space": "ch2better" if wmh_ok else "",
                "symmetry_plane_world_x_mm": 0 if wmh_ok else "",
                "manual_adjudication_applied": False if wmh_ok else "",
                "fixed_dilation_applied": False if wmh_ok else "",
                "t1_normative_profile": "genmind_global_v1_provisional" if t1_ok else "",
                "t1_macro_mapping_version": "muse_macro20_v1_provisional" if t1_ok else "",
                "technical_demo_only": True,
            }
        )

    wmh_z_columns = [name.replace("_ml", "_z_chung") for name in WMH_FEATURES]
    t1_z_columns = ["t1_{}_atrophy_z".format(name) for name in macro_ids]
    _records_to_table(wmh_raw_rows, WMH_FEATURES, tables / "wmh20_raw.tsv")
    _records_to_table(wmh_z_rows, wmh_z_columns, tables / "wmh20_z_chung.tsv")
    _records_to_table(native_rows, native_columns, tables / "t1_muse145_raw.tsv")
    _records_to_table(gm_rows, gm_columns, tables / "t1_gm119_raw.tsv")
    _records_to_table(macro_rows, ["t1_{}_ml".format(name) for name in macro_ids], tables / "t1_macro20_raw.tsv")
    _records_to_table(t1_z_rows, t1_z_columns, tables / "t1_macro20_z_genmind.tsv")
    _records_to_table(computed_rows, wmh_z_columns + t1_z_columns, tables / "features_computed40.tsv")
    # 正式主矩阵必须由人工QC后的export命令重新生成，禁止遗留旧文件造成误用。
    primary_path = tables / "features_primary40.tsv"
    if primary_path.is_file():
        primary_path.unlink()
    pd.DataFrame(qc_rows).to_csv(tables / "subject_qc.tsv", sep="\t", index=False)

    dictionary = []
    for index, name in enumerate(wmh_z_columns, start=1):
        dictionary.append({"feature_index": index, "feature_name": name, "modality": "WMH", "direction": "higher_is_more_disease", "unit": "z", "normative_profile": "chung_ukbb_residual", "age_adjustment_applied": False})
    for offset, (name, macro_id) in enumerate(zip(t1_z_columns, macro_ids), start=21):
        is_official_cortical = offset <= 35
        mapping_note = ""
        if macro_id == "basal_forebrain_ventraldc_brainstem":
            mapping_note = "heterogeneous full-coverage group; brainstem and bilateral ventral diencephalon are outside official GM119"
        dictionary.append(
            {
                "feature_index": offset,
                "feature_name": name,
                "modality": "T1",
                "direction": "higher_is_more_atrophy",
                "unit": "z",
                "normative_profile": "genmind_global_v1_provisional",
                "age_adjustment_applied": True,
                "mapping_version": "muse_macro20_v1_provisional",
                "mapping_scope": "official_NiChart_bilateral_cortical" if is_official_cortical else "project_defined_noncortical",
                "mapping_note": mapping_note,
            }
        )
    pd.DataFrame(dictionary).to_csv(tables / "feature_dictionary.tsv", sep="\t", index=False)
    return {
        "participants": len(participants),
        "computed40_rows": len(computed_rows),
        "primary40_withheld_pending_manual_qc": True,
        "tables_dir": str(tables),
    }


def _safe_managed_path(path: Path, subject_root: Path) -> Path:
    """清理仅允许落在当前受试者衍生目录内，且不能穿过软链接。"""

    if path.is_symlink():
        raise RuntimeError("拒绝清理软链接: {}".format(path))
    resolved = path.resolve()
    try:
        resolved.relative_to(subject_root.resolve())
    except ValueError as exc:
        raise RuntimeError("拒绝清理受试者目录外路径: {}".format(path)) from exc
    return resolved


def stage_cleanup(config: Mapping[str, object], participant: Participant) -> Dict[str, object]:
    """四图QC成功后删除可重建大文件；处理失败病例不会进入本阶段。"""

    subject_root = participant_dir(config, participant)
    qc_status = _status(config, participant, "qc")
    if qc_status.get("status") != "pass" or qc_status.get("details", {}).get("figure_count") != 4:
        raise RuntimeError("只有四图QC成功病例才允许自动清理")
    states = {
        stage: _status(config, participant, stage)
        for stage in ("lesion", "registration", "wmh_seg", "wmh", "t1")
    }
    if any(state.get("status") != "pass" for state in states.values()):
        raise RuntimeError("存在失败处理节点，保留中间件供诊断")

    registration = states["registration"]["details"]
    lesion = states["lesion"]["details"]
    wmh_seg = states["wmh_seg"]["details"]
    wmh = states["wmh"]["details"]
    t1 = states["t1"]["details"]
    retained = {
        Path(str(wmh_seg["segmentation"])).resolve(),
        Path(str(wmh["original_wmh"])).resolve(),
        Path(str(wmh["corrected_wmh"])).resolve(),
        Path(str(wmh["feature_json"])).resolve(),
        Path(str(t1["segmentation"])).resolve(),
        Path(str(t1["macro20_segmentation"])).resolve(),
        Path(str(t1["feature_json"])).resolve(),
    }
    candidates: List[Path] = []
    for key in (
        "t1_flair",
        "t1_mask_flair",
        "t1_mni",
        "atlas_native_flair",
        "atlas_xforms_harmonized",
        "registration_input_t1",
        "t1_brain_mask",
        "flair_brain_mask",
    ):
        if registration.get(key):
            candidates.append(Path(str(registration[key])))
    candidates.extend(
        [
            subject_root / "registration" / "FLAIR_desc-SynthStrip_brain.nii.gz",
            subject_root / "registration" / "T1w_desc-registrationInput_brain_mask_space-ch2better.nii.gz",
            subject_root / "registration" / "t1_to_ch2better_1Warp.nii.gz",
            subject_root / "registration" / "t1_to_ch2better_1InverseWarp.nii.gz",
        ]
    )
    for key in ("lesion_ch2better", "lesion_t1", "lesion_flair"):
        if lesion.get(key):
            candidates.append(Path(str(lesion[key])))
    for key in (
        "probability_map",
        "model_grid_segmentation",
        "model_grid_probability_map",
        "volumes_csv",
    ):
        if wmh_seg.get(key):
            candidates.append(Path(str(wmh_seg[key])))
    for key in (
        "lesion_components",
        "triggered_lesion",
        "wmh_template",
        "lesion_template",
        "mirror_wmh_template",
        "mirror_lesion_template",
        "donor_template",
        "conflict_template",
        "out_of_brain_template",
        "replacement_template",
        "donor_native_flair",
        "conflict_native_flair",
        "out_of_brain_native_flair",
        "replacement_native_flair",
    ):
        if wmh.get(key):
            candidates.append(Path(str(wmh[key])))

    deleted: List[Dict[str, object]] = []
    total_bytes = 0
    for candidate in sorted(set(candidates), key=lambda value: str(value)):
        if not candidate.exists() or candidate.resolve() in retained:
            continue
        resolved = _safe_managed_path(candidate, subject_root)
        if not resolved.is_file():
            raise RuntimeError("清理目标不是普通文件: {}".format(resolved))
        size = resolved.stat().st_size
        digest = sha256(resolved)
        resolved.unlink()
        total_bytes += size
        deleted.append({"path": str(resolved), "size_bytes": size, "sha256_before_delete": digest})

    temp_working = subject_root / "t1" / "nichart_tool_output" / "temp_working_dir"
    deleted_directories: List[str] = []
    if temp_working.exists():
        resolved_temp = _safe_managed_path(temp_working, subject_root)
        if not resolved_temp.is_dir():
            raise RuntimeError("DLMUSE临时目标不是目录: {}".format(resolved_temp))
        shutil.rmtree(str(resolved_temp))
        deleted_directories.append(str(resolved_temp))

    manifest = subject_root / "cleanup" / "cleanup_manifest.json"
    details: Dict[str, object] = {
        "policy": "minimal_success_only_v1",
        "deleted_files": deleted,
        "deleted_directories": deleted_directories,
        "bytes_removed": total_bytes,
        "retained_outputs": sorted(str(path) for path in retained),
        "failed_cases_are_not_cleaned": True,
        "large_nonlinear_transforms_retained": False,
    }
    _write_json(manifest, details)
    details["cleanup_manifest"] = str(manifest)
    return details


def export_reviewed_outputs(
    config: Mapping[str, object], participants: Sequence[Participant]
) -> Dict[str, object]:
    """仅把当前四图人工QC通过病例写入正式40维主矩阵。"""

    root = Path(str(config["project_root"]))
    derivatives = _absolute(root, config["derivatives"])
    tables = derivatives / "tables"
    reviews = load_review_table(participants, qc_dir(config), derivatives)
    pending = reviews.loc[~reviews["review_state"].isin(["pass", "fail"]), "participant_id"].astype(str).tolist()
    if pending:
        raise RuntimeError("人工QC尚未完成，共{}例: {}".format(len(pending), pending[:20]))
    computed_path = tables / "features_computed40.tsv"
    if not computed_path.is_file():
        raise FileNotFoundError("缺少计算完成40维表: {}".format(computed_path))
    computed = pd.read_csv(computed_path, sep="\t", dtype={"participant_id": str})
    passed_ids = reviews.loc[reviews["review_state"] == "pass", "participant_id"].astype(str).tolist()
    missing = sorted(set(passed_ids) - set(computed["participant_id"].astype(str)))
    if missing:
        raise RuntimeError("以下QC通过病例没有完整40维，禁止导出: {}".format(missing))
    primary = computed.loc[computed["participant_id"].astype(str).isin(passed_ids)].copy()
    primary["__order"] = pd.Categorical(primary["participant_id"], categories=passed_ids, ordered=True)
    primary = primary.sort_values("__order").drop(columns="__order")
    primary_path = tables / "features_primary40.tsv"
    temporary = primary_path.with_suffix(primary_path.suffix + ".tmp")
    primary.to_csv(temporary, sep="\t", index=False)
    temporary.replace(primary_path)

    subject_qc_path = tables / "subject_qc.tsv"
    subject_qc = pd.read_csv(subject_qc_path, sep="\t", dtype={"participant_id": str})
    manual = reviews[["participant_id", "review_state", "qc_pass", "reasons_json", "note", "reviewer", "reviewed_at_utc", "image_hash", "status_hash"]].copy()
    manual["manual_qc_failure_reasons"] = manual["reasons_json"].map(
        lambda value: "|".join(json.loads(str(value)))
    )
    manual = manual.rename(
        columns={
            "review_state": "manual_qc_state",
            "qc_pass": "manual_qc_pass",
            "note": "manual_qc_note",
            "reviewer": "manual_qc_reviewer",
            "reviewed_at_utc": "manual_qc_reviewed_at_utc",
            "image_hash": "manual_qc_image_hash",
            "status_hash": "manual_qc_processing_status_hash",
        }
    ).drop(columns="reasons_json")
    old_manual = [column for column in subject_qc.columns if column.startswith("manual_qc_")]
    subject_qc = subject_qc.drop(columns=old_manual).merge(manual, on="participant_id", how="left", validate="one_to_one")
    qc_temporary = subject_qc_path.with_suffix(subject_qc_path.suffix + ".tmp")
    subject_qc.to_csv(qc_temporary, sep="\t", index=False)
    qc_temporary.replace(subject_qc_path)
    return {
        "reviewed_subjects": len(reviews),
        "qc_pass_subjects": len(passed_ids),
        "qc_fail_subjects": int((reviews["review_state"] == "fail").sum()),
        "features_primary40": str(primary_path),
    }


def stage_lowres(config: Mapping[str, object], participant: Participant, profile: str) -> Dict[str, object]:
    """运行三种临床层厚的完整隔离链，并输出 40 维稳定性报告。"""

    return run_lowres_validation(config, participant, profile, participant_dir(config, participant) / "lowres")
