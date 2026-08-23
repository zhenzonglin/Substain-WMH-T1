"""输入、网格、元数据、资源和运行工具的预检。"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd

from .images import grid_info, same_grid, world_bounds
from .resources import sha256
from .schema import Participant


REQUIRED_RESOURCES = [
    "resources/third_party/WMH_progression_modeling",
    "resources/third_party/WMH-SynthSeg",
    "resources/third_party/DLMUSE",
    "resources/third_party/NiChart_DLMUSE",
    "resources/third_party/SynthStrip",
    "resources/normative/Residual_Info.mat",
    "resources/normative/genmind_dataset.csv",
    "resources/normative/genmind_upstream/model/col_dict.npz",
    "resources/normative/genmind_upstream/model/kde_asian_female.npz",
    "resources/normative/genmind_upstream/model/kde_asian_male.npz",
    "resources/normative/genmind_upstream/model/kde_black_female.npz",
    "resources/normative/genmind_upstream/model/kde_black_male.npz",
    "resources/normative/genmind_upstream/model/kde_white_female.npz",
    "resources/normative/genmind_upstream/model/kde_white_male.npz",
    "resources/models/WMH-SynthSeg_v10_231110.pth",
    "resources/models/synthstrip.1.pt",
    "resources/models/SynthStrip_VERSION.txt",
    "resources/templates/MNI_ch2better_WM_20ROIs.nii.gz",
    "resources/templates/ch2better.nii.gz",
    "resources/templates/fsl/MNI152_T1_1mm.nii.gz",
    "resources/templates/fsl/MNI152_T1_2mm.nii.gz",
    "resources/tools/ants-2.5.4/bin/antsApplyTransforms",
    "resources/tools/synthstrip/mri_synthstrip",
    "resources/mappings/muse_macro20_v1_provisional.tsv",
    "resources/third_party/SynthStrip/LICENSE.txt",
    "resources/licenses/SynthStrip_MODEL_MIT.txt",
]

FSL_MNI152_REFERENCES = {
    "1mm": "resources/templates/fsl/MNI152_T1_1mm.nii.gz",
    "2mm": "resources/templates/fsl/MNI152_T1_2mm.nii.gz",
}
FSL_MNI152_SHA256 = {
    "1mm": "83a14faa9b124d4058c181188f91dbb42ad99f0f0a3be881d9bf7949ce0829a3",
    "2mm": "d923a6950f6e61919f3b1819669d9291ae2e7ac9fc79d244e85849028d32aeac",
}


def _mni152_grid(path: Path, project_root: Path) -> Dict[str, object]:
    """严格匹配FSL MNI152 1/2 mm参考网格，并检查qform/sform一致性。"""

    image = nib.load(str(path))
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    errors: List[str] = []
    if int(qcode) == 0 and int(scode) == 0:
        errors.append("lesion_mask 的qform/sform均未定义")
    if int(qcode) > 0 and int(scode) > 0 and not np.allclose(qform, sform, atol=1e-4):
        errors.append("lesion_mask 的qform与sform不一致")
    matches = []
    for resolution, relative in FSL_MNI152_REFERENCES.items():
        reference = project_root / relative
        if reference.is_file() and same_grid(path, reference):
            matches.append(resolution)
    if len(matches) != 1:
        errors.append("lesion_mask不匹配FSL MNI152标准1mm或2mm网格")
    return {
        "space": "FSL_MNI152",
        "resolution": matches[0] if len(matches) == 1 else "",
        "qform_code": int(qcode),
        "sform_code": int(scode),
        "errors": errors,
    }


def _audit_participant(participant: Participant, project_root: Path) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    for name, path in (("t1w", participant.t1w), ("flair", participant.flair), ("lesion_mask", participant.lesion_mask)):
        if not path.is_file():
            errors.append("{} 不存在: {}".format(name, path))

    images: Dict[str, object] = {}
    if not errors:
        for name, path in (("t1w", participant.t1w), ("flair", participant.flair), ("lesion_mask", participant.lesion_mask)):
            try:
                images[name] = grid_info(path)
                images[name]["sha256"] = sha256(path)
            except Exception as exc:
                errors.append("{} 无法读取: {}".format(name, exc))
        if "lesion_mask" in images:
            lesion = nib.load(str(participant.lesion_mask)).get_fdata()
            if not np.isfinite(lesion).all():
                errors.append("lesion_mask 含非有限值")
            unique = np.unique(lesion[np.isfinite(lesion)])
            if not set(float(value) for value in unique).issubset({0.0, 1.0}):
                errors.append("lesion_mask必须是严格0/1二值掩膜，实际值含{}".format(unique[:20].tolist()))
            if np.count_nonzero(lesion) == 0:
                warnings.append("lesion_mask 为空")
        lesion_grid = _mni152_grid(participant.lesion_mask, project_root)
        errors.extend(str(value) for value in lesion_grid["errors"])
        if not same_grid(participant.t1w, participant.flair):
            warnings.append("T1与FLAIR网格不一致；仍按固定规则执行刚体配准")
        else:
            warnings.append("T1与FLAIR网格相同；仍按固定规则执行六自由度刚体配准")
    else:
        lesion_grid = {"space": "FSL_MNI152", "resolution": "", "qform_code": 0, "sform_code": 0, "errors": []}

    return {
        "participant_id": participant.participant_id,
        "age": participant.age,
        "sex": participant.sex,
        "site_id": participant.site_id,
        "images": images,
        "lesion_space_audit": lesion_grid,
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
    }


def run_audit(project_root: Path, participants: Iterable[Participant], output_dir: Path) -> Dict[str, object]:
    """执行只读预检并写出 JSON/TSV，可由 CLI 和 Snakemake 共用。"""

    subjects = [_audit_participant(item, project_root) for item in participants]
    resources = []
    for relative in REQUIRED_RESOURCES:
        path = project_root / relative
        resources.append({"resource": relative, "exists": path.exists(), "path": str(path)})
    # CLI可能由绝对路径启动，不能只看当前PATH；同时检查项目内固定环境和ANTs副本。
    def resolve_tool(command: str, fallback: Path) -> Optional[str]:
        discovered = shutil.which(command)
        if discovered:
            return discovered
        return str(fallback) if fallback.is_file() and fallback.stat().st_mode & 0o111 else None

    tools = {
        "python": str(Path(sys.executable).resolve()),
        "snakemake": resolve_tool("snakemake", project_root / "envs/core-venv/bin/snakemake"),
        "antsRegistrationSyNQuick.sh": resolve_tool(
            "antsRegistrationSyNQuick.sh",
            project_root / "resources/tools/ants-2.5.4/bin/antsRegistrationSyNQuick.sh",
        ),
        "antsApplyTransforms": resolve_tool(
            "antsApplyTransforms",
            project_root / "resources/tools/ants-2.5.4/bin/antsApplyTransforms",
        ),
        "NiChart_DLMUSE": resolve_tool("NiChart_DLMUSE", project_root / "envs/t1/bin/NiChart_DLMUSE"),
        "mri_synthstrip": resolve_tool(
            "mri_synthstrip", project_root / "resources/tools/synthstrip/mri_synthstrip"
        ),
        "wmh_python": resolve_tool("substain-wmh-python", project_root / "envs/wmh/bin/python"),
    }
    hard_missing = [row["resource"] for row in resources if not row["exists"]]
    atlas = project_root / "resources/templates/MNI_ch2better_WM_20ROIs.nii.gz"
    template = project_root / "resources/templates/ch2better.nii.gz"
    template_atlas_world_aligned = bool(
        atlas.is_file()
        and template.is_file()
        and np.allclose(world_bounds(atlas), world_bounds(template), atol=1.1)
    )
    if atlas.is_file() and template.is_file() and not template_atlas_world_aligned:
        hard_missing.append("atlas_template_world_bounds_mismatch")
    synthstrip_model = project_root / "resources/models/synthstrip.1.pt"
    synthstrip_model_sha256 = sha256(synthstrip_model) if synthstrip_model.is_file() else ""
    if synthstrip_model_sha256 and synthstrip_model_sha256 != "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33":
        hard_missing.append("synthstrip_model_sha256_mismatch")
    fsl_mni152_sha256 = {}
    for resolution, relative in FSL_MNI152_REFERENCES.items():
        path = project_root / relative
        observed = sha256(path) if path.is_file() else ""
        fsl_mni152_sha256[resolution] = observed
        if observed and observed != FSL_MNI152_SHA256[resolution]:
            hard_missing.append("fsl_mni152_{}_sha256_mismatch".format(resolution))
    hard_missing.extend("tool:{}".format(name) for name, path in tools.items() if not path)
    report = {
        "schema_version": "1.0",
        "raw_inputs_modified": False,
        "participants": subjects,
        "resources": resources,
        "tools": tools,
        "status": "pass" if all(row["status"] == "pass" for row in subjects) and not hard_missing else "fail",
        "missing_resources": hard_missing,
        "template_atlas_world_aligned": template_atlas_world_aligned,
        "synthstrip_model_sha256": synthstrip_model_sha256,
        "fsl_mni152_sha256": fsl_mni152_sha256,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    rows = []
    for subject in subjects:
        rows.append(
            {
                "participant_id": subject["participant_id"],
                "status": subject["status"],
                "errors": " | ".join(subject["errors"]),
                "warnings": " | ".join(subject["warnings"]),
                "lesion_space": subject["lesion_space_audit"]["space"],
                "lesion_mni_resolution": subject["lesion_space_audit"]["resolution"],
                "lesion_qform_code": subject["lesion_space_audit"]["qform_code"],
                "lesion_sform_code": subject["lesion_space_audit"]["sform_code"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "audit_subjects.tsv", sep="\t", index=False)
    return report
