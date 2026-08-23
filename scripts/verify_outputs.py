#!/usr/bin/env python3
"""核验第1步核心表格契约，并保存机器可读验收摘要。"""

import json
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd

from substain_features.images import _orthogonal_qc_views
from substain_features.wmh import WMH_FEATURES


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "derivatives" / "substain_features" / "tables"


def read_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError("缺少核心表格 {}".format(path))
    return pd.read_csv(path, sep="\t")


def main() -> None:
    mapping = pd.read_csv(ROOT / "resources" / "mappings" / "muse_macro20_v1_provisional.tsv", sep="\t")
    macro_ids = mapping.sort_values("macro_index")["macro_id"].drop_duplicates().astype(str).tolist()
    wmh_z = [name.replace("_ml", "_z_chung") for name in WMH_FEATURES]
    t1_z = ["t1_{}_atrophy_z".format(name) for name in macro_ids]
    contracts = {
        "wmh20_raw.tsv": 20,
        "wmh20_z_chung.tsv": 20,
        "t1_muse145_raw.tsv": 145,
        "t1_gm119_raw.tsv": 119,
        "t1_macro20_raw.tsv": 20,
        "t1_macro20_z_genmind.tsv": 20,
        "features_computed40.tsv": 40,
        "features_primary40.tsv": 40,
    }
    shapes = {}
    for name, feature_count in contracts.items():
        table = read_table(name)
        observed = len(table.columns) - 1
        if observed != feature_count:
            raise AssertionError("{} 特征数应为 {}，实际 {}".format(name, feature_count, observed))
        shapes[name] = {"rows": int(len(table)), "feature_columns": int(observed)}

    primary = read_table("features_primary40.tsv")
    expected_columns = ["participant_id"] + wmh_z + t1_z
    if list(primary.columns) != expected_columns:
        raise AssertionError("features_primary40.tsv 列顺序不符合固定契约")
    values = primary.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("features_primary40.tsv 含非有限值")

    dictionary = read_table("feature_dictionary.tsv")
    if dictionary["feature_index"].tolist() != list(range(1, 41)):
        raise AssertionError("feature_dictionary.tsv 索引不是连续1..40")
    if dictionary["feature_name"].tolist() != expected_columns[1:]:
        raise AssertionError("特征字典与主表顺序不一致")
    valid_directions = {"higher_is_more_disease", "higher_is_more_atrophy"}
    if set(dictionary["direction"]) != valid_directions:
        raise AssertionError("疾病方向字段不符合约定")

    qc = read_table("subject_qc.tsv")
    eligible = (~qc["multimodal_ineligible"].astype(bool)).sum()
    if "manual_qc_pass" not in qc.columns:
        raise AssertionError("subject_qc缺少人工QC门控字段")
    manual_pass = qc["manual_qc_pass"].astype(bool).sum()
    if int(manual_pass) != len(primary):
        raise AssertionError("人工QC通过数与正式主矩阵行数不一致")
    correction_columns = {
        "wmh_lesion_overlap_ml", "wmh_contralateral_donor_ml", "wmh_replacement_added_ml",
        "wmh_original_overlap_removed_ml", "wmh_bilateral_lesion_conflict_removed_ml",
        "wmh_out_of_brain_donor_removed_ml", "symmetry_space", "symmetry_plane_world_x_mm",
        "manual_adjudication_applied", "fixed_dilation_applied",
    }
    if not correction_columns.issubset(qc.columns):
        raise AssertionError("subject_qc缺少自动对侧替代追溯字段")

    # 同时验收本次无session、集中QC和编号化总控的工程契约。
    expected_participant_columns = [
        "participant_id", "age", "sex", "site_id", "t1w", "flair", "lesion_mask"
    ]
    participants = pd.read_csv(ROOT / "config" / "participants.tsv", sep="\t")
    if participants.columns.tolist() != expected_participant_columns:
        raise AssertionError("participants.tsv 必须是固定7列且不含session")

    derivatives = ROOT / "derivatives" / "substain_features"
    subject_roots = sorted(path for path in derivatives.glob("sub-*") if path.is_dir())
    session_directories = sorted(
        str(item.relative_to(ROOT))
        for subject_root in subject_roots
        for item in subject_root.rglob("ses-*")
        if item.is_dir()
    )
    if session_directories:
        raise AssertionError("正式输出仍含session目录: {}".format(session_directories))

    central_qc = sorted((derivatives / "qc").glob("*.png"))
    scattered_qc = sorted(item for subject_root in subject_roots for item in subject_root.rglob("*.png"))
    qc_subjects = len({path.name.split("_", 1)[0] for path in central_qc})
    if len(central_qc) != 4 * qc_subjects:
        raise AssertionError("集中QC必须为每例固定4张: {}".format(len(central_qc)))
    if scattered_qc:
        raise AssertionError("受试者子目录仍有散落QC图")

    source = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    views = _orthogonal_qc_views(source, (1, 1, 2), (1.0, 2.0, 5.0))
    if [view[0] for view in views] != ["Coronal", "Sagittal", "Axial"]:
        raise AssertionError("QC三正交面顺序错误")
    if not np.array_equal(views[0][1], np.fliplr(source[:, 1, :].T)):
        raise AssertionError("QC冠状位不是标准临床放射学方向")
    if [(view[2], view[3]) for view in views] != [(5.0, 1.0), (5.0, 2.0), (2.0, 1.0)]:
        raise AssertionError("QC没有使用真实体素尺寸")

    config_text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    if "max_parallel_jobs: 200" not in config_text or "gpu_policy: auto_one_job_per_device" not in config_text:
        raise AssertionError("并发200或每GPU一任务策略未写入配置")

    controllers = [ROOT / "run_pipeline.sh"] + sorted((ROOT / "scripts" / "steps").glob("*.sh"))
    if not all(item.is_file() and os.access(item, os.X_OK) for item in controllers):
        raise AssertionError("总控或编号脚本缺失执行权限")

    production_files = list((ROOT / "src").rglob("*.py")) + [ROOT / "workflow" / "Snakefile"]
    production_files += list((ROOT / "scripts" / "steps").glob("*.sh"))
    forbidden = []
    for source_path in production_files:
        source_text = source_path.read_text(encoding="utf-8")
        if "session_id" in source_text or "ses-01" in source_text or "session_label" in source_text:
            forbidden.append(str(source_path.relative_to(ROOT)))
    if forbidden:
        raise AssertionError("正式管道仍含session建模: {}".format(forbidden))
    active_text = "\n".join(path.read_text(encoding="utf-8") for path in production_files + [ROOT / "config" / "config.yaml"])
    obsolete_tokens = (
        "lesion_" + "exclusion_mm",
        "dilate_mask_" + "physical",
        "desc-dilated" + "2mm",
    )
    obsolete_lesion_paths = [
        token for token in obsolete_tokens
        if token in active_text
    ]
    if obsolete_lesion_paths:
        raise AssertionError("正式管道仍含固定病灶膨胀路径: {}".format(obsolete_lesion_paths))

    summary = {
        "schema_version": "1.0",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "qc_pass_subjects": primary["participant_id"].astype(str).tolist(),
        "primary40_rows": int(len(primary)),
        "primary40_feature_columns": 40,
        "all_primary_values_finite": True,
        "fixed_column_order_verified": True,
        "feature_dictionary_verified": True,
        "multimodal_eligible_subjects": int(eligible),
        "manual_qc_pass_subjects": int(manual_pass),
        "table_shapes": shapes,
        "participants_columns": expected_participant_columns,
        "session_directories": session_directories,
        "production_session_references": forbidden,
        "central_qc_pngs": len(central_qc),
        "scattered_qc_pngs": len(scattered_qc),
        "qc_subjects": qc_subjects,
        "qc_display_convention": "radiological_RAS_canonical_standard_axes",
        "qc_panel_order": ["Coronal", "Sagittal", "Axial"],
        "qc_physical_aspect_from_voxel_spacing_verified": True,
        "lowres_qc_pngs": len([path for path in central_qc if any(tag in path.name for tag in ("_3mm_", "_5mm_", "_6mm_"))]),
        "numbered_controllers_executable": True,
        "max_parallel_jobs": 200,
        "gpu_policy": "auto_one_job_per_device",
        "obsolete_fixed_lesion_dilation_paths": obsolete_lesion_paths,
    }
    output = TABLES / "verification_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
