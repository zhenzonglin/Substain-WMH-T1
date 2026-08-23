"""临床层厚向下兼容性：隔离重跑完整 WMH/T1 特征链并比较 40 维结果。"""

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
import yaml

from .images import downsample_image
from .resources import sha256
from .schema import PARTICIPANT_COLUMNS, Participant


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_modality_stability(delta_report: Path, output: Path) -> Path:
    """把40维总误差拆成WMH/T1，避免总体相关掩盖单一模态失稳。"""

    deltas = pd.read_csv(delta_report, sep="\t")
    deltas["modality"] = np.where(deltas["feature_name"].astype(str).str.startswith("wmh_"), "WMH", "T1")
    rows: List[Dict[str, object]] = []
    for (participant_id, spacing, modality), group in deltas.groupby(["participant_id", "spacing_mm", "modality"]):
        difference = group["delta_z"].to_numpy(dtype=float)
        native = group["native_z"].to_numpy(dtype=float)
        lowres = group["lowres_z"].to_numpy(dtype=float)
        rows.append(
            {
                "participant_id": participant_id,
                "spacing_mm": spacing,
                "modality": modality,
                "feature_count": int(len(group)),
                "pearson_r": float(np.corrcoef(native, lowres)[0, 1]),
                "mae_z": float(np.mean(np.abs(difference))),
                "rmse_z": float(np.sqrt(np.mean(difference ** 2))),
                "max_abs_delta_z": float(np.max(np.abs(difference))),
            }
        )
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    return output


def run_lowres_validation(config: Mapping[str, object], participant: Participant, profile: str, output: Path) -> Dict[str, object]:
    """对 1×1×3/5/6 mm 输入运行特征链；只输出数值稳定性，不生成QC图片。"""

    root = Path(str(config["project_root"]))
    derivatives = Path(str(config["derivatives"]))
    derivatives = derivatives if derivatives.is_absolute() else root / derivatives
    native_dir = derivatives / participant.bids_id
    native_wmh_status = _read_json(native_dir / "status" / "wmh.json")
    native_t1_status = _read_json(native_dir / "status" / "t1.json")
    if native_wmh_status.get("status") != "pass" or native_t1_status.get("status") != "pass":
        raise RuntimeError("低分辨率验证要求原生 WMH/T1 均已通过")
    native_wmh = _read_json(Path(str(native_wmh_status["details"]["feature_json"])))
    native_t1 = _read_json(Path(str(native_t1_status["details"]["feature_json"])))
    baseline = dict(native_wmh["z_chung"])
    baseline.update(native_t1["macro20_atrophy_z"])
    if len(baseline) != 40:
        raise ValueError("原生基准不是完整 40 维")

    expected_provenance = {
        "schema_version": "2.0_contralateral_synthstrip",
        "participant_id": participant.participant_id,
        "t1w_sha256": sha256(participant.t1w),
        "flair_sha256": sha256(participant.flair),
        "lesion_sha256": sha256(participant.lesion_mask),
        "native_wmh_features_sha256": sha256(Path(str(native_wmh_status["details"]["feature_json"]))),
        "native_t1_features_sha256": sha256(Path(str(native_t1_status["details"]["feature_json"]))),
        "profile": profile,
        "spacings_mm": config["execution"]["lowres_spacings_mm"],  # type: ignore[index]
    }
    provenance_path = output / "lowres_provenance.json"
    completion_path = output / "lowres_validation.json"
    if provenance_path.is_file() and completion_path.is_file():
        previous_provenance = _read_json(provenance_path)
        previous_completion = _read_json(completion_path)
        if previous_provenance == expected_provenance and previous_completion.get("full_feature_stability_completed") is True:
            modality_report = output / "lowres_modality_stability.tsv"
            delta_report = Path(str(previous_completion["feature_delta_report"]))
            _write_modality_stability(delta_report, modality_report)
            previous_completion["modality_report"] = str(modality_report)
            previous_completion["validation_reused"] = True
            _write_json(completion_path, previous_completion)
            return previous_completion

    core_python = root / str(config["execution"]["core_python"])  # type: ignore[index]
    wmh_python = root / str(config["execution"]["wmh_python"])  # type: ignore[index]
    t1_python = root / str(config["execution"]["t1_python"])  # type: ignore[index]
    for executable in (core_python, wmh_python, t1_python):
        if not executable.is_file():
            raise FileNotFoundError("低分辨率验证缺少解释器 {}".format(executable))

    output.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []
    spacings = config["execution"]["lowres_spacings_mm"]  # type: ignore[index]
    for spacing in spacings:
        suffix = "x".join("{:g}".format(float(value)) for value in spacing)
        variant = output / "spacing-{}".format(suffix)
        inputs = variant / "inputs"
        flair_out = inputs / "FLAIR_spacing-{}.nii.gz".format(suffix)
        t1_out = inputs / "T1w_spacing-{}.nii.gz".format(suffix)
        downsample_image(participant.flair, spacing, flair_out, order=1)
        downsample_image(participant.t1w, spacing, t1_out, order=1)

        variant_derivatives = variant / "derivatives"
        participant_table = variant / "config" / "participants.tsv"
        participant_table.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[participant.participant_id, participant.age, participant.sex, participant.site_id,
              str(t1_out), str(flair_out), str(participant.lesion_mask)]],
            columns=PARTICIPANT_COLUMNS,
        ).to_csv(participant_table, sep="\t", index=False)
        variant_config = copy.deepcopy(dict(config))
        variant_config["project_root"] = str(root)
        variant_config["participants"] = str(participant_table)
        variant_config["derivatives"] = str(variant_derivatives)
        variant_config_path = variant / "config" / "config.yaml"
        with variant_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(variant_config, handle, allow_unicode=True, sort_keys=False)

        stage_specs = [
            ("registration", core_python, ["--profile", profile]),
            ("lesion", core_python, []),
            ("wmh_seg", wmh_python, ["--profile", profile]),
            ("wmh", core_python, ["--profile", profile]),
            ("t1", t1_python, ["--profile", profile]),
        ]
        case_status_dir = variant_derivatives / participant.bids_id / "status"
        failed_node = ""
        driver_log = variant / "lowres_driver.log"
        for stage, python, extra in stage_specs:
            status_file = case_status_dir / "{}.json".format(stage)
            command_name = "wmh-seg" if stage == "wmh_seg" else stage
            command = [str(python), "-m", "substain_features.cli", "stage", command_name,
                       "--config-file", str(variant_config_path), "--participant-id", participant.participant_id] + extra
            env = os.environ.copy()
            # core 包装器会把 envs/core-site 注入 PYTHONPATH；不得传给隔离的
            # WMH/T1 Conda 环境，否则会混用 ABI 不兼容的 NumPy 二进制。
            env["PYTHONPATH"] = str(root / "src")
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            with driver_log.open("a", encoding="utf-8") as log_handle:
                completed = subprocess.run(command, cwd=str(root), env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
            state = _read_json(status_file) if status_file.is_file() else {"status": "missing"}
            if completed.returncode != 0 or state.get("status") != "pass":
                failed_node = stage
                break

        row: Dict[str, object] = {
            "participant_id": participant.participant_id,
            "spacing_mm": suffix,
            "status": "fail" if failed_node else "pass",
            "failed_node": failed_node,
            "flair": str(flair_out),
            "t1w": str(t1_out),
        }
        if not failed_node:
            wmh_status = _read_json(case_status_dir / "wmh.json")
            t1_status = _read_json(case_status_dir / "t1.json")
            variant_wmh = _read_json(Path(str(wmh_status["details"]["feature_json"])))
            variant_t1 = _read_json(Path(str(t1_status["details"]["feature_json"])))
            observed = dict(variant_wmh["z_chung"])
            observed.update(variant_t1["macro20_atrophy_z"])
            if list(observed) != list(baseline):
                raise ValueError("{} 低分辨率特征顺序与原生 40 维不一致".format(suffix))
            reference_values = np.asarray(list(baseline.values()), dtype=float)
            observed_values = np.asarray(list(observed.values()), dtype=float)
            delta = observed_values - reference_values
            row.update(
                {
                    "feature_count": 40,
                    "pearson_r": float(np.corrcoef(reference_values, observed_values)[0, 1]),
                    "mae_z": float(np.mean(np.abs(delta))),
                    "rmse_z": float(np.sqrt(np.mean(delta ** 2))),
                    "max_abs_delta_z": float(np.max(np.abs(delta))),
                }
            )
            for feature, native_value, lowres_value, difference in zip(baseline, reference_values, observed_values, delta):
                delta_rows.append(
                    {
                        "participant_id": participant.participant_id,
                        "spacing_mm": suffix,
                        "feature_name": feature,
                        "native_z": float(native_value),
                        "lowres_z": float(lowres_value),
                        "delta_z": float(difference),
                        "abs_delta_z": float(abs(difference)),
                    }
                )
        summary_rows.append(row)

    summary_report = output / "lowres_feature_stability.tsv"
    delta_report = output / "lowres_feature_deltas.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_report, sep="\t", index=False)
    pd.DataFrame(delta_rows).to_csv(delta_report, sep="\t", index=False)
    modality_report = _write_modality_stability(delta_report, output / "lowres_modality_stability.tsv")
    completed_all = len(summary_rows) == len(spacings) and all(row["status"] == "pass" for row in summary_rows)
    result = {
        "report": str(summary_report),
        "feature_delta_report": str(delta_report),
        "modality_report": str(modality_report),
        "spacings": spacings,
        "full_feature_stability_completed": completed_all,
        "failed_variants": [row["spacing_mm"] for row in summary_rows if row["status"] != "pass"],
        "validation_reused": False,
    }
    _write_json(provenance_path, expected_provenance)
    _write_json(completion_path, result)
    if not completed_all:
        failures = ["{}:{}".format(row["spacing_mm"], row["failed_node"]) for row in summary_rows if row["status"] != "pass"]
        raise RuntimeError("低分辨率完整链未全部通过: {}；见 {}".format(failures, summary_report))
    return result
