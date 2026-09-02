"""DLMUSE 调用、145/GM119 原生体积和宏区聚合。"""

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .mapping import aggregate_macro20, assert_volume_conservation, official_gm119
from .resources import sha256


VENTRICLE_LABELS = {4, 11, 49, 50, 51, 52}


def dlmuse_model_provenance() -> Dict[str, str]:
    """确认 DLICV/DLMUSE 离线权重完整，并把关键摘要纳入复用契约。

    单元测试使用伪入口且不安装 DLMUSE，因此模块不存在时返回空字典；正式
    t1 环境一旦能定位 DLMUSE 包，缺失任一关键文件都立即失败，避免先运行
    耗时的 DLICV 后才发现分割模型不完整。
    """

    spec = importlib.util.find_spec("DLMUSE")
    if spec is None or spec.origin is None:
        return {}
    package_dir = Path(spec.origin).resolve().parent
    model_dir = (
        package_dir
        / "nnunet_results"
        / "Dataset903_Task903_DLMUSEV2"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    required = {
        "dlmuse_dataset_json_sha256": model_dir / "dataset.json",
        "dlmuse_plans_json_sha256": model_dir / "plans.json",
        "dlmuse_checkpoint_final_sha256": model_dir / "fold_0" / "checkpoint_final.pth",
    }
    dlicv_spec = importlib.util.find_spec("DLICV")
    if dlicv_spec is None or dlicv_spec.origin is None:
        raise FileNotFoundError("t1 环境能定位 DLMUSE 但缺少 DLICV 包")
    dlicv_model_dir = (
        Path(dlicv_spec.origin).resolve().parent
        / "nnunet_results"
        / "Dataset901_Task901_dlicv"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    required.update(
        {
            "dlicv_dataset_json_sha256": dlicv_model_dir / "dataset.json",
            "dlicv_plans_json_sha256": dlicv_model_dir / "plans.json",
            "dlicv_checkpoint_final_sha256": dlicv_model_dir / "fold_0" / "checkpoint_final.pth",
        }
    )
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("DLMUSE 离线模型不完整: {}".format(missing))
    return {key: sha256(path) for key, path in required.items()}


def display_name_map(derived_mapping_path: Path) -> Dict[int, str]:
    """从 NiChart 派生映射底部读取 GenMIND 使用的原生 ROI 显示名。"""

    names: Dict[int, str] = {}
    with derived_mapping_path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) == 3 and row[0] == row[2]:
                names[int(row[0])] = row[1]
    return names


def genmind_native145_labels(genmind_csv: Path, derived_mapping_path: Path) -> List[int]:
    """用 GenMIND 表头与 NiChart 官方显示名匹配，固定真正参与常模的 145 个标签。"""

    columns = pd.read_csv(genmind_csv, nrows=0).columns.tolist()[4:]
    names = display_name_map(derived_mapping_path)
    inverse = {name: label for label, name in names.items()}
    missing = [name for name in columns if name not in inverse]
    if missing:
        raise ValueError("GenMIND 列无法匹配 MUSE 标签: {}".format(missing[:10]))
    labels = [inverse[name] for name in columns]
    if len(labels) != 145 or len(set(labels)) != 145:
        raise ValueError("GenMIND 原生 ROI 应为 145 个唯一标签")
    return labels


def extract_native_volumes_ml(segmentation: Path, labels: Sequence[int]) -> Dict[int, float]:
    """直接从 DLMUSE 原生标签图计数并转换为 mL。"""

    image = nib.load(str(segmentation))
    data = np.rint(image.get_fdata()).astype(np.int16)
    unique, counts = np.unique(data, return_counts=True)
    count_map = {int(label): int(count) for label, count in zip(unique, counts)}
    # 允许分割中存在 GenMIND 未使用的 6 个标签，但记录交由 QC；这里不静默替换标签。
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return {int(label): float(count_map.get(int(label), 0) * voxel_ml) for label in labels}


def gm119_volumes(native_volumes: Mapping[int, float], derived_mapping_path: Path) -> Dict[int, float]:
    labels = sorted(official_gm119(derived_mapping_path))
    absent = [label for label in labels if label not in native_volumes]
    if absent:
        raise ValueError("native145 中缺少 GM119 标签 {}".format(absent))
    return {label: float(native_volumes[label]) for label in labels}


def nonventricular_tissue_volume_ml(native_volumes: Mapping[int, float]) -> float:
    """GenMIND 145 中排除 6 个脑室标签；该分母不是 DLICV ICV。"""

    value = sum(float(volume) for label, volume in native_volumes.items() if int(label) not in VENTRICLE_LABELS)
    if value <= 0:
        raise ValueError("非脑室 MUSE 组织总体积必须大于 0")
    return float(value)


def _resolve_device(requested: str) -> Tuple[str, str]:
    """官方 NiChart 的旧版 torch 在 Blackwell 上不可执行时，显式回退 CPU。"""

    if requested != "cuda":
        return requested, ""
    try:
        import torch

        value = torch.ones(1, device="cuda")
        if float((value + value).item()) != 2.0:
            raise RuntimeError("CUDA 张量算子返回异常")
        return "cuda", ""
    except Exception as exc:
        return "cpu", "{}: {}".format(type(exc).__name__, exc)


def _resume_after_dlicv(t1w: Path, output_dir: Path, device: str, env: Mapping[str, str], log_path: Path) -> None:
    """DLICV 已成功而 DLMUSE 失败时，从 s3 掩膜继续，禁止重算约 10 分钟的 ICV。"""

    from NiChart_DLMUSE.CalcROIVol import apply_create_roi_csv, combine_roi_csv
    from NiChart_DLMUSE.MaskImage import apply_combine_masks
    from NiChart_DLMUSE.RelabelROI import apply_relabel_rois
    from NiChart_DLMUSE.ReorientImage import apply_reorient_to_init
    from NiChart_DLMUSE.dlmuse_pipeline import (
        DICT_MUSE_DERIVED,
        DICT_MUSE_NNUNET_MAP,
        DICT_MUSE_SINGLE,
        LABEL_FROM,
        LABEL_TO,
    )
    from NiChart_DLMUSE.utils import make_img_list

    working = output_dir / "temp_working_dir"
    s2 = working / "s2_dlicv"
    s3 = working / "s3_masked"
    s4 = working / "s4_dlmuse"
    s5 = working / "s5_relabeled"
    s6 = working / "s6_combined"
    masked = sorted(s3.glob("*_DLICV.nii.gz"))
    if len(masked) != 1 or not sorted(s2.glob("*_DLICV.nii.gz")):
        raise RuntimeError("DLICV 恢复点不完整，不能跳过 ICV 推理")
    dlmuse_entry = Path(sys.executable).resolve().parent / "DLMUSE"
    if not dlmuse_entry.is_file():
        raise FileNotFoundError("恢复点缺少 DLMUSE 入口 {}".format(dlmuse_entry))
    s4.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\nRESUME_FROM_DLICV_CHECKPOINT=true\n")
        completed = subprocess.run(
            [str(dlmuse_entry), "-i", str(s3), "-o", str(s4), "-device", device],
            env=dict(env), stdout=log_handle, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError("DLMUSE 恢复推理失败，exit={}；见 {}".format(completed.returncode, log_path))

    frame = make_img_list(str(t1w))
    prefix = str(frame.iloc[0].img_prefix)
    expected_s4 = s4 / "{}_DLMUSE.nii.gz".format(prefix)
    if not expected_s4.is_file():
        candidates = [path for path in s4.glob("*.nii.gz") if path.is_file()]
        discrete = []
        for candidate in candidates:
            try:
                values = np.unique(np.rint(nib.load(str(candidate)).get_fdata()).astype(np.int16))
                if len(values[values > 0]) > 20:
                    discrete.append(candidate)
            except Exception:
                continue
        if len(discrete) != 1:
            raise RuntimeError("DLMUSE 恢复后无法唯一定位连续索引标签图: {}".format(candidates))
        shutil.copyfile(discrete[0], expected_s4)

    s5.mkdir(parents=True, exist_ok=True)
    apply_relabel_rois(frame, str(s4), "_DLMUSE.nii.gz", str(s5), "_DLMUSE.nii.gz",
                       DICT_MUSE_NNUNET_MAP, LABEL_FROM, LABEL_TO)
    s6.mkdir(parents=True, exist_ok=True)
    apply_combine_masks(frame, str(s5), "_DLMUSE.nii.gz", str(s2), "_DLICV.nii.gz", str(s6), "_DLMUSE.nii.gz")
    apply_reorient_to_init(frame, str(s6), "_DLMUSE.nii.gz", str(output_dir), "_DLMUSE.nii.gz")
    apply_create_roi_csv(frame, str(output_dir), "_DLMUSE.nii.gz", DICT_MUSE_SINGLE, DICT_MUSE_DERIVED,
                         str(output_dir), "_DLMUSE_Volumes.csv")
    combine_roi_csv(frame, str(output_dir), "_DLMUSE_Volumes.csv", str(output_dir), "DLMUSE_Volumes.csv")


def run_nichart_dlmuse(t1w: Path, output_dir: Path, device: str, log_path: Path) -> Dict[str, object]:
    """在专用空目录运行 NiChart_DLMUSE；不让上游清理逻辑接触原始 BIDS。"""

    executable = shutil.which("NiChart_DLMUSE")
    if executable is None:
        sibling = Path(sys.executable).resolve().parent / "NiChart_DLMUSE"
        executable = str(sibling) if sibling.is_file() else None
    if executable is None:
        raise FileNotFoundError("当前 t1 环境中找不到 NiChart_DLMUSE")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    effective_device, fallback_reason = _resolve_device(device)
    provenance_path = output_dir.parent / "nichart_dlmuse_provenance.json"
    expected_provenance = {
        "schema_version": "1.0",
        "t1w_sha256": sha256(t1w),
        "nichart_entry_sha256": sha256(Path(executable)),
        "requested_device": device,
        "effective_device": effective_device,
    }
    expected_provenance.update(dlmuse_model_provenance())
    existing_provenance = {}
    if provenance_path.is_file():
        with provenance_path.open("r", encoding="utf-8") as handle:
            existing_provenance = json.load(handle)
    existing_segmentations = sorted(output_dir.glob("*_DLMUSE.nii.gz")) + sorted(output_dir.glob("*_DLMUSE.nii"))
    reusable = len(existing_segmentations) == 1 and all(
        existing_provenance.get(key) == value for key, value in expected_provenance.items()
    )
    command = [executable, "-i", str(t1w), "-o", str(output_dir), "-d", effective_device, "-c", "1"]
    env = os.environ.copy()
    env["PATH"] = "{}{}{}".format(Path(sys.executable).resolve().parent, os.pathsep, env.get("PATH", ""))
    # 正式分析禁止运行时下载；权重不完整时应在当前节点显式失败。
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    resumed_from_dlicv = False
    resume_candidates = sorted((output_dir / "temp_working_dir" / "s3_masked").glob("*_DLICV.nii.gz"))
    if not reusable and len(resume_candidates) == 1:
        _resume_after_dlicv(t1w, output_dir, effective_device, env, log_path)
        resumed_from_dlicv = True
    elif not reusable:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(
                "requested_device={} effective_device={} fallback_reason={}\n".format(
                    device, effective_device, fallback_reason or "none"
                )
            )
            completed = subprocess.run(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError("NiChart_DLMUSE 失败，exit={}；见 {}".format(completed.returncode, log_path))

    # NiChart 最终标签位于输出目录顶层；temp_working_dir 中含多个中间 DLMUSE 图，不能混选。
    segmentations = sorted(output_dir.glob("*_DLMUSE.nii.gz")) + sorted(output_dir.glob("*_DLMUSE.nii"))
    if not segmentations:
        # 某些版本输出名称不含 DLMUSE；要求标签图具有超过 20 个非零离散值。
        nii_candidates = sorted(output_dir.glob("*.nii.gz")) + sorted(output_dir.glob("*.nii"))
        for path in nii_candidates:
            try:
                data = nib.load(str(path)).get_fdata()
                values = np.unique(np.rint(data).astype(np.int16))
                if len(values[values > 0]) > 20:
                    segmentations.append(path)
            except Exception:
                continue
    if len(segmentations) != 1:
        raise RuntimeError("无法唯一定位 DLMUSE 标签图，候选={}".format([str(path) for path in segmentations]))
    volume_csv = sorted(output_dir.glob("DLMUSE_Volumes.csv"))
    if not reusable:
        provenance_payload = dict(expected_provenance)
        provenance_payload["initial_execution_resumed_from_dlicv_checkpoint"] = resumed_from_dlicv
        with provenance_path.open("w", encoding="utf-8") as handle:
            json.dump(provenance_payload, handle, ensure_ascii=False, indent=2)
    return {
        "segmentation": str(segmentations[0]),
        # NiChart 的派生 ROI 702 即 ICV，保存在同一体积表，而不是独立文件。
        "icv_csv": str(volume_csv[0]) if volume_csv else "",
        "volumes_csv": str(volume_csv[0]) if volume_csv else "",
        "requested_device": device,
        "effective_device": effective_device,
        "device_fallback_reason": fallback_reason,
        "inference_reused": reusable,
        "resumed_from_dlicv_checkpoint": resumed_from_dlicv,
        "initial_execution_resumed_from_dlicv_checkpoint": bool(
            existing_provenance.get("initial_execution_resumed_from_dlicv_checkpoint", resumed_from_dlicv)
        ),
        "provenance": str(provenance_path),
    }


def extract_t1_features(
    segmentation: Path,
    genmind_csv: Path,
    derived_mapping_path: Path,
    macro_mapping_path: Path,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, float], float]:
    """一次生成 native145、GM119、macro20 与非脑室分母。"""

    labels = genmind_native145_labels(genmind_csv, derived_mapping_path)
    native = extract_native_volumes_ml(segmentation, labels)
    gm = gm119_volumes(native, derived_mapping_path)
    mapping = pd.read_csv(macro_mapping_path, sep="\t")
    assert_volume_conservation(native, mapping)
    macro = aggregate_macro20(native, mapping)
    denominator = nonventricular_tissue_volume_ml(native)
    return native, gm, macro, denominator


def write_macro20_segmentation(segmentation: Path, mapping_path: Path, output: Path) -> Path:
    """把 MUSE 原生标签转换为 1..20 宏区图；未纳入宏区的标签保持 0。"""

    image = nib.load(str(segmentation))
    native = np.rint(image.get_fdata()).astype(np.int16)
    mapping = pd.read_csv(mapping_path, sep="\t")
    if mapping["native_label"].duplicated().any():
        raise ValueError("宏区映射包含重复原生标签")
    macro = np.zeros(native.shape, dtype=np.uint8)
    for row in mapping[["native_label", "macro_index"]].itertuples(index=False):
        macro[native == int(row.native_label)] = int(row.macro_index)
    present = sorted(int(value) for value in np.unique(macro) if value > 0)
    if present != list(range(1, 21)):
        raise ValueError("DLMUSE 宏区图未覆盖全部 1..20 标签，实际 {}".format(present))
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(macro, image.affine, image.header), str(output))
    return output
