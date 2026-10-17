"""WMH-SynthSeg 调用、Chung 20区体积和公开残差转换。"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.io import loadmat

from .images import physical_grid_diagnostics


WMH_REGIONS = ["basal_ganglia", "frontal", "occipital", "temporal", "parietal"]
WMH_FEATURES = ["wmh_{}_layer{}_ml".format(region, layer) for region in WMH_REGIONS for layer in range(1, 5)]
WMH_ATLAS_MAX_CORNER_DISPLACEMENT_MM = 0.05


def run_wmh_synthseg(
    flair: Path,
    output_seg: Path,
    output_volumes_csv: Path,
    model: Path,
    source_root: Path,
    device: str,
    log_path: Path,
) -> Path:
    """运行固定源码的可移植副本；模型路径由环境变量注入，不修改 pinned clone。"""

    runtime = source_root.parent.parent / "runtime" / "wmh_synthseg_inference.py"
    if not runtime.is_file():
        raise FileNotFoundError("缺少运行时入口 {}；先运行 scripts/prepare_runtime.sh".format(runtime))
    if not model.is_file():
        raise FileNotFoundError("缺少 WMH-SynthSeg 权重 {}".format(model))
    output_seg.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SUBSTAIN_WMH_MODEL"] = str(model)
    env["PYTHONPATH"] = "{}{}{}".format(source_root / "WMHSynthSeg", os.pathsep, env.get("PYTHONPATH", ""))
    command = [
        # 三环境隔离时 PATH 可能仍先指向 core；必须沿用启动本阶段的解释器。
        sys.executable,
        str(runtime),
        "--i",
        str(flair),
        "--o",
        str(output_seg),
        "--csv_vols",
        str(output_volumes_csv),
        "--device",
        device,
        "--threads",
        "1",
        "--save_lesion_probabilities",
    ]
    if device == "cuda":
        # 16 GB 显卡无法容纳官方固定裁剪尺寸的 FP32 3D U-Net；GPU 推理固定使用
        # FP16，并由运行时在两次前向之间释放显存。CPU 路径仍保持 FP32。
        command.extend(["--crop", "--gpu_fp16"])
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not output_seg.is_file():
        raise RuntimeError("WMH-SynthSeg 失败，exit={}；见 {}".format(completed.returncode, log_path))
    return output_seg


def extract_wmh20_ml(
    corrected_wmh: Path,
    native_atlas: Path,
    grid_details: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """在原生 FLAIR 体素中计算 mL，不对 FLAIR 上采样后计数。"""

    wmh_image = nib.load(str(corrected_wmh))
    atlas_image = nib.load(str(native_atlas))
    comparison = physical_grid_diagnostics(
        wmh_image,
        atlas_image,
        max_corner_displacement_mm=WMH_ATLAS_MAX_CORNER_DISPLACEMENT_MM,
    )
    if grid_details is not None:
        grid_details.update(comparison)
    if not comparison["matches"]:
        raise ValueError(
            "WMH 与 20区图谱网格不一致: "
            "wmh_shape={shape_first}, atlas_shape={shape_second}, "
            "affines_finite={affines_finite}, max_affine_abs_diff={max_affine_abs_diff}, "
            "max_corner_displacement_mm={max_corner_displacement_mm}, threshold_mm={threshold_mm}".format(
                **comparison
            )
        )
    wmh = wmh_image.get_fdata() > 0
    atlas = np.rint(atlas_image.get_fdata()).astype(np.int16)
    labels = sorted(int(value) for value in np.unique(atlas) if value > 0)
    if labels != list(range(1, 21)):
        raise ValueError("Chung 图谱标签必须为 1..20，收到 {}".format(labels))
    voxel_ml = float(np.prod(wmh_image.header.get_zooms()[:3])) / 1000.0
    return {name: float(np.count_nonzero(wmh & (atlas == label)) * voxel_ml) for label, name in enumerate(WMH_FEATURES, 1)}


def _find_residual_arrays(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = {key: value for key, value in loadmat(str(mat_path)).items() if not key.startswith("__")}
    male_candidates = [np.asarray(value, dtype=float) for key, value in data.items() if "male" in key.lower() and "female" not in key.lower()]
    female_candidates = [np.asarray(value, dtype=float) for key, value in data.items() if "female" in key.lower() or "women" in key.lower()]
    male = next((array for array in male_candidates if array.shape == (20, 2)), None)
    female = next((array for array in female_candidates if array.shape == (20, 2)), None)
    if male is None or female is None:
        shaped = [np.asarray(value, dtype=float) for value in data.values() if np.asarray(value).shape == (20, 2)]
        if len(shaped) == 2:
            # 仅作为兼容路径；公开 MAT 当前变量名已包含 sex。
            male, female = shaped
    if male is None or female is None:
        raise ValueError("Residual_Info.mat 中未找到 male/female 20×2 数组")
    return male, female


def chung_zscore(volumes_ml: Mapping[str, float], sex: str, residual_info: Path) -> Dict[str, float]:
    """逐字复现公开 MATLAB 公式 z=(volume-mean)/sd；年龄未进入公式。"""

    if sex not in {"female", "male"}:
        raise ValueError("sex 只允许 female/male")
    male, female = _find_residual_arrays(residual_info)
    reference = female if sex == "female" else male
    values = np.asarray([float(volumes_ml[name]) for name in WMH_FEATURES])
    if np.any(reference[:, 1] <= 0):
        raise ValueError("Residual_Info.mat 包含非正 SD")
    z = (values - reference[:, 0]) / reference[:, 1]
    return {name.replace("_ml", "_z_chung"): float(value) for name, value in zip(WMH_FEATURES, z)}
