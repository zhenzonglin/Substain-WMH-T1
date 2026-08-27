"""WMH-SynthSeg 调用、Chung 20区体积和公开残差转换。"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.io import loadmat


WMH_REGIONS = ["basal_ganglia", "frontal", "occipital", "temporal", "parietal"]
WMH_FEATURES = ["wmh_{}_layer{}_ml".format(region, layer) for region in WMH_REGIONS for layer in range(1, 5)]


def run_wmh_synthseg(
    flair: Path,
    flair_brain_mask: Path,
    output_seg: Path,
    output_volumes_csv: Path,
    output_window_json: Path,
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
    if not flair_brain_mask.is_file():
        raise FileNotFoundError("缺少 FLAIR SynthStrip 脑掩膜 {}".format(flair_brain_mask))
    output_seg.parent.mkdir(parents=True, exist_ok=True)
    output_window_json.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    probability_path = Path(str(output_seg).replace(".nii.gz", ".lesion_probs.nii.gz"))
    expected_outputs = (output_seg, probability_path, output_volumes_csv, output_window_json)
    # 重跑前清除可能残留的旧结果，禁止新进程失败后误用上一次输出。
    for path in expected_outputs:
        if path.is_file():
            path.unlink()
    env = os.environ.copy()
    env["SUBSTAIN_WMH_MODEL"] = str(model)
    project_src = source_root.parents[2] / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root / "WMHSynthSeg"), str(project_src), env.get("PYTHONPATH", "")]
    )
    command = [
        # 三环境隔离时 PATH 可能仍先指向 core；必须沿用启动本阶段的解释器。
        sys.executable,
        str(runtime),
        "--i",
        str(flair),
        "--brain_mask",
        str(flair_brain_mask),
        "--o",
        str(output_seg),
        "--csv_vols",
        str(output_volumes_csv),
        "--window_metadata",
        str(output_window_json),
        "--device",
        device,
        "--threads",
        "1",
        "--save_lesion_probabilities",
        "--crop",
    ]
    if device == "cuda":
        # GPU推理固定使用FP16，并由运行时在两次前向之间释放显存；不回退CPU。
        command.append("--gpu_fp16")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    missing = [str(path) for path in expected_outputs if not path.is_file()]
    if completed.returncode != 0 or missing:
        suffix = "；缺少 {}".format(", ".join(missing)) if missing else ""
        raise RuntimeError("WMH-SynthSeg 失败，exit={}{}；见 {}".format(completed.returncode, suffix, log_path))
    try:
        segmentation_image = nib.load(str(output_seg))
        probability_image = nib.load(str(probability_path))
        if segmentation_image.shape[:3] != probability_image.shape[:3] or not np.allclose(
            segmentation_image.affine, probability_image.affine
        ):
            raise ValueError("硬分割与概率图网格不一致")
        probability = probability_image.get_fdata(dtype=np.float32)
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError("概率图包含非有限值或超出[0,1]")
        csv_lines = [line for line in output_volumes_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(csv_lines) < 2:
            raise ValueError("WMHSynthSeg_volumes.csv 缺少数据行")
        window = json.loads(output_window_json.read_text(encoding="utf-8"))
        if float(window.get("brain_mask_coverage", 0.0)) != 1.0:
            raise ValueError("自适应窗口未完整覆盖FLAIR脑掩膜")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("WMH-SynthSeg 输出验证失败：{}；见 {}".format(exc, log_path)) from exc
    return output_seg


def extract_wmh20_ml(corrected_wmh: Path, native_atlas: Path) -> Dict[str, float]:
    """在原生 FLAIR 体素中计算 mL，不对 FLAIR 上采样后计数。"""

    wmh_image = nib.load(str(corrected_wmh))
    atlas_image = nib.load(str(native_atlas))
    if wmh_image.shape[:3] != atlas_image.shape[:3] or not np.allclose(wmh_image.affine, atlas_image.affine):
        raise ValueError("WMH 与 20区图谱网格不一致")
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
