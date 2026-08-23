#!/usr/bin/env python3
"""在禁网环境实际运行SynthStrip与完整对侧替代最小样例。"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from substain_features.resources import sha256
from substain_features.symmetry import run_contralateral_replacement
from substain_features.synthstrip import run_synthstrip


MODEL_SHA256 = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"


def _save(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 合成连续T1只用于程序可执行性，不用于评价去颅骨准确度。
    coordinates = np.indices((64, 64, 64), dtype=np.float32)
    squared_radius = sum((coordinates[index] - 31.5) ** 2 for index in range(3))
    t1_data = (1000.0 * np.exp(-squared_radius / (2.0 * 13.0 ** 2))).astype(np.float32)
    t1_path = _save(output / "synthstrip_input.nii.gz", t1_data, np.eye(4))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    synthstrip_result = run_synthstrip(
        t1_path,
        output / "synthstrip_brain.nii.gz",
        output / "synthstrip_mask.nii.gz",
        root / "resources/tools/synthstrip/mri_synthstrip",
        root / "resources/models/synthstrip.1.pt",
        Path(sys.executable),
        device,
        1,
        MODEL_SHA256,
        output / "synthstrip.log",
    )
    mask = np.asanyarray(nib.load(str(synthstrip_result["mask"])).dataobj) > 0
    if not mask.any():
        raise RuntimeError("SynthStrip烟雾测试产生空脑掩膜")

    # 同网格恒等变换也走完整FLAIR→模板→FLAIR I/O，验证物理坐标镜像和原生计数。
    shape = (33, 9, 9)
    affine = np.eye(4)
    affine[0, 3] = -16.0
    segmentation = np.zeros(shape, dtype=np.int16)
    lesion = np.zeros(shape, dtype=np.uint8)
    segmentation[12, 4, 4] = 77
    segmentation[20, 4, 4] = 77
    lesion[20, 4, 4] = 1
    wmh_seg = _save(output / "wmh_seg.nii.gz", segmentation, affine)
    lesion_path = _save(output / "lesion_flair.nii.gz", lesion, affine)
    brain_path = _save(output / "flair_brain_mask.nii.gz", np.ones(shape, dtype=np.uint8), affine)
    flair_path = _save(output / "flair.nii.gz", np.ones(shape, dtype=np.float32), affine)
    template_path = _save(output / "ch2better_smoke.nii.gz", np.ones(shape, dtype=np.float32), affine)
    correction = run_contralateral_replacement(
        wmh_seg,
        lesion_path,
        brain_path,
        flair_path,
        template_path,
        77,
        [],
        [],
        root / "resources/tools/ants-2.5.4/bin",
        output / "contralateral",
        output / "contralateral.log",
    )
    corrected = np.asanyarray(nib.load(str(correction["final_wmh"])).dataobj) > 0
    if not corrected[20, 4, 4] or not corrected[12, 4, 4]:
        raise RuntimeError("对侧WMH替代烟雾测试结果不符合预期")

    report = {
        "status": "pass",
        "network_used": False,
        "synthstrip_device": device,
        "synthstrip_model_sha256": sha256(root / "resources/models/synthstrip.1.pt"),
        "synthstrip_mask_voxels": int(mask.sum()),
        "contralateral_metrics": {
            key: value
            for key, value in correction.items()
            if key.startswith("wmh_") or key in {"symmetry_space", "symmetry_plane_world_x_mm"}
        },
    }
    (output / "algorithm_smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
