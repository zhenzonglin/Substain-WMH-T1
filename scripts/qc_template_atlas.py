#!/usr/bin/env python3
"""一次性验证 Chung 图谱在 ch2better 模板上的原始物理空间位置。"""

from pathlib import Path

from substain_features.images import save_overlay


ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "derivatives/substain_features/audit/qc_chung_atlas_on_ch2better.png"
save_overlay(
    ROOT / "resources/templates/ch2better.nii.gz",
    ROOT / "resources/templates/MNI_ch2better_WM_20ROIs.nii.gz",
    output,
    "Chung WMH20 atlas on ch2better (before registration)",
)
print(output)
