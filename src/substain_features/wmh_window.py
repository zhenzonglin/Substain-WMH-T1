"""WMH-SynthSeg 模型窗口的纯几何计算。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


WMH_MODEL_MAX_SHAPE: Tuple[int, int, int] = (192, 224, 192)
WMH_MODEL_STRIDE = 32


class WMHWindowError(ValueError):
    """输入脑组织无法由当前模型窗口完整覆盖。"""


@dataclass(frozen=True)
class AdaptiveWindow:
    """在 1 mm 模型空间中选择的真实影像窗口和临时网络张量尺寸。"""

    method: str
    input_shape_1mm: Tuple[int, int, int]
    brain_bbox_start: Tuple[int, int, int]
    brain_bbox_stop: Tuple[int, int, int]
    brain_bbox_shape: Tuple[int, int, int]
    crop_start: Tuple[int, int, int]
    crop_stop: Tuple[int, int, int]
    window_shape_1mm: Tuple[int, int, int]
    tensor_shape: Tuple[int, int, int]
    max_window_shape: Tuple[int, int, int]
    padding_multiple: int
    brain_mask_voxels: int
    brain_mask_coverage: float
    cropped_background_only: bool

    def as_dict(self) -> Dict[str, object]:
        """转换为可直接写入 JSON 的字典。"""

        return asdict(self)


def _shape3(values: Sequence[int], name: str) -> Tuple[int, int, int]:
    shape = tuple(int(value) for value in values)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("{} 必须是三个正整数，收到 {}".format(name, shape))
    return shape  # type: ignore[return-value]


def nearest_resample_indices(size: int, factor: float) -> np.ndarray:
    """复现WMH-SynthSeg 1 mm重采样坐标，并返回最近邻源体素索引。"""

    if size <= 0 or not np.isfinite(factor) or factor <= 0:
        raise ValueError("size和factor必须为有效正数")
    new_size = int(np.round(size * factor))
    delta = (1.0 - factor) / (2.0 * factor)
    coordinates = delta + np.arange(new_size, dtype=float) / factor
    coordinates = np.clip(coordinates, 0.0, float(size - 1))
    return np.rint(coordinates).astype(np.int64)


def compute_adaptive_window(
    brain_mask: np.ndarray,
    max_shape: Sequence[int] = WMH_MODEL_MAX_SHAPE,
    stride: int = WMH_MODEL_STRIDE,
) -> AdaptiveWindow:
    """依据脑掩膜选择窗口；只允许裁掉脑外背景。

    实际影像窗口保持原始 1 mm 数据，不为凑尺寸而缩放或补零。仅网络张量的
    高端边界补到 ``stride`` 的整数倍，预测后立即裁回 ``window_shape_1mm``。
    """

    if brain_mask.ndim != 3:
        raise ValueError("FLAIR 脑掩膜必须是 3D，收到 {}".format(brain_mask.shape))
    input_shape = _shape3(brain_mask.shape, "input_shape")
    maximum = _shape3(max_shape, "max_shape")
    if stride <= 0:
        raise ValueError("stride 必须为正整数")
    if any(value % stride != 0 for value in maximum):
        raise ValueError("max_shape 必须是 stride 的整数倍")

    mask = np.asarray(brain_mask) > 0
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise WMHWindowError("FLAIR SynthStrip 脑掩膜为空")
    bbox_start_array = coordinates.min(axis=0).astype(int)
    bbox_stop_array = (coordinates.max(axis=0) + 1).astype(int)
    bbox_shape_array = bbox_stop_array - bbox_start_array
    if np.any(bbox_shape_array > np.asarray(maximum, dtype=int)):
        raise WMHWindowError(
            "WMH_MODEL_FOV_EXCEEDED: 脑组织包围盒 {} 超过模型窗口 {}".format(
                tuple(int(value) for value in bbox_shape_array), maximum
            )
        )

    crop_start = []
    crop_stop = []
    for axis, (size, limit) in enumerate(zip(input_shape, maximum)):
        if size <= limit:
            start = 0
            stop = size
        else:
            center = 0.5 * float(bbox_start_array[axis] + bbox_stop_array[axis])
            start = int(np.floor(center - 0.5 * limit))
            start = min(max(start, 0), size - limit)
            stop = start + limit
        crop_start.append(start)
        crop_stop.append(stop)

    slices = tuple(slice(start, stop) for start, stop in zip(crop_start, crop_stop))
    retained_voxels = int(np.count_nonzero(mask[slices]))
    total_voxels = int(coordinates.shape[0])
    coverage = float(retained_voxels / total_voxels)
    if retained_voxels != total_voxels:
        raise WMHWindowError(
            "WMH_MODEL_FOV_EXCEEDED: 自适应窗口未完整覆盖脑掩膜 ({}/{})".format(
                retained_voxels, total_voxels
            )
        )

    window_shape = tuple(stop - start for start, stop in zip(crop_start, crop_stop))
    tensor_shape = tuple(int(np.ceil(value / stride) * stride) for value in window_shape)
    if any(value > limit for value, limit in zip(tensor_shape, maximum)):
        raise WMHWindowError(
            "网络填充尺寸 {} 超过模型窗口 {}".format(tensor_shape, maximum)
        )

    return AdaptiveWindow(
        method="flair_synthstrip_mask_v1",
        input_shape_1mm=input_shape,
        brain_bbox_start=tuple(int(value) for value in bbox_start_array),
        brain_bbox_stop=tuple(int(value) for value in bbox_stop_array),
        brain_bbox_shape=tuple(int(value) for value in bbox_shape_array),
        crop_start=tuple(crop_start),
        crop_stop=tuple(crop_stop),
        window_shape_1mm=window_shape,
        tensor_shape=tensor_shape,
        max_window_shape=maximum,
        padding_multiple=int(stride),
        brain_mask_voxels=total_voxels,
        brain_mask_coverage=coverage,
        cropped_background_only=any(start > 0 or stop < size for start, stop, size in zip(crop_start, crop_stop, input_shape)),
    )
