"""NIfTI网格、物理空间重采样与中央QC可视化。"""

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy.ndimage import binary_closing, binary_fill_holes, label


def grid_info(path: Path) -> Dict[str, object]:
    """返回形状、体素尺寸、方向和仿射；所有空间判断基于世界坐标。"""

    image = nib.load(str(path))
    return {
        "path": str(path),
        "shape": list(image.shape[:3]),
        "zooms_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine).round(6).tolist(),
    }


def same_grid(first: Path, second: Path, atol: float = 1e-4) -> bool:
    """判断两个影像是否真正共享体素网格，而不只比较数组形状。"""

    a = nib.load(str(first))
    b = nib.load(str(second))
    return a.shape[:3] == b.shape[:3] and bool(np.allclose(a.affine, b.affine, atol=atol))


def physical_grid_diagnostics(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    max_corner_displacement_mm: float,
) -> Dict[str, object]:
    """用对应体素角点的世界坐标位移判断两个三维网格是否等价。

    形状必须完全一致，仿射必须全部为有限值。仿射系数差只用于诊断，
    真正的空间容差由八个角点的最大三维欧氏距离决定。
    """

    if not np.isfinite(max_corner_displacement_mm) or max_corner_displacement_mm < 0:
        raise ValueError("max_corner_displacement_mm必须是有限非负数")
    first_shape = tuple(int(value) for value in first.shape[:3])
    second_shape = tuple(int(value) for value in second.shape[:3])
    first_affine = np.asarray(first.affine, dtype=float)
    second_affine = np.asarray(second.affine, dtype=float)
    affines_finite = bool(np.isfinite(first_affine).all() and np.isfinite(second_affine).all())
    max_affine_abs_diff = None
    max_displacement = None
    if affines_finite:
        max_affine_abs_diff = float(np.max(np.abs(first_affine - second_affine)))
        first_extent = np.asarray(first_shape, dtype=float) - 1.0
        second_extent = np.asarray(second_shape, dtype=float) - 1.0
        first_corners = np.asarray(
            [
                [i, j, k, 1.0]
                for i in (0.0, first_extent[0])
                for j in (0.0, first_extent[1])
                for k in (0.0, first_extent[2])
            ],
            dtype=float,
        )
        second_corners = np.asarray(
            [
                [i, j, k, 1.0]
                for i in (0.0, second_extent[0])
                for j in (0.0, second_extent[1])
                for k in (0.0, second_extent[2])
            ],
            dtype=float,
        )
        first_world = (first_affine @ first_corners.T).T[:, :3]
        second_world = (second_affine @ second_corners.T).T[:, :3]
        max_displacement = float(np.linalg.norm(first_world - second_world, axis=1).max())
    shape_equal = first_shape == second_shape
    matches = bool(
        shape_equal
        and affines_finite
        and max_displacement is not None
        and max_displacement <= max_corner_displacement_mm
    )
    return {
        "shape_first": list(first_shape),
        "shape_second": list(second_shape),
        "shape_equal": shape_equal,
        "affines_finite": affines_finite,
        "max_affine_abs_diff": max_affine_abs_diff,
        "max_corner_displacement_mm": max_displacement,
        "threshold_mm": float(max_corner_displacement_mm),
        "matches": matches,
    }


def world_bounds(path: Path) -> np.ndarray:
    """计算 NIfTI 8 个角点的轴对齐世界坐标包围盒。"""

    image = nib.load(str(path))
    shape = np.asarray(image.shape[:3], dtype=float) - 1.0
    corners = np.asarray(
        [[i, j, k, 1.0] for i in (0.0, shape[0]) for j in (0.0, shape[1]) for k in (0.0, shape[2])]
    )
    world = (image.affine @ corners.T).T[:, :3]
    return np.vstack([world.min(axis=0), world.max(axis=0)])


def world_overlap_fraction(source: Path, reference: Path) -> float:
    """计算 source 世界坐标包围盒被 reference 覆盖的体积分数。"""

    src = world_bounds(source)
    ref = world_bounds(reference)
    overlap = np.maximum(0.0, np.minimum(src[1], ref[1]) - np.maximum(src[0], ref[0]))
    src_extent = np.maximum(src[1] - src[0], np.finfo(float).eps)
    return float(np.prod(overlap) / np.prod(src_extent))


def resample_label_to_reference(source: Path, reference: Path, output: Path) -> Path:
    """按 NIfTI 仿射在物理坐标中用最近邻重采样标签。"""

    source_image = nib.load(str(source))
    reference_image = nib.load(str(reference))
    result = resample_from_to(source_image, (reference_image.shape[:3], reference_image.affine), order=0)
    data = np.rint(result.get_fdata()).astype(np.uint16)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, reference_image.affine, reference_image.header), str(output))
    return output


def resample_continuous_to_reference(source: Path, reference: Path, output: Path) -> Path:
    """按世界坐标线性重采样连续概率图，并限制到合法概率范围。"""

    source_image = nib.load(str(source))
    reference_image = nib.load(str(reference))
    result = resample_from_to(source_image, (reference_image.shape[:3], reference_image.affine), order=1)
    data = np.clip(result.get_fdata(dtype=np.float32), 0.0, 1.0).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, reference_image.affine, reference_image.header), str(output))
    return output


def binary_volume_ml(path: Path, positive_label: int = 1) -> float:
    """在影像自身原生体素中计算二值/指定标签体积，单位 mL。"""

    image = nib.load(str(path))
    data = np.rint(image.get_fdata()).astype(np.int32)
    count = int(np.count_nonzero(data == positive_label)) if positive_label > 0 else int(np.count_nonzero(data))
    return count * float(np.prod(image.header.get_zooms()[:3])) / 1000.0



QC_DISPLAY_CONVENTION = "radiological_RAS_canonical_standard_axes"
QC_PANEL_ORDER = ("Coronal", "Sagittal", "Axial")
QC_SLICE_SELECTION = "max_overlay_voxel_count_per_plane"


def _max_overlay_center(mask: np.ndarray) -> Tuple[int, int, int]:
    """分别选择矢状、冠状和轴位上掩膜体素最多的层面。

    多个层面并列时，选择最靠近掩膜几何中心者；空掩膜回退到图像中心。
    这样每个正交面都展示最有信息量的层面，不再用一个三维均值坐标同时决定三幅图。
    """

    values = np.asarray(mask) > 0
    if values.ndim != 3:
        raise ValueError("QC切片选择只接受三维数组")
    nonzero = np.argwhere(values)
    if nonzero.size == 0:
        return tuple(int(value) for value in (np.asarray(values.shape) // 2))
    centroid = nonzero.mean(axis=0)
    counts = (
        values.sum(axis=(1, 2)),
        values.sum(axis=(0, 2)),
        values.sum(axis=(0, 1)),
    )
    selected = []
    for axis, axis_counts in enumerate(counts):
        candidates = np.flatnonzero(axis_counts == axis_counts.max())
        selected.append(int(candidates[np.argmin(np.abs(candidates - centroid[axis]))]))
    return tuple(selected)


def _orthogonal_qc_views(
    array: np.ndarray, center: Sequence[int], zooms_mm: Sequence[float]
) -> Sequence[Tuple[str, np.ndarray, float, float, Tuple[str, str, str, str]]]:
    """返回临床放射学方向的冠状、矢状、轴位切片。

    输入必须已由 ``nib.as_closest_canonical`` 转为RAS。所有平面水平方向翻转，
    使患者右侧显示在图像左侧；同时返回行、列的真实毫米间距。
    """

    x, y, z = (int(value) for value in center)
    dx, dy, dz = (float(value) for value in zooms_mm[:3])
    return (
        ("Coronal", np.fliplr(array[:, y, :].T), dz, dx, ("R", "L", "S", "I")),
        ("Sagittal", np.fliplr(array[x, :, :].T), dz, dy, ("A", "P", "S", "I")),
        ("Axial", np.fliplr(array[:, :, z].T), dy, dx, ("R", "L", "A", "P")),
    )


def _view_extent(array: np.ndarray, row_mm: float, column_mm: float) -> Tuple[float, float, float, float]:
    """用物理尺寸而不是像素数决定子图纵横比。"""

    return (0.0, float(array.shape[1]) * column_mm, 0.0, float(array.shape[0]) * row_mm)


def _annotate_orientation(axis: object, markers: Tuple[str, str, str, str]) -> None:
    """添加不依赖中文字体的解剖方向标记。"""

    left, right, top, bottom = markers
    style = {"color": "white", "fontsize": 9, "bbox": {"facecolor": "black", "alpha": 0.35, "pad": 0.8}}
    axis.text(0.02, 0.5, left, transform=axis.transAxes, ha="left", va="center", **style)
    axis.text(0.98, 0.5, right, transform=axis.transAxes, ha="right", va="center", **style)
    axis.text(0.5, 0.98, top, transform=axis.transAxes, ha="center", va="top", **style)
    axis.text(0.5, 0.02, bottom, transform=axis.transAxes, ha="center", va="bottom", **style)


def _spacing_text(zooms_mm: Sequence[float]) -> str:
    return "voxel {:.3g}×{:.3g}×{:.3g} mm".format(*(float(value) for value in zooms_mm[:3]))


def save_overlay(background_path: Path, overlay_path: Path, output: Path, title: str) -> Path:
    """以放射学方向保存冠状、矢状、轴位叠加图。"""

    background_image = nib.as_closest_canonical(nib.load(str(background_path)))
    overlay_image = nib.as_closest_canonical(nib.load(str(overlay_path)))
    if background_image.shape != overlay_image.shape or not np.allclose(background_image.affine, overlay_image.affine):
        overlay_image = resample_from_to(overlay_image, (background_image.shape, background_image.affine), order=0)
    background = background_image.get_fdata()
    overlay = overlay_image.get_fdata()
    center = _max_overlay_center(overlay)
    zooms = background_image.header.get_zooms()[:3]
    base_views = _orthogonal_qc_views(background, center, zooms)
    overlay_views = _orthogonal_qc_views(overlay, center, zooms)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, base_spec, overlay_spec in zip(axes, base_views, overlay_views):
        name, base_slice, row_mm, column_mm, markers = base_spec
        overlay_slice = overlay_spec[1]
        finite = base_slice[np.isfinite(base_slice)]
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + np.finfo(float).eps
        extent = _view_extent(base_slice, row_mm, column_mm)
        axis.imshow(base_slice, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=extent, aspect="equal")
        masked = np.ma.masked_where(overlay_slice <= 0, overlay_slice)
        axis.imshow(masked, cmap="autumn", alpha=0.55, interpolation="nearest", origin="lower", extent=extent, aspect="equal")
        axis.set_title(name, fontsize=10)
        _annotate_orientation(axis, markers)
        axis.axis("off")
    fig.suptitle("{} | {}".format(title, _spacing_text(zooms)))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def save_dual_overlay(
    background_path: Path,
    first_overlay_path: Path,
    second_overlay_path: Path,
    output: Path,
    title: str,
) -> Path:
    """保存两种二值掩膜叠加图：第一层橙红、第二层蓝色。"""

    background_image = nib.as_closest_canonical(nib.load(str(background_path)))
    overlays = []
    for path in (first_overlay_path, second_overlay_path):
        image = nib.as_closest_canonical(nib.load(str(path)))
        if image.shape != background_image.shape or not np.allclose(image.affine, background_image.affine):
            image = resample_from_to(image, (background_image.shape, background_image.affine), order=0)
        overlays.append(image.get_fdata() > 0)
    background = background_image.get_fdata()
    union = overlays[0] | overlays[1]
    overlap = overlays[0] & overlays[1]
    # WMH–病灶图优先展示真正重叠体素最多的层面；无重叠时才回退到联合掩膜。
    center = _max_overlay_center(overlap if np.any(overlap) else union)
    zooms = background_image.header.get_zooms()[:3]
    base_views = _orthogonal_qc_views(background, center, zooms)
    first_views = _orthogonal_qc_views(overlays[0], center, zooms)
    second_views = _orthogonal_qc_views(overlays[1], center, zooms)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, base_spec, first_spec, second_spec in zip(axes, base_views, first_views, second_views):
        name, base_slice, row_mm, column_mm, markers = base_spec
        finite = base_slice[np.isfinite(base_slice)]
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + np.finfo(float).eps
        extent = _view_extent(base_slice, row_mm, column_mm)
        axis.imshow(base_slice, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=extent, aspect="equal")
        axis.imshow(
            np.ma.masked_where(first_spec[1] <= 0, first_spec[1]),
            cmap="autumn", alpha=0.55, interpolation="nearest", origin="lower", extent=extent, aspect="equal",
        )
        axis.imshow(
            np.ma.masked_where(second_spec[1] <= 0, second_spec[1]),
            cmap="winter", alpha=0.55, interpolation="nearest", origin="lower", extent=extent, aspect="equal",
        )
        axis.set_title(name, fontsize=10)
        _annotate_orientation(axis, markers)
        axis.axis("off")
    fig.suptitle("{} | first=orange, second=blue | {}".format(title, _spacing_text(zooms)))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def save_checkerboard(reference_path: Path, moving_path: Path, output: Path, title: str) -> Path:
    """以放射学方向保存连续结构像的三正交面配准棋盘图。"""

    reference_image = nib.as_closest_canonical(nib.load(str(reference_path)))
    moving_image = nib.as_closest_canonical(nib.load(str(moving_path)))
    if reference_image.shape != moving_image.shape or not np.allclose(reference_image.affine, moving_image.affine):
        moving_image = resample_from_to(moving_image, (reference_image.shape, reference_image.affine), order=1)
    reference = reference_image.get_fdata()
    moving = moving_image.get_fdata()
    positive = reference[reference > 0]
    foreground = np.argwhere(reference > np.percentile(positive, 25)) if positive.size else np.empty((0, 3))
    center = np.asarray(reference.shape) // 2 if foreground.size == 0 else np.rint(foreground.mean(axis=0)).astype(int)
    zooms = reference_image.header.get_zooms()[:3]

    def normalize(array: np.ndarray) -> np.ndarray:
        finite = array[np.isfinite(array)]
        lo, hi = np.percentile(finite, [2, 98])
        return np.clip((array - lo) / max(hi - lo, np.finfo(float).eps), 0.0, 1.0)

    fixed_views = _orthogonal_qc_views(reference, center, zooms)
    moving_views = _orthogonal_qc_views(moving, center, zooms)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, fixed_spec, moving_spec in zip(axes, fixed_views, moving_views):
        name, fixed_slice, row_mm, column_mm, markers = fixed_spec
        moving_slice = moving_spec[1]
        fixed_norm = normalize(fixed_slice)
        moving_norm = normalize(moving_slice)
        yy, xx = np.indices(fixed_norm.shape)
        checker = ((np.floor(yy * row_mm / 20.0) + np.floor(xx * column_mm / 20.0)) % 2) == 0
        image = np.where(checker, fixed_norm, moving_norm)
        extent = _view_extent(image, row_mm, column_mm)
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent, aspect="equal")
        axis.set_title(name, fontsize=10)
        _annotate_orientation(axis, markers)
        axis.axis("off")
    fig.suptitle("{} | checkerboard | {}".format(title, _spacing_text(zooms)))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output

def downsample_image(source: Path, spacing_mm: Sequence[float], output: Path, order: int = 1) -> Path:
    """模拟临床层厚；保持世界坐标覆盖范围，不把输出再上采样用于计数。"""

    image = nib.load(str(source))
    result = resample_to_output(image, voxel_sizes=tuple(float(v) for v in spacing_mm), order=order)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output))
    return output


def brain_mask_from_wmh_synthseg(segmentation: Path, output_mask: Path, output_skullstripped: Path, reference: Path) -> Tuple[Path, Path]:
    """仅为FLAIR构造解剖脑掩膜；T1脑掩膜必须来自SynthStrip。"""

    segmentation_image = nib.load(str(segmentation))
    segmentation_data = np.rint(segmentation_image.get_fdata()).astype(np.int16)
    mask = (segmentation_data != 0) & (segmentation_data != 24)
    components, count = label(mask)
    if count == 0:
        raise ValueError("WMH-SynthSeg 未产生脑解剖标签")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    mask = components == int(np.argmax(sizes))
    mask = binary_fill_holes(binary_closing(mask, iterations=2))
    mask_image = nib.Nifti1Image(mask.astype(np.uint8), segmentation_image.affine, segmentation_image.header)

    reference_image = nib.load(str(reference))
    if mask_image.shape != reference_image.shape[:3] or not np.allclose(mask_image.affine, reference_image.affine):
        mask_image = resample_from_to(mask_image, (reference_image.shape[:3], reference_image.affine), order=0)
        mask = mask_image.get_fdata() > 0
    else:
        mask = mask_image.get_fdata() > 0
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference_image.affine, reference_image.header), str(output_mask))
    reference_data = reference_image.get_fdata(dtype=np.float32)
    nib.save(nib.Nifti1Image((reference_data * mask).astype(np.float32), reference_image.affine, reference_image.header), str(output_skullstripped))
    return output_mask, output_skullstripped
