#!/usr/bin/env python3
"""避免跨 Windows/WSL 多层 shell 引号，机械生成受控运行时副本。"""

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / "resources/runtime"
runtime.mkdir(parents=True, exist_ok=True)
source = ROOT / "resources/third_party/WMH-SynthSeg/WMHSynthSeg/inference.py"
target = runtime / "wmh_synthseg_inference.py"
text = source.read_text(encoding="utf-8")


def replace_once(blob: str, old: str, new: str, label: str) -> str:
    """固定源码锚点必须唯一，避免上游变化后静默生成错误运行时。"""

    if blob.count(old) != 1:
        raise RuntimeError("pinned WMH-SynthSeg 源码锚点异常: {}".format(label))
    return blob.replace(old, new, 1)


def replace_block(blob: str, start: str, end: str, new: str, label: str) -> str:
    """用两个唯一锚点替换一段固定源码，避免复制整段上游实现。"""

    if blob.count(start) != 1 or blob.count(end) != 1:
        raise RuntimeError("pinned WMH-SynthSeg 源码块锚点异常: {}".format(label))
    start_index = blob.index(start)
    end_index = blob.index(end, start_index)
    return blob[:start_index] + new + blob[end_index:]


old = "model_file = os.path.join('/app/models', 'WMH-SynthSeg_v10_231110.pth')"
new = "model_file = os.environ['SUBSTAIN_WMH_MODEL']"
text = replace_once(text, old, new, "model_path")
load_old = "torch.load(model_file, map_location=device)"
load_new = "torch.load(model_file, map_location=device, weights_only=False)"
text = replace_once(text, load_old, load_new, "torch_load")
text = replace_once(
    text,
    "import argparse\n",
    "import argparse\nimport json\n\n"
    "from substain_features.wmh_window import compute_adaptive_window, nearest_resample_indices\n",
    "adaptive_window_import",
)

text = replace_once(
    text,
    '    parser.add_argument("--crop", action="store_true", help="(optional) Does two passes, to limit size to 192x224x192 cuboid (needed for GPU processing)")\n',
    '    parser.add_argument("--crop", action="store_true", help="Use the SynthStrip mask to select an adaptive model window")\n'
    '    parser.add_argument("--brain_mask", required=True, help="FLAIR SynthStrip binary brain mask")\n'
    '    parser.add_argument("--window_metadata", required=True, help="JSON record of adaptive window geometry")\n'
    '    parser.add_argument("--gpu_fp16", action="store_true",\n'
    '                        help="(optional) Use FP16 for CUDA inference and release logits between passes")\n',
    "gpu_fp16_argument",
)
text = replace_once(text, "    crop = args.crop\n", "    crop = args.crop\n    gpu_fp16 = args.gpu_fp16\n", "gpu_fp16_value")
text = replace_once(
    text,
    "    output_csv_path = args.csv_vols\n",
    "    output_csv_path = args.csv_vols\n"
    "    brain_mask_path = args.brain_mask\n"
    "    window_metadata_path = args.window_metadata\n",
    "adaptive_window_values",
)
text = replace_once(
    text,
    "    if os.path.exists(input_path) is False:\n        raise Exception('Input does not exist')\n",
    "    if os.path.exists(input_path) is False:\n"
    "        raise Exception('Input does not exist')\n"
    "    if not os.path.isfile(brain_mask_path):\n"
    "        raise Exception('FLAIR SynthStrip brain mask does not exist')\n"
    "    if not crop:\n"
    "        raise Exception('Project runtime requires --crop adaptive window mode')\n"
    "    if os.path.isdir(input_path):\n"
    "        raise Exception('Adaptive window runtime accepts one input image per process')\n"
    "    metadata_parent = os.path.dirname(window_metadata_path)\n"
    "    if metadata_parent and not os.path.isdir(metadata_parent):\n"
    "        raise Exception('Parent directory of window metadata does not exist')\n",
    "adaptive_window_validation",
)
text = replace_once(
    text,
    "    device = torch.device(device)\n",
    "    device = torch.device(device)\n"
    "    if gpu_fp16 and device.type != 'cuda':\n"
    "        raise ValueError('--gpu_fp16 只允许与 --device cuda 一起使用')\n",
    "gpu_fp16_validation",
)
text = replace_once(
    text,
    "                image, aff = MRIread(input_file)\n"
    "                image_torch = torch.tensor(np.squeeze(image).astype(float), device=device)\n"
    "                while len(image_torch.shape) > 3:\n"
    "                    image_torch = image_torch.mean(image, dim=-1)\n"
    "                image_torch, aff2 = align_volume_to_ref(image_torch, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)\n",
    "                image, aff = MRIread(input_file)\n"
    "                image_array = np.squeeze(image).astype(float)\n"
    "                image_torch = torch.tensor(image_array, device=device)\n"
    "                while len(image_torch.shape) > 3:\n"
    "                    image_torch = image_torch.mean(dim=-1)\n"
    "                brain_mask, brain_mask_aff = MRIread(brain_mask_path)\n"
    "                brain_mask_array = np.squeeze(brain_mask).astype(float)\n"
    "                if tuple(brain_mask_array.shape) != tuple(image_torch.shape) or not np.allclose(brain_mask_aff, aff, atol=1e-4):\n"
    "                    raise ValueError('FLAIR and SynthStrip brain mask grids do not match')\n"
    "                brain_mask_torch = torch.tensor(brain_mask_array, device=device)\n"
    "                image_torch, aff2 = align_volume_to_ref(image_torch, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)\n"
    "                brain_mask_torch, brain_mask_aff2 = align_volume_to_ref(brain_mask_torch, brain_mask_aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)\n"
    "                if tuple(brain_mask_torch.shape) != tuple(image_torch.shape) or not np.allclose(brain_mask_aff2, aff2, atol=1e-4):\n"
    "                    raise ValueError('Aligned FLAIR and brain mask grids do not match')\n",
    "adaptive_window_mask_load",
)
text = replace_once(
    text,
    "        model.load_state_dict(checkpoint['model_state_dict'])\n",
    "        model.load_state_dict(checkpoint['model_state_dict'])\n"
    "        model.eval()\n"
    "        if gpu_fp16:\n"
    "            # 192x224x192 的 FP32 3D U-Net 峰值超过 16 GB。权重和输入使用 FP16，\n"
    "            # softmax 与体积汇总仍转回 FP32，避免累计精度继续下降。\n"
    "            model.half()\n"
    "            print('Using FP16 CUDA inference with staged GPU memory release')\n",
    "gpu_fp16_model",
)
text = replace_block(
    text,
    "                if crop:\n",
    "                upscaled_padded = torch.zeros(tuple((np.ceil(np.array(upscaled.shape) / 32.0) * 32).astype(int)),  device=device)\n",
    "                mask_indices = [\n"
    "                    torch.as_tensor(nearest_resample_indices(int(size), float(factor)), device=device)\n"
    "                    for size, factor in zip(brain_mask_torch.shape, factors)\n"
    "                ]\n"
    "                mask_upscaled = brain_mask_torch[\n"
    "                    mask_indices[0][:, None, None],\n"
    "                    mask_indices[1][None, :, None],\n"
    "                    mask_indices[2][None, None, :],\n"
    "                ] > 0.5\n"
    "                if tuple(mask_upscaled.shape) != tuple(upscaled.shape):\n"
    "                    raise ValueError('1 mm FLAIR and brain mask shapes do not match')\n"
    "                window = compute_adaptive_window(mask_upscaled.detach().cpu().numpy())\n"
    "                crop_slices = tuple(slice(start, stop) for start, stop in zip(window.crop_start, window.crop_stop))\n"
    "                upscaled = upscaled[crop_slices]\n"
    "                aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] + np.matmul(\n"
    "                    aff_upscaled[:-1, :-1], np.asarray(window.crop_start, dtype=float)\n"
    "                )\n"
    "",
    "adaptive_window_crop",
)
text = replace_once(
    text,
    "                upscaled_padded = torch.zeros(tuple((np.ceil(np.array(upscaled.shape) / 32.0) * 32).astype(int)),  device=device)\n",
    "                upscaled_padded = torch.zeros(window.tensor_shape, device=device)\n",
    "adaptive_window_tensor_padding",
)
text = replace_once(
    text,
    "                pred1 = model(upscaled_padded[None, None, ...])[:, :, :upscaled.shape[0], :upscaled.shape[1],\n"
    "                        :upscaled.shape[2]]\n"
    "                pred2 = torch.flip(model(torch.flip(upscaled_padded, [0])[None, None, ...]), [2])[:, :,\n"
    "                        :upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]]\n",
    "                if gpu_fp16:\n"
    "                    # 两个方向依次在 GPU 上计算，logits 立即移到 CPU；否则第二次前向会与\n"
    "                    # 第一次输出共同占用显存并再次触发 Windows/WSL 显存驻留失败。\n"
    "                    pred1_gpu = model(upscaled_padded.half()[None, None, ...])[:, :, :upscaled.shape[0],\n"
    "                                :upscaled.shape[1], :upscaled.shape[2]]\n"
    "                    pred1 = pred1_gpu.float().cpu()\n"
    "                    del pred1_gpu\n"
    "                    torch.cuda.empty_cache()\n"
    "                    pred2_gpu = torch.flip(model(torch.flip(upscaled_padded, [0]).half()[None, None, ...]), [2])[:, :,\n"
    "                                :upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]]\n"
    "                    pred2 = pred2_gpu.float().cpu()\n"
    "                    del pred2_gpu\n"
    "                    torch.cuda.empty_cache()\n"
    "                    label_list_segmentation_torch = label_list_segmentation_torch.cpu()\n"
    "                else:\n"
    "                    pred1 = model(upscaled_padded[None, None, ...])[:, :, :upscaled.shape[0], :upscaled.shape[1],\n"
    "                            :upscaled.shape[2]]\n"
    "                    pred2 = torch.flip(model(torch.flip(upscaled_padded, [0])[None, None, ...]), [2])[:, :,\n"
    "                            :upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]]\n",
    "gpu_fp16_staged_forward",
)
text = replace_once(
    text,
    "                print(' ')\n\n            except Exception as e:\n"
    "                print(' '); print(\"     An error occurred in this volume:\", str(e)); print(' ');\n",
    "                window_record = window.as_dict()\n"
    "                window_record['output_model_grid_shape'] = [int(value) for value in pred_seg.shape]\n"
    "                window_record['output_model_grid_affine'] = np.asarray(aff_upscaled).round(8).tolist()\n"
    "                with open(window_metadata_path, 'w', encoding='utf-8') as metadata_handle:\n"
    "                    json.dump(window_record, metadata_handle, ensure_ascii=False, indent=2)\n"
    "                print(' ')\n\n"
    "            except Exception as e:\n"
    "                print(' '); print(\"     An error occurred in this volume:\", str(e)); print(' ');\n"
    "                raise RuntimeError('WMH-SynthSeg volume failed: ' + str(e)) from e\n",
    "runtime_failure_propagation_and_metadata",
)
target.write_text(text, encoding="utf-8")

# SynthStrip固定源码保持不变；项目运行时副本只显式适配PyTorch 2.7的安全加载默认值。
synthstrip_source = ROOT / "resources/third_party/SynthStrip/mri_synthstrip/mri_synthstrip"
synthstrip_runtime_dir = ROOT / "resources/tools/synthstrip"
synthstrip_runtime_dir.mkdir(parents=True, exist_ok=True)
synthstrip_target = synthstrip_runtime_dir / "mri_synthstrip"
synthstrip_text = synthstrip_source.read_text(encoding="utf-8")
synthstrip_load_old = "torch.load(modelfile, map_location=device)"
synthstrip_load_new = "torch.load(modelfile, map_location=device, weights_only=False)"
if synthstrip_load_old not in synthstrip_text:
    raise RuntimeError("pinned SynthStrip源码中的torch.load锚点发生变化")
synthstrip_target.write_text(
    synthstrip_text.replace(synthstrip_load_old, synthstrip_load_new, 1),
    encoding="utf-8",
)
synthstrip_target.chmod(synthstrip_target.stat().st_mode | 0o111)

dlmuse_runtime = runtime / "DLMUSE"
if not dlmuse_runtime.exists():
    shutil.copytree(ROOT / "resources/third_party/DLMUSE", dlmuse_runtime)

# NiChart 元数据错误地声明 PyPI 的 pathlib 回移植包；Python 3.10 已内置 pathlib。
# 固定源码 clone 保持不变，仅在可审计运行时副本中删除这一无效依赖。
nichart_runtime = runtime / "NiChart_DLMUSE"
if not nichart_runtime.exists():
    shutil.copytree(ROOT / "resources/third_party/NiChart_DLMUSE", nichart_runtime)
nichart_setup = nichart_runtime / "setup.py"
nichart_text = nichart_setup.read_text(encoding="utf-8")
pathlib_requirement = '        "pathlib",\n'
if pathlib_requirement in nichart_text:
    nichart_setup.write_text(nichart_text.replace(pathlib_requirement, "", 1), encoding="utf-8")

(runtime / "RUNTIME_PATCHES.tsv").write_text(
    "component\tfixed_source_unchanged\truntime_change\treason\n"
    "WMH-SynthSeg\ttrue\tmodel path reads SUBSTAIN_WMH_MODEL; SynthStrip-mask adaptive window; strict failure propagation; CUDA FP16 staged inference\toffline model path, clinical FLAIR geometry compatibility, explicit failures, and GPU memory safety\n"
    "SynthStrip\ttrue\ttorch.load weights_only=False\tPyTorch 2.7 compatibility for hash-verified model v1\n"
    "NiChart_DLMUSE\ttrue\tremove PyPI pathlib dependency\tPython 3.10 pathlib is standard library\n",
    encoding="utf-8",
)

for name in (
    "install_core.sh",
    "prepare_runtime.sh",
    "download_resources.sh",
    "install_offline.sh",
    "build_offline_bundle.sh",
    "build_mapping.py",
    "prepare_runtime.py",
    "install_full_envs.sh",
    "offline_smoke.sh",
):
    # 这里只处理项目脚本；虚拟环境入口可能位于只允许执行、不允许 chmod 的共享存储。
    path = ROOT / "scripts" / name
    path.chmod(path.stat().st_mode | 0o111)
print("运行时副本已准备：{}".format(runtime))
