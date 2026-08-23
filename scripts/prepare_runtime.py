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


old = "model_file = os.path.join('/app/models', 'WMH-SynthSeg_v10_231110.pth')"
new = "model_file = os.environ['SUBSTAIN_WMH_MODEL']"
text = replace_once(text, old, new, "model_path")
load_old = "torch.load(model_file, map_location=device)"
load_new = "torch.load(model_file, map_location=device, weights_only=False)"
text = replace_once(text, load_old, load_new, "torch_load")

text = replace_once(
    text,
    '    parser.add_argument("--crop", action="store_true", help="(optional) Does two passes, to limit size to 192x224x192 cuboid (needed for GPU processing)")\n',
    '    parser.add_argument("--crop", action="store_true", help="(optional) Does two passes, to limit size to 192x224x192 cuboid (needed for GPU processing)")\n'
    '    parser.add_argument("--gpu_fp16", action="store_true",\n'
    '                        help="(optional) Use FP16 for CUDA inference and release logits between passes")\n',
    "gpu_fp16_argument",
)
text = replace_once(text, "    crop = args.crop\n", "    crop = args.crop\n    gpu_fp16 = args.gpu_fp16\n", "gpu_fp16_value")
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
text = replace_once(
    text,
    "                    pred = model(cuboid[None, None, ...])\n"
    "                    seg_p = Softmax(dim=0)(pred[0, 0:n_labels, ...])\n",
    "                    model_input = cuboid.half() if gpu_fp16 else cuboid\n"
    "                    pred = model(model_input[None, None, ...])\n"
    "                    seg_p = Softmax(dim=0)(pred[0, 0:n_labels, ...].float())\n",
    "gpu_fp16_preliminary",
)
text = replace_once(
    text,
    "                    upscaled = upscaled[:min(192,upscaled.shape[0]), :min(224,upscaled.shape[1]), :min(192,upscaled.shape[2])]\n",
    "                    upscaled = upscaled[:min(192,upscaled.shape[0]), :min(224,upscaled.shape[1]), :min(192,upscaled.shape[2])]\n"
    "\n"
    "                    if gpu_fp16:\n"
    "                        # 初定位张量不再参与最终推理，必须在第二次大体积前向前释放。\n"
    "                        del pred, seg_p, p_th_lv, vi, vj, vk, gi, gj, gk, den, model_input, cuboid\n"
    "                        torch.cuda.empty_cache()\n",
    "gpu_fp16_preliminary_release",
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
    "WMH-SynthSeg\ttrue\tmodel path reads SUBSTAIN_WMH_MODEL; torch.load weights_only=False; CUDA FP16 staged inference\toffline model path, PyTorch compatibility, and 16 GB GPU memory safety\n"
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
    "../envs/core-venv/bin/python",
    "../envs/core-venv/bin/substain-features",
    "../envs/core-venv/bin/snakemake",
    "../envs/core-venv/bin/pytest",
):
    path = ROOT / "scripts" / name if not name.startswith("../") else ROOT / name[3:]
    path.chmod(path.stat().st_mode | 0o111)
print("运行时副本已准备：{}".format(runtime))
