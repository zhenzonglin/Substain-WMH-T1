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
old = "model_file = os.path.join('/app/models', 'WMH-SynthSeg_v10_231110.pth')"
new = "model_file = os.environ['SUBSTAIN_WMH_MODEL']"
if old not in text:
    raise RuntimeError("pinned WMH-SynthSeg 源码中的模型路径锚点发生变化")
text = text.replace(old, new, 1)
load_old = "torch.load(model_file, map_location=device)"
load_new = "torch.load(model_file, map_location=device, weights_only=False)"
if load_old not in text:
    raise RuntimeError("pinned WMH-SynthSeg 源码中的 torch.load 锚点发生变化")
target.write_text(text.replace(load_old, load_new, 1), encoding="utf-8")

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
    "WMH-SynthSeg\ttrue\tmodel path reads SUBSTAIN_WMH_MODEL; torch.load weights_only=False\toffline model path and PyTorch 2.6 compatibility for hash-verified checkpoint\n"
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
