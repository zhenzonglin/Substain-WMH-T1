"""FreeSurfer SynthStrip 的固定版本调用封装。"""

import os
import subprocess
from pathlib import Path
from typing import Dict

from .resources import sha256


def run_synthstrip(
    t1w: Path,
    output_brain: Path,
    output_mask: Path,
    runtime: Path,
    model: Path,
    python_executable: Path,
    device: str,
    border_mm: float,
    expected_model_sha256: str,
    log_path: Path,
) -> Dict[str, object]:
    """运行唯一允许的T1去颅骨方案；失败时不调用其他工具。"""

    for path in (t1w, runtime, model, python_executable):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    actual_hash = sha256(model)
    if actual_hash != expected_model_sha256:
        raise ValueError("SynthStrip权重SHA256不匹配: {}".format(actual_hash))

    output_brain.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not float(border_mm).is_integer():
        raise ValueError("SynthStrip v7.4.1的-b参数必须是整数毫米")
    command = [
        str(python_executable),
        str(runtime),
        "-i", str(t1w),
        "-o", str(output_brain),
        "-m", str(output_mask),
        "--model", str(model),
        "-b", str(int(border_mm)),
    ]
    if device == "cuda":
        command.append("-g")
    elif device != "cpu":
        raise ValueError("SynthStrip device只允许cpu/cuda")

    environment = os.environ.copy()
    # registration由core包装器启动；不得把core-site的Python 3.8二进制扩展
    # 传给WMH Python 3.10，否则NumPy ABI会串环境。
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n=== SynthStrip invocation ===\n")
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    if completed.returncode != 0:
        raise RuntimeError("SynthStrip失败 exit={}: {}".format(completed.returncode, " ".join(command)))
    if not output_brain.is_file() or not output_mask.is_file():
        raise RuntimeError("SynthStrip未生成去颅骨T1或脑掩膜")
    return {
        "brain": str(output_brain),
        "mask": str(output_mask),
        "device": device,
        "border_mm": border_mm,
        "model_sha256": actual_hash,
        "runtime_sha256": sha256(runtime),
        "command": command,
    }
