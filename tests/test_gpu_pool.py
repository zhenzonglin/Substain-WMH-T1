import os
import sys
from pathlib import Path
from types import SimpleNamespace

from substain_features.gpu_pool import detect_gpu_ids, run_with_gpu_lock


def test_gpu_detection_reads_all_device_indices(monkeypatch) -> None:
    monkeypatch.setattr(
        "substain_features.gpu_pool.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0\n2\n"),
    )
    assert detect_gpu_ids({}) == ["0", "2"]


def test_gpu_lock_sets_one_visible_device(tmp_path: Path) -> None:
    output = tmp_path / "device.txt"
    code = "import os,pathlib; pathlib.Path({!r}).write_text(os.environ['CUDA_VISIBLE_DEVICES'])".format(str(output))
    exit_code = run_with_gpu_lock([sys.executable, "-c", code], ["3"], tmp_path / "locks")
    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "3"
    assert (tmp_path / "locks" / "gpu-3.lock").is_file()
