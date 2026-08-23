from pathlib import Path
from types import SimpleNamespace
import sys

import nibabel as nib
import numpy as np
import pandas as pd

from substain_features.t1 import run_nichart_dlmuse, write_macro20_segmentation


def test_nichart_runtime_is_forced_offline(tmp_path: Path, monkeypatch) -> None:
    """正式 T1 推理不得在权重缺失时静默访问 Hugging Face。"""

    captured = {}
    fake_entry = tmp_path / "fake" / "NiChart_DLMUSE"
    fake_entry.parent.mkdir()
    fake_entry.write_text("entry", encoding="utf-8")
    monkeypatch.setattr("substain_features.t1.shutil.which", lambda name: str(fake_entry))

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        output_dir = Path(command[command.index("-o") + 1])
        data = np.arange(27, dtype=np.int16).reshape((3, 3, 3))
        nib.save(nib.Nifti1Image(data, np.eye(4)), output_dir / "case_DLMUSE.nii.gz")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("substain_features.t1.subprocess.run", fake_run)
    t1w = tmp_path / "t1.nii.gz"
    t1w.write_bytes(b"t1")
    result = run_nichart_dlmuse(t1w, tmp_path / "out", "cpu", tmp_path / "run.log")
    assert Path(result["segmentation"]).is_file()
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["env"]["PATH"].split(":", 1)[0] == str(Path(sys.executable).resolve().parent)


def test_nichart_falls_back_to_current_environment_bin(tmp_path: Path, monkeypatch) -> None:
    """显式调用环境 Python 时，PATH 不含同目录也能定位入口。"""

    fake_python = tmp_path / "env" / "bin" / "python"
    fake_entry = fake_python.parent / "NiChart_DLMUSE"
    fake_python.parent.mkdir(parents=True)
    fake_entry.write_text("entry", encoding="utf-8")
    captured = {}
    monkeypatch.setattr("substain_features.t1.shutil.which", lambda name: None)
    monkeypatch.setattr("substain_features.t1.sys.executable", str(fake_python))

    def fake_run(command, **kwargs):
        captured["command"] = command
        output_dir = Path(command[command.index("-o") + 1])
        nib.save(nib.Nifti1Image(np.arange(27, dtype=np.int16).reshape((3, 3, 3)), np.eye(4)), output_dir / "case_DLMUSE.nii.gz")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("substain_features.t1.subprocess.run", fake_run)
    t1w = tmp_path / "t1.nii.gz"
    t1w.write_bytes(b"t1")
    run_nichart_dlmuse(t1w, tmp_path / "out2", "cpu", tmp_path / "run2.log")
    assert captured["command"][0] == str(fake_entry)


def test_macro20_segmentation_contains_all_groups(tmp_path: Path, project_root: Path) -> None:
    mapping_path = project_root / "resources/mappings/muse_macro20_v1_provisional.tsv"
    mapping = pd.read_csv(mapping_path, sep="\t")
    values = mapping["native_label"].astype(np.int16).to_numpy().reshape((-1, 1, 1))
    source = tmp_path / "muse.nii.gz"
    output = tmp_path / "macro20.nii.gz"
    nib.save(nib.Nifti1Image(values, np.eye(4)), source)
    write_macro20_segmentation(source, mapping_path, output)
    observed = sorted(int(value) for value in np.unique(nib.load(output).get_fdata()) if value > 0)
    assert observed == list(range(1, 21))
