import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from substain_features.synthstrip import run_synthstrip


def test_synthstrip_uses_fixed_model_border_and_gpu_flag(tmp_path: Path, monkeypatch: object) -> None:
    t1w = tmp_path / "t1.nii.gz"
    runtime = tmp_path / "mri_synthstrip"
    model = tmp_path / "synthstrip.1.pt"
    python = tmp_path / "python"
    for path, content in ((t1w, b"t1"), (runtime, b"runtime"), (model, b"model"), (python, b"python")):
        path.write_bytes(content)
    expected_hash = hashlib.sha256(b"model").hexdigest()
    output_brain = tmp_path / "brain.nii.gz"
    output_mask = tmp_path / "mask.nii.gz"
    captured = {}

    def fake_run(command: object, **kwargs: object) -> object:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        output_brain.write_bytes(b"brain")
        output_mask.write_bytes(b"mask")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("substain_features.synthstrip.subprocess.run", fake_run)
    run_synthstrip(
        t1w,
        output_brain,
        output_mask,
        runtime,
        model,
        python,
        "cuda",
        1.0,
        expected_hash,
        tmp_path / "run.log",
    )
    command = captured["command"]
    assert command[0] == str(python)
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("-b") + 1] == "1"
    assert "-g" in command
    assert "PYTHONPATH" not in captured["environment"]
    assert captured["environment"]["PYTHONNOUSERSITE"] == "1"


def test_synthstrip_rejects_wrong_model_hash_before_execution(tmp_path: Path, monkeypatch: object) -> None:
    paths = [tmp_path / name for name in ("t1", "runtime", "model", "python")]
    for path in paths:
        path.write_bytes(b"x")
    called = {"value": False}

    def fake_run(*args: object, **kwargs: object) -> object:
        called["value"] = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("substain_features.synthstrip.subprocess.run", fake_run)
    with pytest.raises(ValueError, match="SHA256"):
        run_synthstrip(
            paths[0], tmp_path / "brain", tmp_path / "mask", paths[1], paths[2], paths[3],
            "cpu", 1, "0" * 64, tmp_path / "run.log"
        )
    assert called["value"] is False
