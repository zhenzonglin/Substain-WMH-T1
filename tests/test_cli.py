from pathlib import Path
from types import SimpleNamespace

import click
from click.testing import CliRunner

from substain_features import cli


def test_snakemake_target_precedes_config_values(project_root: Path, monkeypatch) -> None:
    """显式 Snakemake 目标不能被 --config 当成 name=value 吞掉。"""

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "shutil_which", lambda name: "/fake/snakemake")
    monkeypatch.setattr(cli, "detect_gpu_ids", lambda: [])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--config-file",
            str(project_root / "config" / "config.yaml"),
            "--participant-id",
            "all",
            "--target",
            "lowres_validation",
            "--profile",
            "cpu",
            "--skip-prepare",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    command = captured["command"]
    assert command.index("lowres_validation") < command.index("--resources")
    assert command.index("lowres_validation") < command.index("--config")
    assert command.index("--dry-run") < command.index("--config")


def test_guarded_stage_failure_returns_nonzero(tmp_path: Path, monkeypatch) -> None:
    """直接调用内部阶段仍以非零退出；Snakemake只容忍已物化的单例fail状态。"""

    participant = SimpleNamespace(participant_id="TMSXXX", bids_id="sub-TMSXXX")
    status = tmp_path / "status.json"
    monkeypatch.setattr(cli, "status_path", lambda *args: status)
    monkeypatch.setattr(cli, "guarded_stage", lambda *args: False)
    try:
        cli._run_guarded({}, participant, "lowres", lambda: None)
    except click.ClickException:
        pass
    else:
        raise AssertionError("阶段失败未产生非零 CLI 异常")
