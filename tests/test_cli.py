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
    assert "active_config_file={}".format((project_root / "config" / "config.yaml").resolve()) in command


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


def test_aggregate_manifest_uses_configured_derivatives(tmp_path: Path, monkeypatch) -> None:
    """替代配置的资源清单必须写入该配置声明的衍生目录。"""

    configured = tmp_path / "custom_derivatives"
    captured = {}
    monkeypatch.setattr(cli, "_context", lambda path: ({"derivatives": str(configured)}, tmp_path, []))
    monkeypatch.setattr(cli, "select_participants", lambda participants, selected: [])
    monkeypatch.setattr(cli, "aggregate_outputs", lambda config, participants: {"participants": 0})

    def fake_manifest(root, output):
        captured["output"] = output

    monkeypatch.setattr(cli, "build_manifest", fake_manifest)
    result = CliRunner().invoke(
        cli.main,
        ["stage", "aggregate", "--config-file", str(tmp_path / "config.yaml"), "--participant-id", "all"],
    )
    assert result.exit_code == 0, result.output
    assert captured["output"] == configured / "tables" / "resource_manifest.tsv"


def test_lesion_order_only_dependencies_are_ancient(project_root: Path) -> None:
    """病灶状态更新不能误触发不消费病灶数值的WMH分割或T1重算。"""

    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")
    assert 'ancient(stage_pattern("lesion"))' in snakefile
    assert 'lesion=ancient(stage_pattern("lesion"))' in snakefile


def test_all_controller_finishes_computation_without_manual_qc_gate(project_root: Path) -> None:
    """批量all只能完成计算；人工QC和正式导出必须由用户随后单独启动。"""

    controller = (project_root / "run_pipeline.sh").read_text(encoding="utf-8")
    all_branch = controller.split("  all)\n", 1)[1].split("    ;;", 1)[0]
    assert 'scripts/steps/02_features.sh' in all_branch
    assert 'scripts/steps/03_qc.sh' not in all_branch
    assert 'scripts/steps/04_export.sh' not in all_branch
    assert 'scripts/steps/05_verify.sh' not in all_branch

    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")
    rule_all = snakefile.split("rule all:", 1)[1].split("# 01", 1)[0]
    assert "features_computed40.tsv" in rule_all
    assert "features_primary40.tsv" not in rule_all
