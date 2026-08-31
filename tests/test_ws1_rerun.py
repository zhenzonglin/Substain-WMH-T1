import json
import subprocess
import sys
from pathlib import Path


def _write_status(path: Path, stage: str, status: str, error: str = "", timestamp: str = "2026-08-31T01:00:00+00:00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "participant_id": path.parents[1].name[4:],
                "stage": stage,
                "status": status,
                "details": {"error": error} if error else {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_rerun_tool_defaults_to_dry_run_and_applies_safe_reset(project_root: Path, tmp_path: Path) -> None:
    root = tmp_path / "project"
    derivatives = root / "derivatives" / "substain_features"
    grid = derivatives / "sub-1303105005"
    interrupted = derivatives / "sub-I01"
    _write_status(grid / "status" / "wmh.json", "wmh", "fail", "WMH 与 20区图谱网格不一致")
    _write_status(grid / "status" / "cleanup.json", "cleanup", "pass")
    (grid / "logs").mkdir()
    (grid / "logs" / "wmh.log").write_text("evidence", encoding="utf-8")
    (grid / "wmh").mkdir()
    (grid / "wmh" / "intermediate.nii.gz").write_bytes(b"image")

    _write_status(interrupted / "status" / "registration.json", "registration", "pass", timestamp="2026-08-31T00:00:00+00:00")
    _write_status(interrupted / "status" / "wmh.json", "wmh", "fail", "exit=-15", "2026-08-31T01:00:00+00:00")
    _write_status(interrupted / "status" / "qc.json", "qc", "fail", "dependency failed", "2026-08-31T01:01:00+00:00")
    (interrupted / "registration").mkdir()
    (interrupted / "registration" / "valid.nii.gz").write_bytes(b"keep")

    command = [sys.executable, str(project_root / "scripts" / "prepare_ws1_v1_0_9_rerun.py"), "--project-root", str(root)]
    dry_run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    assert "DRY-RUN" in dry_run.stdout
    assert (grid / "status" / "wmh.json").is_file()

    applied = subprocess.run(command + ["--apply"], capture_output=True, text=True, check=False)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    assert not any(grid.iterdir())
    assert (interrupted / "status" / "registration.json").is_file()
    assert not (interrupted / "status" / "wmh.json").exists()
    assert not (interrupted / "status" / "qc.json").exists()
    assert (interrupted / "registration" / "valid.nii.gz").is_file()
    pointer = root / "logs" / "ws1_v1.0.9_rerun_manifest.path"
    manifest = json.loads(Path(pointer.read_text(encoding="utf-8").strip()).read_text(encoding="utf-8"))
    assert manifest["grid_failure_participants"] == ["1303105005"]
    assert manifest["representative_participants"] == ["1303105005"]
    assert manifest["interrupted_participants"] == ["I01"]


def test_ws1_scheduler_runs_priority_phases_without_audit(project_root: Path) -> None:
    script = (project_root / "scripts" / "finish_ws1_v1_0_9.sh").read_text(encoding="utf-8")
    order = [
        script.index('run_phase "代表病例"'),
        script.index('run_phase "WMH网格失败"'),
        script.index('run_phase "TERM中断"'),
        script.index('run_phase "未处理"'),
    ]
    assert order == sorted(order)
    assert 'run_phase "代表病例" "${plan_dir}/representative.targets" 3' in script
    assert "BATCH_SIZE:-200" in (project_root / "scripts" / "finish_backlog_then_all.sh").read_text(encoding="utf-8")
    assert "rule audit:" not in script
    assert "audit --config-file" not in script
