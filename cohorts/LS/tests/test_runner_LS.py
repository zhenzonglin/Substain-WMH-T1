"""Linux orchestration regressions; exercise control flow without model inference."""

import csv
import json
from pathlib import Path
import shutil
import types

import pytest
import yaml

pytest.importorskip("fcntl", reason="The workstation runner intentionally requires Linux flock")

from test_overlay_LS import build_overlay, fixture_project, load_file, PACKAGE

runner = load_file("test_runner_LS_module", PACKAGE / "runner_LS.py")


@pytest.fixture(scope="module")
def gb18_baseline(tmp_path_factory):
    root, _, _ = fixture_project(tmp_path_factory.mktemp("gb18_runner_baseline"), version="gb18")
    return root


@pytest.fixture
def project(tmp_path, gb18_baseline):
    root = tmp_path / "Substain_GB"
    shutil.copytree(str(gb18_baseline), str(root), ignore=shutil.ignore_patterns("__pycache__"))
    input_root = tmp_path / "LS"
    input_root.mkdir()
    output = tmp_path / "Sustain_LS" / "derivatives"
    output.mkdir(parents=True)
    cfg = runner.config_for(root, input_root, output)
    config_path = root / "config" / "config_LS.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for key in ("log_dir", "runtime_dir", "preparation_dir"):
        Path(cfg["ls"][key]).mkdir(parents=True, exist_ok=True)
    (root / "scripts/cohort_LS").mkdir(parents=True)
    return root, cfg, config_path


def write_roster(cfg, people=("P0001",)):
    rows = []
    source = Path(cfg["ls"]["input_root"])
    mask = Path(cfg["ls"]["preparation_dir"]) / "healthy_zero_LS.nii.gz"
    mask.write_bytes(b"test-zero-mask")
    for person in people:
        directory = source / person
        directory.mkdir(exist_ok=True)
        t1 = directory / (person + "_T1w.nii.gz")
        flair = directory / (person + "_T2wFLAIR.nii.gz")
        t1.write_bytes(b"fixture-t1")
        flair.write_bytes(b"fixture-flair")
        rows.append(dict(participant_id=person, age="50.3", sex="male", site_id="LS",
                         t1w=str(t1), flair=str(flair), lesion_mask=str(mask)))
    with Path(cfg["participants"]).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("participant_id", "age", "sex", "site_id", "t1w", "flair", "lesion_mask"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_state(cfg, person, stage="cleanup", status="pass"):
    target = Path(cfg["derivatives"]) / ("sub-" + person) / "status" / (stage + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(participant_id=person, stage=stage, status=status)), encoding="utf-8")


def frozen_identity(cfg, config_path):
    write_roster(cfg)
    build_overlay(Path(cfg["project_root"]), config_path, Path(cfg["ls"]["runtime_dir"]))
    Path(cfg["ls"]["clinical_file"]).write_bytes(b"test-clinical-workbook")
    manifest = Path(cfg["ls"]["resource_manifest_path"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("relative_path\tsize_bytes\tsha256\n", encoding="utf-8")
    log_dir = Path(cfg["ls"]["log_dir"])
    (log_dir / "environment_LS.json").write_text("{}\n", encoding="utf-8")
    identity = runner.capture_identity(cfg, config_path)
    (log_dir / "run_manifest_LS.json").write_text(runner.canonical(identity), encoding="utf-8")
    return identity


def test_preserves_virtual_environment_python_symlinks(project):
    root, cfg, _ = project
    base_path = root / "config/config.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    target = root / "system-python"
    target.write_bytes(b"system interpreter stand-in")
    expected = {}
    for name in ("core", "wmh", "t1"):
        path = root / "envs" / name / "bin/python"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
        base["execution"][name + "_python"] = str(path.relative_to(root))
        expected[name + "_python"] = str(path)
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    current = runner.config_for(root, Path(cfg["ls"]["input_root"]), Path(cfg["derivatives"]))
    for key, value in expected.items():
        assert current["execution"][key] == value
        assert Path(current["execution"][key]).is_symlink()
        assert current["execution"][key] != str(target)


def test_inherits_gb18_resource_policy_and_physical_gpu_lock(project):
    root, cfg, config_path = project
    base = yaml.safe_load((root / "config/config.yaml").read_text(encoding="utf-8"))
    for key in ("cpu_cores", "cpu_threads_per_job", "registration_threads", "max_compute_jobs",
                "max_parallel_jobs", "gpu_devices", "gpu_slots_per_device"):
        assert cfg["execution"][key] == base["execution"][key]
    assert cfg["execution"]["cpu_cores"] == 64
    assert cfg["execution"]["max_compute_jobs"] == 18
    assert cfg["execution"]["max_parallel_jobs"] == 200
    assert cfg["execution"]["gpu_lock_dir"] == str(runner.absolute(root, base["execution"]["gpu_lock_dir"]))
    command = runner.snakemake_command(cfg, config_path, ["pilot-target"], (root / "order_LS.txt", root / "tokens_LS"))
    assert command[command.index("--cores") + 1] == "64"
    assert all(value in command for value in ("compute_jobs=18", "gpu=4", "wmh_exclusive=2",
                                               "rolling_window=200", "gpu_devices=0,1", "gpu_slots_per_device=2"))
    assert command.index("pilot-target") < command.index("--resources") < command.index("--config")


def test_unchanged_identity_is_reused_but_changed_image_is_rejected(project):
    _, cfg, config_path = project
    identity = frozen_identity(cfg, config_path)
    assert runner.check_identity(cfg, config_path)
    assert runner.capture_identity(cfg, config_path) == identity
    image = Path(cfg["ls"]["input_root"]) / "P0001/P0001_T1w.nii.gz"
    image.write_bytes(b"changed-image")
    with pytest.raises(RuntimeError, match="LS run identity changed"):
        runner.check_identity(cfg, config_path)
    assert runner.read_json(Path(cfg["ls"]["log_dir"]) / "run_manifest_LS.json") == identity


@pytest.mark.parametrize("relative", ("P0001/second_T1w.nii.gz", "P9999/P9999_T2wFLAIR.nii.gz"))
def test_new_candidate_or_participant_invalidates_resume_identity(project, relative):
    _, cfg, config_path = project
    identity = frozen_identity(cfg, config_path)
    extra = Path(cfg["ls"]["input_root"]) / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"new-image")
    with pytest.raises(RuntimeError, match="LS image-directory inventory"):
        runner.check_identity(cfg, config_path)
    assert runner.read_json(Path(cfg["ls"]["log_dir"]) / "run_manifest_LS.json") == identity


def test_operation_lock_excludes_other_writer_and_releases(tmp_path):
    path = tmp_path / "logs/LS/operation_LS.lock"
    with runner.locked(path):
        with pytest.raises(RuntimeError, match="Another LS operation"):
            with runner.locked(path):
                pytest.fail("A second writer entered the lock")
    with runner.locked(path):
        pass


@pytest.mark.parametrize("action", ("install", "prepare", "first", "resume", "verify", "export", "qc"))
def test_mutating_entrypoints_share_operation_lock(project, monkeypatch, action):
    root, cfg, config_path = project
    monkeypatch.setattr(runner, "require_idle", lambda *args, **kwargs: pytest.fail("Writer bypassed the operation lock"))
    with runner.locked(Path(cfg["ls"]["log_dir"]) / "operation_LS.lock"):
        result = runner.main([action, "--root", str(root), "--config", str(config_path)])
    assert result == 1


def test_queue_targets_ls_completion_and_preserves_completed_subject(project, monkeypatch):
    _, cfg, config_path = project
    write_roster(cfg, ("P0001", "P0005"))
    write_state(cfg, "P0001")
    completed = Path(cfg["derivatives"]) / "sub-P0001/status/cleanup.json"
    before = (completed.read_bytes(), completed.stat().st_mtime_ns)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if "snakemake" in command:
            write_state(cfg, "P0005")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", execute)
    runner.run_queue(cfg, config_path)
    assert len(calls) == 2
    order = Path(calls[0][calls[0].index("--order-file") + 1])
    assert order.read_text(encoding="utf-8") == "P0005\n"
    token_dir = Path(calls[0][calls[0].index("--token-dir") + 1])
    assert str(token_dir / "all_complete_LS.json") in calls[1]
    assert str(token_dir / "all_complete.json") not in calls[1]
    assert (completed.read_bytes(), completed.stat().st_mtime_ns) == before


def worker_stubs(monkeypatch, events):
    monkeypatch.setattr(runner, "require_idle", lambda *args, **kwargs: events.append("idle"))
    monkeypatch.setattr(runner, "prepare", lambda *args: events.append("prepare"))
    def identity(*args):
        events.append("identity")
        return "identity-hash"
    monkeypatch.setattr(runner, "check_identity", identity)
    monkeypatch.setattr(runner, "ensure_audit", lambda *args: events.append("audit"))
    monkeypatch.setattr(runner, "choose_pilot", lambda *args: "P0001")
    def healthy(*args):
        events.append("healthy")
        return {"participant_id": "P0001", "status": "pass"}
    monkeypatch.setattr(runner, "healthy_check", healthy)
    monkeypatch.setattr(runner, "summarize", lambda *args: events.append("summary"))


def test_completed_resume_never_aggregates_or_changes_primary(project, monkeypatch):
    _, cfg, config_path = project
    write_roster(cfg)
    write_state(cfg, "P0001")
    tables = Path(cfg["derivatives"]) / "tables"
    tables.mkdir(parents=True)
    (tables / "run_summary_LS.json").write_text("{}", encoding="utf-8")
    primary = tables / "features_primary40_LS.tsv"
    primary.write_bytes(b"human-reviewed-final-table\n")
    before = (primary.read_bytes(), primary.stat().st_mtime_ns)
    pilot = Path(cfg["ls"]["log_dir"]) / "pilot_verification_LS.json"
    pilot.write_text(json.dumps({"participant_id": "P0001", "status": "pass", "run_manifest_sha256": "identity-hash"}))
    events = []
    worker_stubs(monkeypatch, events)
    monkeypatch.setattr(runner, "run_queue", lambda *args: pytest.fail("Completed resume queued inference"))
    monkeypatch.setattr(runner, "cli", lambda *args: pytest.fail("Completed resume called aggregate"))
    runner.worker(cfg, config_path, "resume")
    assert events == ["idle", "identity", "audit", "healthy", "summary"]
    assert (primary.read_bytes(), primary.stat().st_mtime_ns) == before


def test_failed_pilot_prevents_full_queue(project, monkeypatch):
    _, cfg, config_path = project
    write_roster(cfg)
    events = []
    worker_stubs(monkeypatch, events)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: events.append("pilot"))
    def unhealthy(*args):
        raise ValueError("Healthy WMH identity failed")
    monkeypatch.setattr(runner, "healthy_check", unhealthy)
    monkeypatch.setattr(runner, "run_queue", lambda *args: pytest.fail("Failed pilot reached full queue"))
    monkeypatch.setattr(runner, "cli", lambda *args: pytest.fail("Failed pilot reached aggregate"))
    with pytest.raises(ValueError, match="Healthy WMH identity failed"):
        runner.worker(cfg, config_path, "first")
    assert events == ["idle", "prepare", "identity", "audit", "pilot"]
    assert not (Path(cfg["ls"]["log_dir"]) / "pilot_verification_LS.json").exists()


def test_first_run_control_flow_audit_pilot_full_aggregate(project, monkeypatch):
    _, cfg, config_path = project
    write_roster(cfg, ("P0001", "P0005"))
    events = []
    worker_stubs(monkeypatch, events)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: events.append("pilot"))
    monkeypatch.setattr(runner, "run_queue", lambda *args: events.append("full"))
    monkeypatch.setattr(runner, "cli", lambda _cfg, _path, command, *args: events.append(command))
    runner.worker(cfg, config_path, "first")
    assert events == ["idle", "prepare", "identity", "audit", "pilot", "healthy", "full", "stage aggregate", "summary"]
    record = runner.read_json(Path(cfg["ls"]["log_dir"]) / "pilot_verification_LS.json")
    assert record["run_manifest_sha256"] == "identity-hash"


def fake_proc(tmp_path, monkeypatch, records):
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid in records:
        (proc / str(pid)).mkdir()
    monkeypatch.setattr(runner, "process_identity", lambda pid, _proc: records[pid])
    return proc


@pytest.mark.parametrize("argv", (
    ["python", "-m", "snakemake", "--configfile", "config/config.yaml"],
    ["bash", "scripts/finish_ws1_v1_0_9.sh"],
    ["python", "-m", "substain_features.cli", "stage", "t1", "--config-file", "config/another_cohort.yaml"],
    ["/some/environment/bin/snakemake", "--configfile", "config/config.yaml"],
))
def test_idle_check_blocks_gb_relative_commands_and_other_analysis(project, tmp_path, monkeypatch, argv):
    root, _, _ = project
    records = {51: {"pid": 51, "state": "S", "argv": argv, "cwd": str(root)}}
    proc = fake_proc(tmp_path, monkeypatch, records)
    with pytest.raises(RuntimeError, match="Existing GB/LS analysis is active"):
        runner.require_idle(root, own_pid=999, proc=proc)


def test_idle_check_ignores_launcher_monitor_and_other_project(project, tmp_path, monkeypatch):
    root, _, config_path = project
    records = {
        51: {"pid": 51, "state": "S", "argv": ["python", str(root / "scripts/cohort_LS/runner_LS.py"), "first", "--config", str(config_path)], "cwd": str(root)},
        52: {"pid": 52, "state": "S", "argv": ["python", "monitor_LS.py"], "cwd": str(root)},
        53: {"pid": 53, "state": "S", "argv": ["python", "-m", "snakemake"], "cwd": str(tmp_path / "other_project")},
        54: {"pid": 54, "state": "Z", "argv": ["python", "-m", "snakemake"], "cwd": str(root)},
        55: {"pid": 55, "state": "S", "argv": ["python", "runner_LS.py", "worker"], "cwd": str(root)},
    }
    proc = fake_proc(tmp_path, monkeypatch, records)
    runner.require_idle(root, own_pid=55, proc=proc)
