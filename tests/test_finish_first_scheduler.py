import json
import os
import re
import subprocess
from pathlib import Path

import pytest


requires_posix_shell = pytest.mark.skipif(
    os.name == "nt",
    reason="调度脚本行为测试需要POSIX Bash；Windows由静态测试覆盖入口和DAG。",
)


def _rule_body(snakefile: str, rule: str) -> str:
    match = re.search(r"^rule {}:\n(?P<body>.*?)(?=^rule |\Z)".format(rule), snakefile, re.MULTILINE | re.DOTALL)
    assert match is not None, "缺少规则: {}".format(rule)
    return match.group("body")


def _test_scheduler(project_root: Path, root: Path, statuses: dict) -> subprocess.CompletedProcess:
    script = root / "scripts" / "finish_backlog_then_all.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        (project_root / "scripts" / "finish_backlog_then_all.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    script.chmod(0o755)

    python = root / "envs" / "core-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8")
    python.chmod(0o755)

    for participant, stage_statuses in statuses.items():
        status_dir = root / "derivatives" / "substain_features" / "sub-{}".format(participant) / "status"
        status_dir.mkdir(parents=True)
        for stage, payload in stage_statuses.items():
            if isinstance(payload, str):
                payload = {"status": payload}
            (status_dir / "{}.json".format(stage)).write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

    return subprocess.run(
        [str(script), "96"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "BACKLOG_LIST_ONLY": "1",
            "BATCH_SIZE": "200",
            "CUDA_VISIBLE_DEVICES": "2",
        },
    )


def test_priorities_strictly_finish_subjects_before_opening_upstream(project_root: Path) -> None:
    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")
    expected = {
        "cleanup": 170,
        "qc": 160,
        "t1": 150,
        "wmh_segmentation": 165,
        "wmh": 140,
        "lesion_processing": 130,
        "registration": 120,
        "skullstrip": 10,
    }
    for rule, priority in expected.items():
        assert "priority: {}".format(priority) in _rule_body(snakefile, rule)


def test_t1_uses_wmh_as_ancient_scheduler_gate(project_root: Path) -> None:
    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")
    t1_rule = _rule_body(snakefile, "t1")
    assert 'wmh=ancient(stage_pattern("wmh"))' in t1_rule
    assert "rules.audit.output" not in t1_rule
    assert "t1_cpu" not in t1_rule
    assert "gpu=0" in t1_rule


def test_cpu_heavy_rules_share_only_the_global_core_pool(project_root: Path) -> None:
    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")
    for rule in ("skullstrip", "registration", "lesion_processing", "wmh", "t1"):
        body = _rule_body(snakefile, rule)
        assert "threads: CPU_HEAVY_THREADS" in body
        assert "gpu=0" in body
        assert "finish_cpu" not in body
        assert "skullstrip_cpu" not in body
        assert "t1_cpu" not in body

    assert "gpu=1 if PROFILE == \"gpu\" else 0" in _rule_body(
        snakefile, "wmh_segmentation"
    )
    assert "finish_cpu" not in snakefile
    assert "skullstrip_cpu" not in snakefile
    assert "t1_cpu" not in snakefile
    assert 'config.get("cpu_threads_per_job", EXECUTION_CONFIG.get("cpu_threads_per_job", 8))' in snakefile


def test_full_gpu_entry_uses_strict_scheduler_but_single_subject_stays_direct(project_root: Path) -> None:
    step = (project_root / "scripts" / "steps" / "02_features.sh").read_text(encoding="utf-8")
    strict_condition = '[[ "${participant}" == "all" && "${profile}" == "gpu" ]]'
    strict_exec = 'exec "${root}/scripts/finish_backlog_then_all.sh" "${cores}"'
    direct_cli = "-m substain_features.cli run"
    assert 'cores="${3:-96}"' in step
    assert strict_condition in step
    assert strict_exec in step
    assert step.index(strict_condition) < step.index(strict_exec) < step.index(direct_cli)


def test_backlog_and_new_subjects_both_use_cleanup_bounded_waves(project_root: Path) -> None:
    script = (project_root / "scripts" / "finish_backlog_then_all.sh").read_text(encoding="utf-8")
    backlog_list = script.index('mapfile -t backlog_targets < "${targets_file}"')
    backlog_loop = script.index("for ((offset = 0, wave = 1; offset < backlog_count;")
    remaining_list = script.index('mapfile -t remaining_targets < "${remaining_targets_file}"')
    remaining_loop = script.index("for ((offset = 0, wave = 1; offset < remaining_count;")
    aggregate = script.index("-m substain_features.cli stage aggregate")
    assert backlog_list < backlog_loop < remaining_list < remaining_loop < aggregate
    assert 'batch_size="${BATCH_SIZE:-200}"' in script
    assert 'backlog_wave_targets=("${backlog_targets[@]:offset:batch_size}")' in script
    assert 'remaining_wave_targets=("${remaining_targets[@]:offset:batch_size}")' in script
    assert 'if [[ ! -f "${target}" ]]' in script
    assert "停止开放下一波" in script
    assert 'exec "${root}/run_pipeline.sh"' not in script


@requires_posix_shell
def test_strict_scheduler_uses_only_global_cores_and_one_gpu_token(
    project_root: Path, tmp_path: Path
) -> None:
    completed = _test_scheduler(
        project_root,
        tmp_path / "project",
        {"A": {"skullstrip": "pass"}},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "严格调度资源: cores=96, heavy_threads=8, cpu_custom_caps=none, gpu=1, gpu_device=2"
        in completed.stdout
    )
    script = (project_root / "scripts" / "finish_backlog_then_all.sh").read_text(
        encoding="utf-8"
    )
    assert '"gpu=1"' in script
    assert '"gpu_devices=${gpu_device}"' in script
    assert '"cpu_threads_per_job=${cpu_threads}"' in script
    assert "finish_cpu" not in script
    assert "skullstrip_cpu" not in script
    assert "t1_cpu" not in script


@requires_posix_shell
def test_strict_scheduler_rejects_multiple_visible_gpus(
    project_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    script = root / "scripts" / "finish_backlog_then_all.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        (project_root / "scripts" / "finish_backlog_then_all.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    script.chmod(0o755)
    completed = subprocess.run(
        [str(script), "96"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BACKLOG_LIST_ONLY": "1", "CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert completed.returncode == 2
    assert "只接受一张GPU编号" in completed.stderr


@requires_posix_shell
def test_backlog_orders_closest_subjects_and_preserves_real_failures(
    project_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    statuses = {
        "A": {"skullstrip": "pass"},
        "B": {"t1": "pass"},
        "C": {"registration": {"status": "fail", "details": {"error": "real failure"}}},
        "D": {"skullstrip": "pass", "cleanup": "pass"},
        "E": {"skullstrip": "pass", "wmh_seg": "pass", "registration": "pass"},
        "F": {"skullstrip": {"status": "fail", "details": {"error": "SIGTERM"}}},
    }
    completed = _test_scheduler(project_root, root, statuses)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    participants = (root / "logs" / "backlog_participants.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert participants == ["E", "B", "C", "A"]
    assert (root / "derivatives" / "substain_features" / "sub-C" / "status" / "registration.json").is_file()
    assert not (root / "derivatives" / "substain_features" / "sub-F" / "status" / "skullstrip.json").exists()
    assert list((root / "archive").rglob("sub-F/status/skullstrip.json"))


@requires_posix_shell
def test_450_backlog_subjects_are_planned_as_200_200_50(
    project_root: Path, tmp_path: Path
) -> None:
    statuses = {
        "P{:03d}".format(index): {"skullstrip": "pass"}
        for index in range(450)
    }
    completed = _test_scheduler(project_root, tmp_path / "project", statuses)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "积压严格波次: batch_size=200, backlog=450, waves=3" in completed.stdout
    assert "计划积压波次 1/3: 200例" in completed.stdout
    assert "计划积压波次 2/3: 200例" in completed.stdout
    assert "计划积压波次 3/3: 50例" in completed.stdout
