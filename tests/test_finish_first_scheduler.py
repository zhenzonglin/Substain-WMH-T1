import os
import re
import subprocess
from pathlib import Path


def _rule_body(snakefile: str, rule: str) -> str:
    match = re.search(r"^rule {}:\n(?P<body>.*?)(?=^rule |\Z)".format(rule), snakefile, re.MULTILINE | re.DOTALL)
    assert match is not None, "缺少规则: {}".format(rule)
    return match.group("body")


def test_priorities_finish_subjects_before_opening_upstream(project_root: Path) -> None:
    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")
    expected = {
        "wmh_segmentation": 130,
        "cleanup": 120,
        "qc": 110,
        "wmh": 100,
        "lesion_processing": 90,
        "registration": 80,
        "skullstrip": 20,
        "t1": 10,
    }
    for rule, priority in expected.items():
        assert "priority: {}".format(priority) in _rule_body(snakefile, rule)


def test_t1_and_upstream_have_independent_concurrency_caps(project_root: Path) -> None:
    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")
    assert "skullstrip_cpu=1" in _rule_body(snakefile, "skullstrip")
    assert "t1_cpu=1" in _rule_body(snakefile, "t1")

    cli = (project_root / "src" / "substain_features" / "cli.py").read_text(encoding="utf-8")
    assert '"skullstrip_cpu={}".format(skullstrip_slots)' in cli
    assert '"t1_cpu={}".format(t1_slots)' in cli


def test_backlog_precedes_bounded_cleanup_waves(project_root: Path) -> None:
    script = (project_root / "scripts" / "finish_backlog_then_all.sh").read_text(encoding="utf-8")
    backlog_run = script.index('mapfile -t targets < "${targets_file}"')
    wave_list = script.index('mapfile -t remaining_targets < "${remaining_targets_file}"')
    wave_run = script.index("for ((offset = 0, wave = 1;")
    aggregate = script.index('exec "${root}/run_pipeline.sh" run all gpu "${cores}"')
    assert backlog_run < wave_list < wave_run < aggregate
    assert 'batch_size="${BATCH_SIZE:-200}"' in script
    assert 'wave_targets=("${remaining_targets[@]:offset:batch_size}")' in script
    assert '"t1_cpu=${t1_slots}"' in script
    assert 'progress_stages = ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc")' in script


def test_backlog_freezes_any_successful_partial_subject(project_root: Path, tmp_path: Path) -> None:
    root = tmp_path / "project"
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

    statuses = {
        "A": {"skullstrip": "pass"},
        "B": {"t1": "pass"},
        "C": {"registration": "fail"},
        "D": {"skullstrip": "pass", "cleanup": "pass"},
    }
    for participant, stage_statuses in statuses.items():
        status_dir = (
            root / "derivatives" / "substain_features" / "sub-{}".format(participant) / "status"
        )
        status_dir.mkdir(parents=True)
        for stage, status in stage_statuses.items():
            (status_dir / "{}.json".format(stage)).write_text(
                '{{"status": "{}"}}'.format(status),
                encoding="utf-8",
            )

    completed = subprocess.run(
        [str(script), "96"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BACKLOG_LIST_ONLY": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    participants = (root / "logs" / "backlog_participants_v1.0.5.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert participants == ["A", "B"]
