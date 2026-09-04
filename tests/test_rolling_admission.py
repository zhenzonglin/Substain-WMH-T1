import json
import subprocess
import sys
import time
from pathlib import Path


def _write_cleanup(derivatives: Path, participant_id: str, status: str = "pass") -> None:
    status_dir = derivatives / "sub-{}".format(participant_id) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "cleanup.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )


def _wait_until_running(process: subprocess.Popen, delay: float = 0.15) -> None:
    time.sleep(delay)
    assert process.poll() is None


def test_rolling_window_refills_one_slot_per_finished_subject(
    project_root: Path, tmp_path: Path
) -> None:
    script = project_root / "scripts" / "rolling_admission.py"
    order_file = tmp_path / "order.txt"
    token_dir = tmp_path / "tokens"
    derivatives = tmp_path / "derivatives"
    order_file.write_text("A\nB\nC\nD\n", encoding="utf-8")

    initialized = subprocess.run(
        [
            sys.executable,
            str(script),
            "initialize",
            "--order-file",
            str(order_file),
            "--token-dir",
            str(token_dir),
            "--window",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    assert sorted(path.stem for path in token_dir.glob("*.json")) == ["A", "B"]

    third = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "wait",
            "--order-file",
            str(order_file),
            "--derivatives",
            str(derivatives),
            "--participant-id",
            "C",
            "--output",
            str(token_dir / "C.json"),
            "--window",
            "2",
            "--poll-seconds",
            "0.01",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_until_running(third)
    _write_cleanup(derivatives, "A")
    stdout, stderr = third.communicate(timeout=5)
    assert third.returncode == 0, stdout + stderr
    assert (token_dir / "C.json").is_file()

    fourth = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "wait",
            "--order-file",
            str(order_file),
            "--derivatives",
            str(derivatives),
            "--participant-id",
            "D",
            "--output",
            str(token_dir / "D.json"),
            "--window",
            "2",
            "--poll-seconds",
            "0.01",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_until_running(fourth)
    _write_cleanup(derivatives, "B", status="fail")
    stdout, stderr = fourth.communicate(timeout=5)
    assert fourth.returncode == 0, stdout + stderr
    assert (token_dir / "D.json").is_file()


def test_rolling_queue_rejects_duplicate_participants(project_root: Path, tmp_path: Path) -> None:
    order_file = tmp_path / "order.txt"
    order_file.write_text("A\nA\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "rolling_admission.py"),
            "initialize",
            "--order-file",
            str(order_file),
            "--token-dir",
            str(tmp_path / "tokens"),
            "--window",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "重复participant_id" in completed.stderr


def test_ws1_runner_uses_one_rolling_dag_and_bounded_admission(project_root: Path) -> None:
    runner = (project_root / "scripts" / "finish_ws1_v1_0_9.sh").read_text(
        encoding="utf-8"
    )
    snakefile = (project_root / "workflow" / "Snakefile").read_text(encoding="utf-8")

    assert 'run_rolling "${plan_dir}/rolling.targets"' in runner
    assert '"rolling_window=${window}"' in runner
    assert '"${completion_target}"' in runner
    assert 'run_phase "WMH网格失败"' not in runner
    assert 'run_phase "TERM中断"' not in runner
    assert 'run_phase "未处理"' not in runner

    assert "rule rolling_admission:" in snakefile
    assert "rule rolling_completion:" in snakefile
    assert "admission=ROLLING_ADMISSION_INPUT" in snakefile
    assert "ROLLING_ADMISSION_INPUT = ancient(ROLLING_TOKEN_PATTERN)" in snakefile
    assert "threads: 1" in snakefile
