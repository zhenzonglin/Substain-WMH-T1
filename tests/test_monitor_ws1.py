import importlib.util
import json
import sys
from pathlib import Path


def _load_script(project_root: Path, name: str):
    path = project_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _status(root: Path, participant: str, stage: str, status: str, timestamp: str, runtime=None) -> None:
    path = root / "sub-{}".format(participant) / "status" / "{}.json".format(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    details = {}
    if runtime is not None:
        details["runtime"] = runtime
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "participant_id": participant,
                "stage": stage,
                "status": status,
                "details": details,
            }
        ),
        encoding="utf-8",
    )


def test_progress_counts_running_and_reads_latest_runtime(project_root: Path, tmp_path: Path) -> None:
    monitor = _load_script(project_root, "monitor_ws1.py")
    derivatives = tmp_path / "derivatives"
    _status(
        derivatives,
        "A",
        "wmh",
        "pass",
        "2026-08-31T01:00:00+00:00",
        {
            "started_at_utc": "2026-08-31T00:58:00+00:00",
            "finished_at_utc": "2026-08-31T01:00:00+00:00",
            "duration_seconds": 120.0,
        },
    )
    _status(derivatives, "B", "wmh", "fail", "2026-08-31T02:00:00+00:00")

    rows = monitor.summarize_progress(derivatives, ["A", "B", "C"], {("wmh", "B")})
    wmh = next(row for row in rows if row["stage"] == "wmh")

    assert (wmh["passed"], wmh["failed"], wmh["running"], wmh["pending"]) == (1, 0, 1, 1)
    assert wmh["latest_participant"] == "B"
    assert wmh["latest_duration"] is None
    assert monitor.format_duration(wmh["latest_duration"]) == "NA"


def test_process_argument_parser_recognizes_hyphenated_stage(project_root: Path) -> None:
    monitor = _load_script(project_root, "monitor_ws1.py")
    process = monitor.ProcessInfo(
        pid=10,
        ppid=1,
        pgid=10,
        elapsed_seconds=30,
        cpu_percent=1.0,
        memory_percent=1.0,
        rss_kb=1024,
        state="S",
        args="python -m substain_features.cli stage wmh-seg --participant-id A01",
    )
    assert monitor.parse_job(process) == ("wmh_seg", "A01")
