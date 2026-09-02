import json
import time
from pathlib import Path

from substain_features.status import guarded_stage


def test_guarded_stage_records_runtime_on_success(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"

    assert guarded_stage(status_path, "wmh", "A01", lambda: {"result": 1}) is True

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    runtime = payload["details"]["runtime"]
    assert payload["status"] == "pass"
    assert runtime["started_at_utc"] <= runtime["finished_at_utc"]
    assert runtime["duration_seconds"] >= 0
    assert "peak_rss_mb_self" in runtime
    assert "peak_rss_mb_children" in runtime


def test_guarded_stage_records_runtime_on_failure(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"

    def fail():
        time.sleep(0.01)
        raise RuntimeError("expected")

    assert guarded_stage(status_path, "wmh", "A02", fail) is False

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["details"]["error"] == "expected"
    assert payload["details"]["runtime"]["duration_seconds"] >= 0


def test_old_status_without_runtime_remains_readable(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    payload = {
        "schema_version": "1.0",
        "timestamp_utc": "2026-08-31T00:00:00+00:00",
        "participant_id": "A03",
        "stage": "wmh",
        "status": "pass",
        "details": {},
    }
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = json.loads(status_path.read_text(encoding="utf-8"))
    assert loaded["details"].get("runtime") is None
