"""LS monitor isolation and truthful counting regression tests."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("monitor_LS", Path(__file__).resolve().parents[1] / "monitor_LS.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config_path = self.root / "config_LS.yaml"
        self.participants = self.root / "participants_LS.tsv"
        self.derivatives = self.root / "derivatives"
        self.logs = self.root / "logs_LS"
        self.logs.mkdir()
        self.runner_path = self.root / "runner_LS.py"
        self.config = {"_config_path": str(self.config_path), "participants": str(self.participants),
                       "derivatives": str(self.derivatives), "input": {"cohort": "LS"},
                       "ls": {"log_dir": str(self.logs)}}
        self.people("P0001", "P0005", "P0009", "P0010", "P0011")
        self.processes = []
        self.visible = True
        self.process_patch = patch.object(monitor, "read_processes", side_effect=lambda: (self.processes, self.visible, []))
        self.process_patch.start()
        self.addCleanup(self.process_patch.stop)
        self.resource_patch = patch.object(monitor, "resource_snapshot", side_effect=lambda pids: {"pids": sorted(pids)})
        self.resource_patch.start()
        self.addCleanup(self.resource_patch.stop)

    def people(self, *people):
        self.participants.write_text("participant_id\tsite_id\n" + "".join(person + "\tLS\n" for person in people), encoding="utf-8")

    def status(self, person, stage, status="pass", payload=None):
        path = self.derivatives / ("sub-" + person) / "status" / (stage + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        value = payload if payload is not None else {"participant_id": person, "stage": stage, "status": status}
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def process(self, pid, argv, ppid=1, pgid=50, ticks=1234):
        return {"pid": pid, "ppid": ppid, "pgid": pgid, "start_ticks": ticks, "state": "S", "argv": argv}

    def stage(self, pid, person, stage="t1", config=None, **kwargs):
        return self.process(pid, ["python", "-m", "substain_features", "stage", stage,
                                 "--participant-id", person] + (["--config-file", str(config)] if config else []), **kwargs)

    def master(self, ticks=1234):
        metadata = {"pid": 50, "pgid": 50, "start_ticks": ticks, "config": str(self.config_path),
                    "runner": str(self.runner_path), "mode": "first", "started_at": "2026-10-17T00:00:00Z"}
        (self.logs / "full_run_LS.json").write_text(json.dumps(metadata), encoding="utf-8")
        (self.logs / "full_run_LS.pid").write_text("50", encoding="utf-8")
        return self.process(50, ["python", str(self.runner_path), "worker", "--config", str(self.config_path)], ticks=ticks)

    def test_counts_deduplicate_people_and_process_wrappers(self):
        for stage in monitor.STAGES:
            self.status("P0001", stage)
        self.status("P0005", "t1", "fail")
        self.status("P0005", "wmh", "fail")
        self.status("P0011", "qc", payload={"participant_id": "WRONG", "stage": "qc", "status": "pass"})
        self.processes = [self.master(), self.stage(51, "P0009", ppid=50), self.stage(52, "P0009", ppid=51)]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["overall"], {"total": 5, "pass": 1, "fail": 1, "running": 1, "pending": 1,
                                             "unknown": 1, "incomplete": 4})
        self.assertEqual(result["stages"]["t1"]["running"], 1)
        self.assertEqual(len(result["active_jobs"]), 1)
        for row in result["stages"].values():
            self.assertEqual(sum(row[key] for key in monitor.COUNTERS), row["total"])

    def test_gb_processes_and_results_are_not_counted(self):
        self.people("P0001")
        self.status("c1p001", "t1")
        self.processes = [self.master(), self.stage(51, "P0001", config=self.root / "config_GB.yaml", ppid=50),
                          self.stage(70, "P0001", config=self.root / "config_GB.yaml", pgid=70),
                          self.stage(80, "P0001", pgid=80),
                          self.stage(81, "P0001", config=self.config_path, pgid=81)]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["stages"]["t1"]["running"], 1)
        self.assertEqual(result["resources"]["pids"], [50, 81])
        self.assertEqual(result["overall"]["total"], 1)

    def test_pid_reuse_is_rejected_even_for_identical_command(self):
        self.people("P0001")
        master = self.master(ticks=1234)
        master["start_ticks"] = 9999
        self.processes = [master, self.stage(51, "P0001", ppid=50)]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["runner"]["status"], "stale_pid")
        self.assertEqual(result["stages"]["t1"]["running"], 0)
        self.assertEqual(result["overall"]["pending"], 1)

    def test_pid_reuse_to_gb_is_rejected(self):
        self.people("P0001")
        master = self.master()
        master["argv"] = ["bash", "gb_verify_then_all.sh"]
        self.processes = [master, self.stage(51, "P0001", ppid=50)]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["runner"]["status"], "stale_pid")
        self.assertEqual(result["active_jobs"], [])

    def test_bad_json_and_permission_errors_are_unknown(self):
        self.people("P0001", "P0005")
        path = self.status("P0001", "t1")
        path.write_text("{broken", encoding="utf-8")
        denied = self.derivatives / "sub-P0005" / "status" / "wmh.json"
        original = Path.read_text
        def selective_read(path, *args, **kwargs):
            if path == denied:
                raise PermissionError("denied by test")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", selective_read):
            result = monitor.snapshot(self.config)
        self.assertEqual(result["overall"]["unknown"], 2)
        self.assertEqual(result["stages"]["t1"]["unknown"], 1)
        self.assertEqual(result["stages"]["wmh"]["unknown"], 1)
        self.assertEqual(len(result["errors"]), 2)

    def test_failed_process_visibility_does_not_claim_pending(self):
        self.people("P0001")
        self.visible = False
        self.status("P0001", "skullstrip")
        result = monitor.snapshot(self.config)
        self.assertEqual(result["overall"]["unknown"], 1)
        self.assertEqual(result["stages"]["skullstrip"]["pass"], 1)
        self.assertEqual(result["stages"]["t1"]["pending"], 0)
        self.assertEqual(result["stages"]["t1"]["unknown"], 1)

    def test_invalid_status_is_not_hidden_by_running_job(self):
        self.people("P0001")
        self.status("P0001", "t1", payload={"participant_id": "P0001", "stage": "wmh", "status": "pass"})
        self.processes = [self.stage(51, "P0001", config=self.config_path)]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["stages"]["t1"]["unknown"], 1)
        self.assertEqual(result["stages"]["t1"]["running"], 0)

    def test_duplicate_roster_is_not_silently_deduplicated(self):
        self.people("P0001", "P0001")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            monitor.snapshot(self.config)

    def test_bad_runner_json_is_reported(self):
        self.people("P0001")
        (self.logs / "full_run_LS.json").write_text("[]", encoding="utf-8")
        result = monitor.snapshot(self.config)
        self.assertEqual(result["runner"]["status"], "unknown")
        self.assertTrue(result["errors"])

    def test_running_retry_supersedes_stale_failure(self):
        self.people("P0001")
        self.status("P0001", "wmh_seg", "fail")
        self.processes = [self.stage(51, "P0001", "wmh-seg", config=self.config_path)]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["overall"]["running"], 1)
        self.assertEqual(result["stages"]["wmh_seg"]["fail"], 0)


if __name__ == "__main__":
    unittest.main()
