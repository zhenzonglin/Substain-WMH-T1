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
        self.resource_patch = patch.object(monitor, "resource_snapshot", side_effect=lambda pids: {
            "pids": sorted(pids), "ls_process_count": len(pids), "ls_cpu_percent": None,
            "ls_rss_mib": None, "gpus": []})
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
        result = monitor.snapshot(self.config)
        self.assertEqual(result["roster_state"], "unknown")
        self.assertIsNone(result["overall"]["total"])
        self.assertTrue(any("duplicate" in error for error in result["errors"]))

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

    def test_missing_roster_preserves_active_runner_and_unknown_counts(self):
        self.participants.unlink()
        process = self.master()
        process.update(state="D", cpu_seconds=34.5, read_bytes=4096, rchar=8192)
        self.processes = [process]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["roster_state"], "not_ready")
        self.assertEqual(result["runner"]["status"], "running")
        self.assertEqual(result["runner"]["state"], "D")
        self.assertEqual(result["runner"]["cpu_seconds"], 34.5)
        self.assertEqual(result["runner"]["read_bytes"], 4096)
        self.assertEqual(result["runner"]["rchar"], 8192)
        self.assertTrue(all(value is None for value in result["overall"].values()))
        self.assertTrue(all(value is None for row in result["stages"].values() for value in row.values()))
        self.assertEqual(result["resources"]["pids"], [50])
        output = monitor.format_snapshot(result)
        self.assertIn("waiting for preparation", output)
        self.assertIn("CPU seconds=34.5", output)
        self.assertNotIn("ALL_SUBJECTS", output)

    def test_missing_roster_exited_unknown_stale_and_zombie_runner(self):
        self.participants.unlink()
        process = self.master()
        self.assertEqual(monitor.snapshot(self.config)["runner"]["status"], "exited")
        self.visible = False
        self.assertEqual(monitor.snapshot(self.config)["runner"]["status"], "unknown")
        self.visible = True
        process["start_ticks"] += 1
        self.processes = [process]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["runner"]["status"], "stale_pid")
        self.assertIsNone(result["runner"]["cpu_seconds"])
        process["start_ticks"] -= 1
        process["state"] = "Z"
        self.assertEqual(monitor.snapshot(self.config)["runner"]["status"], "exited")

    def test_missing_roster_without_runner_does_not_claim_preparing(self):
        self.participants.unlink()
        result = monitor.snapshot(self.config)
        self.assertEqual(result["runner"]["status"], "not_running")
        self.assertIn("does not confirm active preparation", monitor.format_snapshot(result))
        self.assertIsNone(result["overall"]["total"])

    def test_roster_permission_error_is_unknown_and_keeps_runner(self):
        self.processes = [self.master()]
        original = Path.open
        def selective_open(path, *args, **kwargs):
            if path == self.participants:
                raise PermissionError("denied roster")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", selective_open):
            result = monitor.snapshot(self.config)
        self.assertEqual(result["roster_state"], "unknown")
        self.assertEqual(result["runner"]["status"], "running")
        self.assertIsNone(result["overall"]["total"])
        self.assertTrue(any("denied roster" in error for error in result["errors"]))

    def test_bad_roster_keeps_monitor_available(self):
        self.participants.write_text("wrong_column\nP0001\n", encoding="utf-8")
        result = monitor.snapshot(self.config)
        self.assertEqual(result["roster_state"], "unknown")
        self.assertIsNone(result["overall"]["total"])
        self.assertIn("counts: unknown", monitor.format_snapshot(result))

    def test_main_log_tail_is_bounded_and_reports_preparation(self):
        self.participants.unlink()
        log = self.logs / "current_full_run_LS.log"
        log.write_text("x" * 20000 + "\nold\nLS prepare: read Excel, match all image directories and hash inputs\n", encoding="utf-8")
        (self.logs / "full_run_LS.logpath").write_text(str(log), encoding="utf-8")
        self.processes = [self.master()]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["main_log"]["status"], "available")
        self.assertLessEqual(len(result["main_log"]["lines"]), 3)
        self.assertTrue(all(len(line) <= 500 for line in result["main_log"]["lines"]))
        self.assertIn("LS prepare: read Excel", monitor.format_snapshot(result))

    def test_main_log_outside_ls_directory_is_refused(self):
        log = self.root / "private.log"
        log.write_text("must not be read", encoding="utf-8")
        (self.logs / "full_run_LS.logpath").write_text(str(log), encoding="utf-8")
        result = monitor.snapshot(self.config)
        self.assertEqual(result["main_log"]["status"], "unknown")
        self.assertNotIn("must not be read", monitor.format_snapshot(result))

    def test_old_failed_input_report_does_not_override_active_worker(self):
        self.participants.unlink()
        prep = self.root / "preparation_LS"
        prep.mkdir()
        self.config["ls"]["preparation_dir"] = str(prep)
        (prep / "input_report_LS.json").write_text(json.dumps({"status": "fail", "issues": [
            {"severity": "error", "participant_id": "P0005", "message": "missing T1"},
            {"severity": "error", "participant_id": "P0009", "message": "unknown SEX"}]}), encoding="utf-8")
        self.processes = [self.master()]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["runner"]["status"], "running")
        self.assertEqual(result["input_report"]["status"], "fail")
        self.assertEqual(result["input_report"]["error_count"], 2)
        self.assertIsNotNone(result["input_report"]["mtime_utc"])
        self.assertEqual(result["input_report"]["run_attribution"], "unverified")
        output = monitor.format_snapshot(result)
        self.assertIn("P0005: missing T1", output)
        self.assertIn("may predate this worker", output)

    def test_interval_switches_to_stage_table_when_roster_appears(self):
        self.participants.unlink()
        self.processes = [self.master()]
        # Exercise the real CLI loop while avoiding YAML and wall-clock sleeps.
        sleeps = []
        def next_poll(interval):
            sleeps.append(interval)
            if len(sleeps) == 1:
                self.people("P0001")
            else:
                raise KeyboardInterrupt
        with patch.object(monitor, "_configuration", return_value=(self.config, self.config_path)), \
                patch.object(monitor.time, "sleep", side_effect=next_poll), patch("builtins.print") as printed:
            self.assertEqual(monitor.main(["--config", str(self.config_path), "--interval", "1"]), 0)
        output = [call.args[0] for call in printed.call_args_list if call.args]
        self.assertIn("Participant roster: not_ready", output[0])
        self.assertIn("Participant roster: ready", output[1])
        self.assertIn("ALL_SUBJECTS", output[1])

    def test_missing_optional_process_telemetry_remains_unknown(self):
        self.participants.unlink()
        self.processes = [self.master()]
        result = monitor.snapshot(self.config)
        self.assertEqual(result["runner"]["status"], "running")
        for key in ("cpu_seconds", "read_bytes", "rchar"):
            self.assertIsNone(result["runner"][key])

    def test_process_scan_reads_cpu_and_io_without_guessing(self):
        proc = self.root / "proc"
        process = proc / "50"
        process.mkdir(parents=True)
        fields = ["0"] * 20
        fields[0], fields[1], fields[2] = "D", "1", "50"
        fields[11], fields[12], fields[19] = "3400", "50", "1234"
        (process / "stat").write_text("50 (python) " + " ".join(fields), encoding="utf-8")
        (process / "cmdline").write_bytes(b"python\0runner_LS.py\0worker\0")
        (process / "io").write_text("read_bytes: 4096\nrchar: 8192\n", encoding="utf-8")
        self.process_patch.stop()
        with patch.object(monitor, "Path", return_value=proc), patch.object(monitor.os, "sysconf", return_value=100):
            processes, visible, errors = monitor.read_processes()
        self.assertTrue(visible)
        self.assertEqual(errors, [])
        self.assertEqual(processes[0]["state"], "D")
        self.assertEqual(processes[0]["cpu_seconds"], 34.5)
        self.assertEqual(processes[0]["read_bytes"], 4096)
        self.assertEqual(processes[0]["rchar"], 8192)

    def test_optional_proc_io_denied_does_not_hide_process(self):
        proc = self.root / "proc"
        process = proc / "50"
        process.mkdir(parents=True)
        fields = ["0"] * 20
        fields[0], fields[1], fields[2], fields[19] = "S", "1", "50", "1234"
        (process / "stat").write_text("50 (python) " + " ".join(fields), encoding="utf-8")
        (process / "cmdline").write_bytes(b"python\0runner_LS.py\0worker\0")
        original = Path.read_text
        def selective_read(path, *args, **kwargs):
            if path == process / "io":
                raise PermissionError("io denied")
            return original(path, *args, **kwargs)
        self.process_patch.stop()
        with patch.object(monitor, "Path", return_value=proc), patch.object(Path, "read_text", selective_read):
            processes, visible, errors = monitor.read_processes()
        self.assertTrue(visible)
        self.assertEqual(errors, [])
        self.assertEqual(processes[0]["pid"], 50)
        self.assertIsNone(processes[0]["read_bytes"])
        self.assertIsNone(processes[0]["rchar"])


if __name__ == "__main__":
    unittest.main()
