#!/usr/bin/env python3
"""Read-only LS progress monitor; Python 3.8+, independent of GB validators."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Set, Tuple, Union


STAGES = ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc", "cleanup")
COUNTERS = ("pass", "fail", "running", "pending", "unknown")
CONFIG_OPTIONS = ("--config", "--config-file", "--configfile")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _absolute(value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("{} must be an absolute path".format(label))
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("{} must be an absolute path: {}".format(label, path))
    return path.resolve()


def _configuration(config: Union[str, Path, Mapping]) -> Tuple[dict, Path]:
    if isinstance(config, (str, Path)):
        import yaml
        path = _absolute(config, "--config")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = dict(config)
        value = data.get("_config_path") or data.get("ls", {}).get("config_path")
        path = _absolute(value, "config['_config_path'] or config['ls']['config_path']")
    if not isinstance(data, dict) or not isinstance(data.get("ls"), dict):
        raise ValueError("LS configuration must contain an ls mapping")
    if data.get("input", {}).get("cohort", "LS") != "LS":
        raise ValueError("Refusing a non-LS cohort configuration")
    return data, path


def _roster(path: Path) -> List[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "participant_id" not in reader.fieldnames:
            raise ValueError("participants_LS.tsv is missing participant_id")
        rows = list(reader)
    people = [(row.get("participant_id") or "").strip() for row in rows]
    if any(not SAFE_ID.fullmatch(person) for person in people):
        raise ValueError("Participant roster contains an empty or unsafe ID; no rows were dropped")
    if len(set(person.casefold() for person in people)) != len(people):
        raise ValueError("Participant roster contains duplicate IDs; no rows were dropped")
    if any(row.get("site_id") not in (None, "", "LS") for row in rows):
        raise ValueError("Participant roster contains a non-LS site_id")
    return people


def read_processes() -> Tuple[List[dict], bool, List[str]]:
    """Read exact argv and Linux start ticks, without shell/cwd heuristics."""
    result, errors = [], []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        return [], False, ["Process visibility unavailable: {}".format(exc)]
    complete = True
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            argv = [os.fsdecode(value) for value in (entry / "cmdline").read_bytes().split(b"\0") if value]
            result.append({"pid": int(entry.name), "ppid": int(fields[1]), "pgid": int(fields[2]),
                           "start_ticks": int(fields[19]), "state": fields[0], "argv": argv})
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process may exit between the directory scan and reading stat.
        except (OSError, ValueError, IndexError):
            complete = False
    if not complete:
        errors.append("Some process records could not be read; unobserved unfinished stages are unknown")
    return result, complete, errors


def _option(argv: Sequence[str], names: Sequence[str]) -> List[str]:
    values = []
    for index, token in enumerate(argv):
        if token in names:
            values.append(argv[index + 1] if index + 1 < len(argv) else "")
        else:
            for name in names:
                if token.startswith(name + "="):
                    values.append(token[len(name) + 1:])
    return values


def _config_context(argv: Sequence[str], config_path: Path) -> Optional[bool]:
    values = _option(argv, CONFIG_OPTIONS)
    if not values:
        return None
    return all(Path(value).is_absolute() and Path(value).resolve() == config_path for value in values)


def _runner_state(log_dir: Path, config_path: Path, processes: Mapping[int, dict]) -> Tuple[dict, List[str]]:
    state = {"status": "not_running", "pid": None, "pgid": None}
    errors = []
    metadata_path = log_dir / "full_run_LS.json"
    pid_path = log_dir / "full_run_LS.pid"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        try:
            pid_path.stat()
        except FileNotFoundError:
            return state, errors
        except OSError as exc:
            return dict(state, status="unknown"), ["Cannot inspect LS PID pointer: {}".format(exc)]
        return dict(state, status="unknown"), ["LS PID pointer exists without identity JSON"]
    except (OSError, ValueError, UnicodeError) as exc:
        return dict(state, status="unknown"), ["Cannot read LS runner identity: {}".format(exc)]
    try:
        pid = int(metadata["pid"])
        pgid = int(metadata["pgid"])
        start_ticks = int(metadata["start_ticks"])
        runner = _absolute(metadata["runner"], "runner")
        if (pid <= 0 or pgid <= 0 or start_ticks < 0 or
                _absolute(metadata["config"], "runner config") != config_path or runner.name != "runner_LS.py"):
            raise ValueError("invalid or foreign LS runner identity")
        try:
            recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            recorded_pid = pid
        if recorded_pid != pid:
            raise ValueError("LS PID and JSON pointers disagree")
    except (KeyError, TypeError, OSError, ValueError, UnicodeError) as exc:
        return dict(state, status="unknown"), ["Invalid LS runner identity: {}".format(exc)]
    state.update(pid=pid, pgid=pgid)
    process = processes.get(pid)
    if process is None:
        return dict(state, status="exited"), errors
    argv = process["argv"]
    runner_matches = any(Path(token).is_absolute() and Path(token).resolve() == runner for token in argv)
    if (process["start_ticks"] != start_ticks or process["pgid"] != pgid or
            _config_context(argv, config_path) is not True or not runner_matches or "worker" not in argv):
        return dict(state, status="stale_pid"), ["Recorded PID has a different identity; it is not counted as LS"]
    return dict(state, status="running"), errors


def _active_jobs(people: Sequence[str], config_path: Path, processes: Sequence[dict], runner: dict
                 ) -> Tuple[Set[Tuple[str, str]], Set[int]]:
    live = {process["pid"]: process for process in processes if process["state"] not in ("Z", "X")}
    contexts = {pid: _config_context(process["argv"], config_path) for pid, process in live.items()}
    related = set()  # type: Set[int]
    pgid = None
    if runner["status"] == "running":
        related.add(runner["pid"])
        pgid = runner["pgid"]
    while True:
        children = {pid for pid, process in live.items()
                    if process["ppid"] in related and contexts[pid] is not False}
        if children.issubset(related):
            break
        related.update(children)
    known = set(people)
    jobs = set()  # type: Set[Tuple[str, str]]
    for pid, process in live.items():
        argv = process["argv"]
        context = contexts[pid]
        if context is False or not (context is True or pid in related or (pgid and process["pgid"] == pgid)):
            continue
        related.add(pid)
        participant = _option(argv, ("--participant-id",))
        if len(participant) != 1 or participant[0] not in known:
            continue
        for index, token in enumerate(argv[:-1]):
            stage = argv[index + 1].replace("-", "_")
            if token == "stage" and stage in STAGES:
                jobs.add((stage, participant[0]))
                break
    return jobs, related


def _status(path: Path, person: str, stage: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError, UnicodeError) as exc:
        return "unknown", "{}/{}: {}".format(person, stage, exc)
    if (not isinstance(payload, dict) or payload.get("participant_id") != person or
            payload.get("stage") != stage or payload.get("status") not in ("pass", "fail")):
        return "unknown", "{}/{}: status, stage, or participant ID mismatch".format(person, stage)
    return payload["status"], None


def resource_snapshot(pids: Set[int]) -> dict:
    """Optional resource observations; GPU values are host-wide, never claimed as LS use."""
    resources = {"ls_process_count": len(pids), "ls_cpu_percent": None, "ls_rss_mib": None, "gpus": []}
    if pids:
        try:
            output = subprocess.run(["ps", "-p", ",".join(str(pid) for pid in sorted(pids)),
                                     "-o", "%cpu=,rss="], capture_output=True, text=True,
                                    check=False, timeout=5)
            rows = [line.split() for line in output.stdout.splitlines() if line.strip()]
            if output.returncode == 0 and rows:
                resources["ls_cpu_percent"] = round(sum(float(row[0]) for row in rows), 1)
                resources["ls_rss_mib"] = round(sum(int(row[1]) for row in rows) / 1024.0, 1)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                check=False, timeout=5)
        if output.returncode == 0:
            for row in csv.reader(output.stdout.splitlines()):
                if len(row) == 4:
                    resources["gpus"].append(dict(zip(("index", "utilization_percent", "memory_used_mib", "memory_total_mib"),
                                                     (value.strip() for value in row))))
    except (OSError, subprocess.SubprocessError):
        pass
    return resources


def snapshot(config: Union[str, Path, Mapping]) -> dict:
    """Return exclusive stage/subject counts; incomplete includes fail/running/pending/unknown.

    Dict callers supply ``_config_path`` or ``ls.config_path``. Paths in an LS
    config are absolute. A currently running retry supersedes an old pass/fail;
    unreadable or mismatched status records always remain unknown.
    """
    data, config_path = _configuration(config)
    participants = _absolute(data.get("participants"), "participants")
    derivatives = _absolute(data.get("derivatives"), "derivatives")
    log_dir = _absolute(data["ls"].get("log_dir"), "ls.log_dir")
    people = _roster(participants)
    processes, visible, process_errors = read_processes()
    live = {process["pid"]: process for process in processes if process["state"] not in ("Z", "X")}
    runner, runner_errors = _runner_state(log_dir, config_path, live)
    if not visible and runner["status"] == "exited":
        runner["status"] = "unknown"
    jobs, pids = _active_jobs(people, config_path, processes, runner)
    errors = process_errors + runner_errors
    stages = {stage: dict({key: 0 for key in COUNTERS}, total=len(people)) for stage in STAGES}
    overall = dict({key: 0 for key in COUNTERS}, total=len(people), incomplete=0)
    for person in people:
        states = []
        for stage in STAGES:
            state, error = _status(derivatives / ("sub-" + person) / "status" / (stage + ".json"), person, stage)
            if error:
                errors.append(error)
            if state != "unknown" and (stage, person) in jobs:
                state = "running"
            if state is None:
                state = "pending" if visible else "unknown"
            stages[stage][state] += 1
            states.append(state)
        category = ("unknown" if "unknown" in states else "running" if "running" in states else
                    "fail" if "fail" in states else "pass" if all(state == "pass" for state in states) else "pending")
        overall[category] += 1
    overall["incomplete"] = len(people) - overall["pass"]
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": str(config_path), "participants": str(participants), "derivatives": str(derivatives),
            "stages": stages, "overall": overall, "runner": runner, "process_visibility_complete": visible,
            "active_jobs": [{"stage": stage, "participant_id": person} for stage, person in sorted(jobs)],
            "errors": errors, "resources": resource_snapshot(pids)}


def format_snapshot(result: Mapping) -> str:
    lines = ["LS | {}".format(result["timestamp_utc"]), "Results: {}".format(result["derivatives"]),
             "Runner: {} | PID={}".format(result["runner"]["status"], result["runner"]["pid"]),
             "{:<14} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}".format("stage", "total", "pass", "fail", "running", "pending", "unknown")]
    for stage in STAGES + ("ALL_SUBJECTS",):
        row = result["overall"] if stage == "ALL_SUBJECTS" else result["stages"][stage]
        lines.append("{:<14} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}".format(stage, row["total"], *(row[key] for key in COUNTERS)))
    lines.append("Incomplete subjects: {} (all subjects minus 8-stage pass; includes failed subjects)".format(result["overall"]["incomplete"]))
    lines.append("Counts are exclusive; active retries supersede old pass/fail, unreadable states stay unknown.")
    resources = result["resources"]
    lines.append("LS processes={} | CPU={} % (ps lifetime average) | RSS={} MiB".format(
        resources["ls_process_count"], resources["ls_cpu_percent"], resources["ls_rss_mib"]))
    for gpu in resources["gpus"]:
        lines.append("Host GPU {index}: use={utilization_percent}% memory={memory_used_mib}/{memory_total_mib} MiB".format(**gpu))
    if result["errors"]:
        lines.append("Read/identity warnings: {} (unknown is not pending)".format(len(result["errors"])))
        lines.extend("  " + error for error in result["errors"][:3])
        if len(result["errors"]) > 3:
            lines.append("  Additional warning details are available from snapshot(config)['errors'].")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Absolute path to config_LS.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Print one snapshot (default)")
    mode.add_argument("--interval", type=float, help="Repeat every N seconds; must be positive")
    args = parser.parse_args(argv)
    if args.interval is not None and args.interval <= 0:
        parser.error("--interval must be greater than zero")
    try:
        while True:
            print(format_snapshot(snapshot(args.config)), flush=True)
            if args.interval is None:
                return 0
            time.sleep(args.interval)
            print()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print("LS monitor error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
