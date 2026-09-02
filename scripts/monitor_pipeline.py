#!/usr/bin/env python3
"""主进程、活动子任务和阶段进度的只读监测。"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


STAGES = ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc", "cleanup")
STAGE_ARGUMENT = re.compile(r"\bstage\s+(skullstrip|wmh-seg|registration|lesion|wmh|t1|qc|cleanup)\b")
PARTICIPANT_ARGUMENT = re.compile(r"--participant-id(?:=|\s+)([^\s]+)")


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    pgid: int
    elapsed_seconds: int
    cpu_percent: float
    memory_percent: float
    rss_kb: int
    state: str
    args: str


def _run(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout if completed.returncode == 0 else ""


def read_processes() -> List[ProcessInfo]:
    output = _run(["ps", "-eo", "pid=,ppid=,pgid=,etimes=,%cpu=,%mem=,rss=,stat=,args="])
    processes = []
    for line in output.splitlines():
        fields = line.strip().split(None, 8)
        if len(fields) != 9:
            continue
        try:
            processes.append(
                ProcessInfo(
                    pid=int(fields[0]),
                    ppid=int(fields[1]),
                    pgid=int(fields[2]),
                    elapsed_seconds=int(fields[3]),
                    cpu_percent=float(fields[4]),
                    memory_percent=float(fields[5]),
                    rss_kb=int(fields[6]),
                    state=fields[7],
                    args=fields[8],
                )
            )
        except ValueError:
            continue
    return processes


def parse_job(process: ProcessInfo) -> Optional[Tuple[str, str]]:
    stage_match = STAGE_ARGUMENT.search(process.args)
    participant_match = PARTICIPANT_ARGUMENT.search(process.args)
    if not stage_match or not participant_match:
        return None
    stage = stage_match.group(1).replace("-", "_")
    return stage, participant_match.group(1)


def _read_pid(path: Path) -> Optional[int]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def _meminfo() -> Dict[str, int]:
    values: Dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, _, remainder = line.partition(":")
        number = remainder.strip().split()[0]
        if number.isdigit():
            values[key] = int(number)
    return values


def _oom_count() -> int:
    try:
        lines = Path("/proc/vmstat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in lines:
        key, _, value = line.partition(" ")
        if key == "oom_kill" and value.strip().isdigit():
            return int(value.strip())
    return 0


def _gpu_rows() -> List[Dict[str, object]]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "utilization": int(fields[2]),
                    "memory_used_mb": int(fields[3]),
                    "memory_total_mb": int(fields[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _gpu_process_memory() -> Dict[int, int]:
    output = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: Dict[int, int] = {}
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            rows[int(fields[0])] = int(fields[1])
    return rows


def _append_tsv(path: Path, fieldnames: Sequence[str], row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _baseline(path: Path, swap_used_kb: int, oom_count: int) -> Dict[str, int]:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {"swap_used_kb": int(payload["swap_used_kb"]), "oom_count": int(payload["oom_count"])}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    payload = {"swap_used_kb": swap_used_kb, "oom_count": oom_count}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _fmt_gib(kb: int) -> str:
    return "{:.2f}".format(kb / 1024.0 / 1024.0)


def format_duration(value: object) -> str:
    if value is None:
        return "NA"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "NA"
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return "{}:{:02d}:{:02d}".format(hours, minutes, secs)


def main_view(root: Path, pid_file: Path, processes: Sequence[ProcessInfo]) -> str:
    pid = _read_pid(pid_file)
    main_process = next((process for process in processes if process.pid == pid), None)
    pgid = main_process.pgid if main_process else None
    project_processes = [
        process
        for process in processes
        if str(root) in process.args
        and "monitor_pipeline.py" not in process.args
    ]
    project_rss_kb = sum(process.rss_kb for process in project_processes)
    mem = _meminfo()
    total_kb = mem.get("MemTotal", 0)
    available_kb = mem.get("MemAvailable", 0)
    swap_used_kb = max(0, mem.get("SwapTotal", 0) - mem.get("SwapFree", 0))
    oom_count = _oom_count()
    gpus = _gpu_rows()
    baseline = _baseline(root / "logs" / "monitor_baseline.json", swap_used_kb, oom_count)
    orphan = [process for process in project_processes if pgid is not None and process.pgid != pgid]
    warnings = []
    if total_kb and available_kb / total_kb < 0.10:
        warnings.append("MemAvailable低于10%")
    if total_kb and project_rss_kb / total_kb > 0.80:
        warnings.append("项目RSS超过总RAM的80%")
    if swap_used_kb - baseline["swap_used_kb"] >= 2 * 1024 * 1024:
        warnings.append("swap较监测基线增加至少2 GiB")
    if oom_count > baseline["oom_count"]:
        warnings.append("OOM kill计数增加")
    if orphan:
        warnings.append("发现{}个不属于主PGID的项目进程".format(len(orphan)))
    for gpu in gpus:
        if int(gpu["utilization"]) < 5 and int(gpu["memory_used_mb"]) > 1024:
            warnings.append("GPU{}空闲但占用显存超过1 GiB".format(gpu["index"]))

    timestamp = datetime.now(timezone.utc).isoformat()
    _append_tsv(
        root / "logs" / "resource_history.tsv",
        (
            "timestamp_utc", "main_pid", "main_alive", "pgid", "project_processes",
            "project_rss_kb", "mem_total_kb", "mem_available_kb", "swap_used_kb",
            "oom_kill_count", "gpu_memory_used_mb", "gpu_utilization_percent", "warning_count",
        ),
        {
            "timestamp_utc": timestamp,
            "main_pid": pid or "",
            "main_alive": bool(main_process),
            "pgid": pgid or "",
            "project_processes": len(project_processes),
            "project_rss_kb": project_rss_kb,
            "mem_total_kb": total_kb,
            "mem_available_kb": available_kb,
            "swap_used_kb": swap_used_kb,
            "oom_kill_count": oom_count,
            "gpu_memory_used_mb": sum(int(gpu["memory_used_mb"]) for gpu in gpus),
            "gpu_utilization_percent": max([int(gpu["utilization"]) for gpu in gpus] or [0]),
            "warning_count": len(warnings),
        },
    )
    lines = [timestamp, "主进程"]
    if main_process:
        lines.append(
            "PID={} PGID={} elapsed={} CPU={:.1f}% RSS={}GiB state={}".format(
                main_process.pid,
                main_process.pgid,
                format_duration(main_process.elapsed_seconds),
                main_process.cpu_percent,
                _fmt_gib(main_process.rss_kb),
                main_process.state,
            )
        )
    else:
        lines.append("未运行或PID文件无效: {}".format(pid_file))
    lines.append(
        "项目进程={} 项目RSS={}GiB RAM可用={}GiB/{:.2f}GiB swap={}GiB OOM={}".format(
            len(project_processes),
            _fmt_gib(project_rss_kb),
            _fmt_gib(available_kb),
            total_kb / 1024.0 / 1024.0 if total_kb else 0.0,
            _fmt_gib(swap_used_kb),
            oom_count,
        )
    )
    for gpu in gpus:
        lines.append(
            "GPU{index}: util={utilization}% memory={memory_used_mb}/{memory_total_mb} MiB".format(**gpu)
        )
    lines.append("告警: {}".format("；".join(warnings) if warnings else "无"))
    return "\n".join(lines)


def jobs_view(root: Path, pid_file: Path, processes: Sequence[ProcessInfo]) -> str:
    pid = _read_pid(pid_file)
    main_process = next((process for process in processes if process.pid == pid), None)
    pgid = main_process.pgid if main_process else None
    gpu_memory = _gpu_process_memory()
    rows = []
    for process in processes:
        if pgid is not None and process.pgid != pgid:
            continue
        parsed = parse_job(process)
        if parsed:
            rows.append((process, parsed[0], parsed[1]))
    rows.sort(key=lambda value: (STAGES.index(value[1]), value[2], value[0].pid))
    lines = [datetime.now(timezone.utc).isoformat(), "活动子任务: {}".format(len(rows))]
    lines.append("PID      stage         participant       elapsed    RSS_MiB GPU_MiB state")
    for process, stage, participant in rows:
        lines.append(
            "{:<8} {:<13} {:<17} {:>9} {:>8.1f} {:>7} {}".format(
                process.pid,
                stage,
                participant,
                format_duration(process.elapsed_seconds),
                process.rss_kb / 1024.0,
                gpu_memory.get(process.pid, 0),
                process.state,
            )
        )
    return "\n".join(lines)


def _participants(root: Path) -> List[str]:
    config = root / "config" / "config.yaml"
    participant_value = None
    for line in config.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^participants:\s*['\"]?([^'\"#]+)", line)
        if match:
            participant_value = match.group(1).strip()
            break
    if not participant_value:
        raise RuntimeError("config.yaml缺少顶层participants")
    path = Path(participant_value)
    if not path.is_absolute():
        path = root / path
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["participant_id"]) for row in csv.DictReader(handle, delimiter="\t")]


def _load_status(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("status") in {"pass", "fail"} else None


def _collect_statuses(
    derivatives: Path, participant_ids: Sequence[str]
) -> Dict[Tuple[str, str], Dict[str, object]]:
    statuses: Dict[Tuple[str, str], Dict[str, object]] = {}
    for participant_id in participant_ids:
        status_dir = derivatives / "sub-{}".format(participant_id) / "status"
        for stage in STAGES:
            payload = _load_status(status_dir / "{}.json".format(stage))
            if payload:
                statuses[(stage, participant_id)] = payload
    return statuses


def summarize_progress(
    derivatives: Path,
    participant_ids: Sequence[str],
    running_jobs: Set[Tuple[str, str]],
    statuses: Optional[Dict[Tuple[str, str], Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    statuses = statuses if statuses is not None else _collect_statuses(derivatives, participant_ids)
    rows = []
    for stage in STAGES:
        passed = failed = running = 0
        latest: Optional[Tuple[str, str, str, object]] = None
        for participant_id in participant_ids:
            payload = statuses.get((stage, participant_id))
            if (stage, participant_id) in running_jobs:
                running += 1
            elif payload and payload["status"] == "pass":
                passed += 1
            elif payload and payload["status"] == "fail":
                failed += 1
            if payload:
                details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
                runtime = details.get("runtime", {}) if isinstance(details.get("runtime"), dict) else {}
                finished = str(runtime.get("finished_at_utc") or payload.get("timestamp_utc") or "")
                candidate = (finished, participant_id, str(payload["status"]), runtime.get("duration_seconds"))
                if latest is None or candidate[0] > latest[0]:
                    latest = candidate
        total = len(participant_ids)
        pending = max(0, total - passed - failed - running)
        rows.append(
            {
                "stage": stage,
                "passed": passed,
                "failed": failed,
                "running": running,
                "pending": pending,
                "total": total,
                "pass_percent": 100.0 * passed / total if total else 0.0,
                "ended_percent": 100.0 * (passed + failed) / total if total else 0.0,
                "latest_finished": latest[0] if latest else "NA",
                "latest_participant": latest[1] if latest else "NA",
                "latest_status": latest[2] if latest else "NA",
                "latest_duration": latest[3] if latest else None,
            }
        )
    return rows


def _record_runtime_history(
    root: Path,
    participant_ids: Sequence[str],
    statuses: Dict[Tuple[str, str], Dict[str, object]],
) -> None:
    output = root / "logs" / "stage_runtime_history.tsv"
    known: Set[Tuple[str, str, str]] = set()
    if output.is_file():
        with output.open("r", encoding="utf-8", newline="") as handle:
            known = {
                (row["participant_id"], row["stage"], row["finished_at_utc"])
                for row in csv.DictReader(handle, delimiter="\t")
            }
    fields = (
        "participant_id", "stage", "status", "started_at_utc", "finished_at_utc",
        "duration_seconds", "peak_rss_mb_self", "peak_rss_mb_children",
    )
    for participant_id in participant_ids:
        for stage in STAGES:
            payload = statuses.get((stage, participant_id))
            if not payload:
                continue
            details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
            runtime = details.get("runtime", {}) if isinstance(details.get("runtime"), dict) else {}
            finished = str(runtime.get("finished_at_utc") or "")
            if not finished or (participant_id, stage, finished) in known:
                continue
            row = {name: runtime.get(name, "") for name in fields}
            row.update({"participant_id": participant_id, "stage": stage, "status": payload["status"]})
            _append_tsv(output, fields, row)
            known.add((participant_id, stage, finished))


def progress_view(root: Path, processes: Sequence[ProcessInfo]) -> str:
    participant_ids = _participants(root)
    running_jobs = {parsed for process in processes for parsed in [parse_job(process)] if parsed is not None}
    derivatives = root / "derivatives" / "substain_features"
    statuses = _collect_statuses(derivatives, participant_ids)
    rows = summarize_progress(derivatives, participant_ids, running_jobs, statuses=statuses)
    _record_runtime_history(root, participant_ids, statuses)
    lines = [datetime.now(timezone.utc).isoformat(), "当前participants总量: {}".format(len(participant_ids))]
    lines.append(
        "stage         pass  fail running pending total  pass% ended% latest_id         latest status duration  finished"
    )
    for row in rows:
        lines.append(
            "{stage:<13} {passed:>5} {failed:>5} {running:>7} {pending:>7} {total:>5} "
            "{pass_percent:>5.1f} {ended_percent:>6.1f} {latest_participant:<17} "
            "{latest_status:<6} {duration:<9} {latest_finished}".format(
                duration=format_duration(row["latest_duration"]), **row
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("view", choices=("main", "jobs", "progress"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    pid_file = args.pid_file or (root / "logs" / "full_run.pid")
    if not pid_file.is_absolute():
        pid_file = root / pid_file
    if args.interval <= 0:
        raise SystemExit("--interval必须大于0")
    while True:
        processes = read_processes()
        if args.view == "main":
            output = main_view(root, pid_file, processes)
        elif args.view == "jobs":
            output = jobs_view(root, pid_file, processes)
        else:
            output = progress_view(root, processes)
        if not args.once and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
