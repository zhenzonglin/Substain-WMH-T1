#!/usr/bin/env python3
"""Compare fresh successful T1 CPU and CUDA runtimes from status history."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


def percentile(values: List[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Iterable[float]) -> Dict[str, float]:
    sample = list(values)
    if not sample:
        return {}
    return {
        "n": len(sample),
        "mean_seconds": statistics.fmean(sample),
        "sd_seconds": statistics.stdev(sample) if len(sample) > 1 else 0.0,
        "median_seconds": statistics.median(sample),
        "q1_seconds": percentile(sample, 0.25),
        "q3_seconds": percentile(sample, 0.75),
        "min_seconds": min(sample),
        "max_seconds": max(sample),
    }


def iter_payloads(derivatives: Path) -> Iterable[Tuple[Mapping[str, object], Path]]:
    for history_path in derivatives.glob("sub-*/status/status_history.jsonl"):
        try:
            lines = history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("无法解析{}第{}行: {}".format(history_path, line_number, error))
            if isinstance(payload, dict):
                yield payload, history_path

    # 兼容极旧版本：若只有当前t1.json而没有完整history，仍纳入候选并按时间去重。
    for status_path in derivatives.glob("sub-*/status/t1.json"):
        if (status_path.parent / "status_history.jsonl").is_file():
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            yield payload, status_path


def collect(derivatives: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    latest: Dict[Tuple[str, str], Dict[str, object]] = {}
    counters = {
        "t1_payloads_seen": 0,
        "failed": 0,
        "missing_runtime": 0,
        "missing_or_unknown_device": 0,
        "reused_inference": 0,
        "resumed_checkpoint": 0,
    }
    for payload, source in iter_payloads(derivatives):
        if payload.get("stage") != "t1":
            continue
        counters["t1_payloads_seen"] += 1
        if payload.get("status") != "pass":
            counters["failed"] += 1
            continue
        details = payload.get("details")
        if not isinstance(details, dict):
            counters["missing_runtime"] += 1
            continue
        runtime = details.get("runtime")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("duration_seconds"), (int, float)):
            counters["missing_runtime"] += 1
            continue
        duration = float(runtime["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            counters["missing_runtime"] += 1
            continue
        device = str(details.get("effective_device", "")).lower()
        if device == "gpu":
            device = "cuda"
        if device not in {"cpu", "cuda"}:
            counters["missing_or_unknown_device"] += 1
            continue
        if bool(details.get("inference_reused", False)):
            counters["reused_inference"] += 1
            continue
        if bool(details.get("initial_execution_resumed_from_dlicv_checkpoint", False)):
            counters["resumed_checkpoint"] += 1
            continue
        participant = str(payload.get("participant_id", "")).strip()
        if not participant:
            continue
        timestamp = str(payload.get("timestamp_utc", payload.get("timestamp", "")))
        record = {
            "participant_id": participant,
            "device": device,
            "duration_seconds": duration,
            "duration_minutes": duration / 60.0,
            "timestamp": timestamp,
            "requested_device": str(details.get("requested_device", "")),
            "source": str(source),
        }
        key = (participant, device)
        previous = latest.get(key)
        if previous is None or timestamp >= str(previous["timestamp"]):
            latest[key] = record
    return sorted(latest.values(), key=lambda item: (str(item["device"]), str(item["participant_id"]))), counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--derivatives", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    derivatives = (
        args.derivatives.resolve()
        if args.derivatives is not None
        else project_root / "derivatives" / "substain_features"
    )
    if not derivatives.is_dir() or derivatives.is_symlink():
        raise SystemExit("derivatives不存在或是软链接: {}".format(derivatives))

    records, counters = collect(derivatives)
    by_device = {
        device: {str(item["participant_id"]): float(item["duration_seconds"]) for item in records if item["device"] == device}
        for device in ("cpu", "cuda")
    }
    cpu_stats = describe(by_device["cpu"].values())
    gpu_stats = describe(by_device["cuda"].values())
    paired_ids = sorted(set(by_device["cpu"]) & set(by_device["cuda"]))
    paired_ratios = [by_device["cpu"][participant] / by_device["cuda"][participant] for participant in paired_ids]
    paired_differences = [by_device["cpu"][participant] - by_device["cuda"][participant] for participant in paired_ids]

    summary: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "derivatives": str(derivatives),
        "selection": "latest successful fresh non-resumed T1 run per participant per effective device",
        "counters": counters,
        "cpu": cpu_stats,
        "cuda": gpu_stats,
        "unpaired_median_speedup_cpu_over_cuda": (
            cpu_stats["median_seconds"] / gpu_stats["median_seconds"] if cpu_stats and gpu_stats else None
        ),
        "paired": {
            "n": len(paired_ids),
            "median_speedup_cpu_over_cuda": statistics.median(paired_ratios) if paired_ratios else None,
            "median_seconds_saved": statistics.median(paired_differences) if paired_differences else None,
            "participant_ids": paired_ids,
        },
        "interpretation_limit": (
            "Unpaired CPU/GPU cohorts can differ in image dimensions and case mix; "
            "use the paired estimate when paired n is adequate."
        ),
    }

    output_dir = args.output_dir
    if output_dir is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "logs" / "t1_runtime_comparison_{}".format(tag)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "t1_runtime_records.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["participant_id", "device", "duration_seconds"] , delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def show(label: str, values: Mapping[str, float]) -> None:
        if not values:
            print("{}: n=0".format(label))
            return
        print(
            "{}: n={} median={:.2f} min IQR={:.2f}-{:.2f} min mean={:.2f} min sd={:.2f} min".format(
                label,
                int(values["n"]),
                values["median_seconds"] / 60.0,
                values["q1_seconds"] / 60.0,
                values["q3_seconds"] / 60.0,
                values["mean_seconds"] / 60.0,
                values["sd_seconds"] / 60.0,
            )
        )

    print("T1 CPU vs GPU fresh-runtime comparison")
    show("CPU", cpu_stats)
    show("GPU", gpu_stats)
    if cpu_stats and gpu_stats:
        print("非配对中位数加速倍数: {:.2f}x".format(summary["unpaired_median_speedup_cpu_over_cuda"]))
    print("配对病例: n={}".format(len(paired_ids)))
    if paired_ratios:
        print("配对中位数加速倍数: {:.2f}x".format(statistics.median(paired_ratios)))
        print("配对中位数节省: {:.2f} min".format(statistics.median(paired_differences) / 60.0))
    print("排除: reused={} resumed={} missing_runtime={} unknown_device={} failed={}".format(
        counters["reused_inference"],
        counters["resumed_checkpoint"],
        counters["missing_runtime"],
        counters["missing_or_unknown_device"],
        counters["failed"],
    ))
    print("输出目录: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
