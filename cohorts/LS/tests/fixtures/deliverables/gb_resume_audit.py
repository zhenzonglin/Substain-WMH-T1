#!/usr/bin/env python3
"""Reuse a passed full-cohort audit, checking identity and file metadata on resume.

This does not rehash NIfTI content or repeat image/grid validation. The initial
capture trusts the recorded audit and rejects image mtimes newer than that audit.
Later resumes compare the captured device/inode, size, mtime and ctime.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

CACHE = "logs/gb/resume_audit_cache.json"
AUDIT_CODE = ("audit.py", "images.py", "schema.py", "resources.py")
RUNTIME_KEYS = {"cpu_cores", "cpu_threads_per_job", "registration_threads",
                "max_compute_jobs", "max_parallel_jobs", "gpu_devices",
                "gpu_slots_per_device", "gpu_policy", "gpu_lock_dir",
                "t1_gpu_min_free_memory_mib", "wmh_gpu_min_free_memory_mib"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def absolute(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def config_identity(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name in RUNTIME_KEYS:
        config.get("execution", {}).pop(name, None)
    return digest(json.dumps(config, sort_keys=True).encode("utf-8"))


def signature(path):
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Expected regular file: " + str(path))
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def scan(paths):
    result = {}
    print("GB_AUDIT_METADATA checking {} files (no NIfTI rehash)".format(len(paths)), flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (path, value) in enumerate(zip(paths, pool.map(signature, paths)), 1):
            result[path] = value
            if index % 5000 == 0 or index == len(paths):
                print("GB_AUDIT_METADATA {}/{}".format(index, len(paths)), flush=True)
    return result


def identity(root, config_path, report_path, participants_path):
    return {
        "project_root": str(root), "config_path": str(config_path),
        "config_identity": config_identity(config_path),
        "report_path": str(report_path), "report_sha256": digest(report_path.read_bytes()),
        "participants_path": str(participants_path),
        "participants_sha256": digest(participants_path.read_bytes()),
        "audit_code": {name: digest((root / "src/substain_features" / name).read_bytes())
                       for name in AUDIT_CODE},
    }


def context(root, config_path=None):
    root = Path(root).resolve(strict=True)
    config_path = Path(config_path or root / "config/config.yaml").resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    declared = absolute(config_path.parent, config["project_root"]).resolve(strict=True)
    if declared != root or config.get("input", {}).get("cohort") != "GB":
        raise ValueError("Expected the existing GB project and its own config")
    report = absolute(root, config["derivatives"]) / "audit/audit_report.json"
    return root, config_path, report, absolute(root, config["participants"])


def capture(root):
    root, config_path, report_path, participants_path = context(root)
    before = identity(root, config_path, report_path, participants_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("missing_resources"):
        raise ValueError("Full audit has not passed; no changes made")
    with participants_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    subjects = report.get("participants", [])
    by_id = {row["participant_id"]: row for row in subjects}
    ids = [row["participant_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or len(by_id) != len(subjects) or set(ids) != set(by_id):
        raise ValueError("Audit must cover the entire current cohort, not only the pilot")
    images = set()
    for row in rows:
        saved = by_id[row["participant_id"]]
        if (saved.get("status") != "pass" or saved.get("errors") or
                float(saved["age"]) != float(row["age"]) or
                str(saved["sex"]) != row["sex"] or str(saved["site_id"]) != row["site_id"]):
            raise ValueError("Audit metadata/status mismatch: " + row["participant_id"])
        for field in ("t1w", "flair", "lesion_mask"):
            path = absolute(root, row[field])
            audited = saved["images"][field]
            same_path = str(path) == str(Path(audited["path"])) or path.resolve(strict=True) == Path(audited["path"]).resolve(strict=True)
            if (not same_path or
                    not re.fullmatch(r"[0-9a-fA-F]{64}", str(audited.get("sha256", "")))):
                raise ValueError("Audit image identity mismatch: " + str(path))
            images.add(str(path))
    resources = set()
    for item in report.get("resources", []):
        path = Path(item["path"])
        if not item.get("exists") or not path.exists():
            raise ValueError("Missing audited resource: " + str(path))
        if path.is_file():
            resources.add(str(path))
    if not resources:
        raise ValueError("Audit contains no resource records")
    states = scan(sorted(images | resources))
    report_time = report_path.stat().st_mtime_ns
    for path in images | resources:
        if states[path][3] > report_time:
            raise ValueError("File newer than passed audit; re-audit explicitly: " + path)
    if before != identity(root, config_path, report_path, participants_path):
        raise ValueError("Inputs/config/report changed during capture; retry")
    return dict(before, schema_version=1, subject_count=len(ids), file_states=states)


def validate(root, config_path=None):
    root, config_path, report_path, participants_path = context(root, config_path)
    cache_path = root / CACHE
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    if saved.get("schema_version") != 1:
        raise ValueError("Unsupported GB audit cache")
    current = identity(root, config_path, report_path, participants_path)
    for key, value in current.items():
        if saved.get(key) != value:
            raise ValueError("GB cached audit changed: {}. Explicit re-audit/recapture required.".format(key))
    states = saved.get("file_states", {})
    if not states:
        raise ValueError("GB audit cache has no file records")
    observed = scan(sorted(states))
    for path, value in observed.items():
        if value != states[path]:
            raise ValueError("GB input/resource changed; stopped without re-auditing: " + path)
    if current != identity(root, config_path, report_path, participants_path):
        raise ValueError("Audit identity changed during metadata check")
    print("GB_AUDIT_REUSED status=pass subjects={} image_rehash=0".format(saved["subject_count"]), flush=True)
    return str(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "capture"))
    parser.add_argument("--root", type=Path, default=Path("/data02/linzhenzong/Substain_GB"))
    args = parser.parse_args()
    if args.mode == "check":
        validate(args.root)
    else:
        payload = capture(args.root)
        target = args.root / CACHE
        if target.exists():
            raise SystemExit("Cache already exists; preserve it and explicitly resolve the changed audit first")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp-{}".format(os.getpid()))
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(str(temporary), str(target))
