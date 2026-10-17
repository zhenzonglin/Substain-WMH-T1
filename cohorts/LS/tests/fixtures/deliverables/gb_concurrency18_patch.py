#!/usr/bin/env python3
"""Small, backed-up edits to the installed GB project; no archive/environment build."""
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
import gb_resume_audit as reuse

MARKER = "# GB_RESUME_AUDIT_V1"
COMPUTE_RULES = ("skullstrip", "wmh_segmentation", "registration", "lesion_processing",
                 "wmh", "t1", "qc", "cleanup", "lowres", "aggregate")


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError("Unexpected source context; no files changed: " + old[:110])
    return text.replace(old, new)


def planned_edits(root):
    originals = {}
    planned = {}

    def read(name):
        path = root / name
        if path.is_symlink():
            raise ValueError("Refusing symlink: " + name)
        data = path.read_bytes()
        originals[name] = data
        return data.decode("utf-8")

    snake = read("workflow/Snakefile")
    if MARKER in snake:
        raise ValueError("Patch already installed; use --verify instead of applying it again")
    if 'ROLLING_ADMISSION_INPUT = ancient(ROLLING_TOKEN_PATTERN)' not in snake:
        raise ValueError("Expected the existing 200-subject rolling workflow")

    old_text = read("config/config.yaml")
    old = yaml.safe_load(old_text)
    execution = old["execution"]
    if (execution.get("cpu_cores") != 64 or execution.get("cpu_threads_per_job") != 4 or
            execution.get("max_parallel_jobs") != 200 or execution.get("gpu_slots_per_device") != 2 or
            [str(item) for item in execution.get("gpu_devices", [])] != ["0", "1"] or
            execution.get("t1_gpu_min_free_memory_mib") != 12288 or
            execution.get("wmh_gpu_min_free_memory_mib") != 20480):
        raise ValueError("Expected the confirmed CPU64 / heavy4 / dual-GPU GB configuration")
    new_text = replace(old_text, "  cpu_threads_per_job: 4\n",
                       "  cpu_threads_per_job: 2\n  registration_threads: 3\n  max_compute_jobs: 18\n")
    expected = copy.deepcopy(old)
    expected["execution"].update(cpu_threads_per_job=2, registration_threads=3, max_compute_jobs=18)
    if yaml.safe_load(new_text) != expected:
        raise ValueError("Unexpected config change")
    planned["config/config.yaml"] = new_text

    start = snake.index("rule audit:\n")
    stop = snake.index("\n\nif ROLLING_ENABLED:", start)
    original_audit = snake[start:stop]
    if ('config=str(CONFIG_FILE)' not in original_audit or
            'images=PARTICIPANT_INPUTS' not in original_audit or
            '-m substain_features.cli audit' not in original_audit):
        raise ValueError("Unexpected audit rule")
    guard = (MARKER + "\nimport importlib.util as _gb_importlib\n"
             '_gb_spec = _gb_importlib.spec_from_file_location("gb_resume_audit", ROOT / "scripts/gb_resume_audit.py")\n'
             "_gb_audit = _gb_importlib.module_from_spec(_gb_spec)\n"
             "_gb_spec.loader.exec_module(_gb_audit)\n"
             "GB_AUDIT_INPUT = ancient(_gb_audit.validate(ROOT, CONFIG_FILE))\n")
    snake = snake[:start] + guard + snake[stop:]
    snake = replace(snake, "audit=rules.audit.output,", "audit=GB_AUDIT_INPUT,")
    snake = replace(snake, "    threads: 8\n", "    threads: 3\n")
    for rule in COMPUTE_RULES:
        pattern = r"(?ms)^rule " + re.escape(rule) + r":\n.*?(?=^rule |^if ROLLING_ENABLED:|\Z)"
        matches = list(re.finditer(pattern, snake))
        if len(matches) != 1:
            raise ValueError("Expected one rule: " + rule)
        block = matches[0].group()
        block = replace(block, "    resources:\n", "    resources:\n        compute_jobs=1,\n")
        snake = snake[:matches[0].start()] + block + snake[matches[0].end():]
    planned["workflow/Snakefile"] = snake

    text = read("scripts/start_ws1_v1_0_9.sh")
    text = replace(text, "CPU_THREADS_PER_JOB=4", "CPU_THREADS_PER_JOB=2")
    text = text.replace("full_run_4threads_", "full_run_18jobs_")
    planned["scripts/start_ws1_v1_0_9.sh"] = text
    text = read("scripts/finish_ws1_v1_0_9.sh")
    text = replace(text, 'cpu_threads="${CPU_THREADS_PER_JOB:-4}"', 'cpu_threads="${CPU_THREADS_PER_JOB:-2}"')
    text = replace(text, '"${cpu_threads}" == 4', '"${cpu_threads}" == 2')
    text = replace(text, "GB requires 4 threads and window 200", "GB requires 2 threads and window 200")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        text = replace(text, "export " + variable + "=4", "export " + variable + "=2")
    text = replace(text, '--resources "gpu=${gpu_slots}" "wmh_exclusive=2"',
                   '--resources "gpu=${gpu_slots}" "wmh_exclusive=2" "compute_jobs=18"', count=2)
    text = replace(text, 'echo "正式资源: cores=', 'echo "正式资源: max_compute_jobs=18, registration_threads=3, cores=')
    planned["scripts/finish_ws1_v1_0_9.sh"] = text

    text = read("run_pipeline.sh")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        text = replace(text, variable + "=4", variable + "=2")
    # Resume the fixed cohort while preserving the requested participant/profile arguments.
    # An explicit `audit` command is still available outside ordinary resume entrypoints.
    text = replace(text,
        '    "${root}/scripts/steps/00_prepare.sh"\n    "${root}/scripts/steps/01_audit.sh" "${participant}"',
        '    echo "GB resume: reuse passed cohort audit; inputs are fixed."')
    planned["run_pipeline.sh"] = text
    text = read("src/substain_features/cli.py")
    text = replace(text, '            "--resources",\n', '            "--resources",\n            "compute_jobs=18",\n')
    planned["src/substain_features/cli.py"] = text
    text = read("src/substain_features/registration.py")
    text = replace(text, 'command[command.index("-n") + 1] = "8"', 'command[command.index("-n") + 1] = "3"')
    text = replace(text, 'command.extend(["-n", "8"])', 'command.extend(["-n", "3"])')
    planned["src/substain_features/registration.py"] = text
    text = read("src/substain_features/wmh.py")
    text = replace(text, '        "--threads",\n        "4",', '        "--threads",\n        "2",')
    planned["src/substain_features/wmh.py"] = text

    manifest_name = "logs/gb/rerun_manifest.json"
    manifest = json.loads(read(manifest_name))
    manifest.update(cpu_cores=64, cpu_threads_per_job=2, registration_threads=3, max_compute_jobs=18)
    planned[manifest_name] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    helper_name = "scripts/gb_resume_audit.py"
    if (root / helper_name).exists():
        raise ValueError("Existing helper must not be overwritten: " + helper_name)
    originals[helper_name] = None
    planned[helper_name] = Path(reuse.__file__).read_text(encoding="utf-8")
    for name, text in planned.items():
        if name.endswith(".py"):
            compile(text, str(root / name), "exec")
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    return originals, {name: value.encode("utf-8") for name, value in planned.items()}


def require_idle(root):
    path = root / "logs/gb/full_run.pid"
    if path.exists():
        pid = path.read_text().strip()
        if not pid.isdigit() or int(pid) <= 1:
            raise ValueError("Invalid full_run.pid")
        result = subprocess.run(["pgrep", "-g", pid], capture_output=True, text=True)
        if result.returncode != 1:
            raise RuntimeError("GB process group still exists (or cannot be checked); wait for drain first")
    # Reject a separately launched scheduler or stage using this project.
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = (path / "cmdline").read_bytes().split(b"\0")
        except (OSError, ProcessLookupError):
            continue
        if not argv or b"python" not in argv[0].lower():
            continue
        if (str(root).encode() in b" ".join(argv) and
                (b"snakemake" in argv or b"substain_features.cli" in argv)):
            raise RuntimeError("Another GB scheduler/stage is active: PID=" + path.name)


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=".gb18-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    if path.exists():
        shutil.copymode(str(path), str(temporary))
    os.replace(str(temporary), str(path))


def apply(root, check_only=False):
    root = root.resolve(strict=True)
    originals, planned = planned_edits(root)
    payload = reuse.capture(root)
    if check_only:
        print("Preflight passed: full audit reusable; patch contexts checked; no project files changed")
        return
    require_idle(root)
    cache_name = reuse.CACHE
    if (root / cache_name).exists():
        raise ValueError("Existing audit cache must not be overwritten")
    originals[cache_name] = None
    planned[cache_name] = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    backup = root / "logs/gb" / ("concurrency18-backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    backup.mkdir(parents=True)
    for name, data in originals.items():
        if data is not None:
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(root / name), str(saved))
    (backup / "changes.json").write_text(json.dumps({name: {
        "before": reuse.digest(originals[name]) if originals[name] is not None else None,
        "after": reuse.digest(data)} for name, data in planned.items()}, indent=2) + "\n", encoding="utf-8")
    for name, data in originals.items():
        path = root / name
        if (path.read_bytes() if path.exists() else None) != data:
            raise RuntimeError("File changed during patch: " + name)
    # All checks precede mutations; keep full originals for rollback if a write fails.
    written = []
    try:
        for name, data in planned.items():
            write_atomic(root / name, data)
            written.append(name)
    except Exception:
        for name in reversed(written):
            if originals[name] is None:
                (root / name).unlink()
            else:
                shutil.copy2(str(backup / name), str(root / name))
        raise
    print("GB18_APPLIED CPU=64 heavy=2 registration=3 compute_jobs=18 GPU=0,1 rolling=200")
    print("Backup:", backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "apply", "verify"))
    parser.add_argument("--root", type=Path, default=Path("/data02/linzhenzong/Substain_GB"))
    args = parser.parse_args()
    if args.mode == "verify":
        reuse.validate(args.root)
    else:
        apply(args.root, check_only=args.mode == "check")
