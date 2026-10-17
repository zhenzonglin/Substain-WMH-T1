#!/usr/bin/env python3
"""Direct edits to the existing Substain_GB archive; no replacement bundle."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

BASE = Path('/data02/linzhenzong')
RESULTS = '/mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_GB/derivatives/substain_features'
BIDS = '/mnt/share/GPUStore2/usersdir/linzhenzong/GB2/GB_BIDS_NIFTI_QC'
LESIONS = '/data02/linzhenzong/GB_lesion_MNI_BH/C3_lesion_MNI_BH_1mm_3mm'
POOL_SHA256 = '9e4eef8f27c5dd4b51dee7437edbf099d980e8187aed73c90374ddf64badcbed'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def extract(base: Path) -> None:
    archive = base / 'Substain_GB.tar.gz'
    checksum = archive.with_name(archive.name + '.sha256')
    lines = [line for line in checksum.read_text().splitlines() if line.strip()]
    if len(lines) != 1 or not re.fullmatch(r'[0-9a-fA-F]{64}', lines[0].split()[0]):
        raise ValueError('Invalid archive SHA256 file')
    expected = lines[0].split()[0].lower()
    if digest(archive) != expected:
        raise ValueError('Archive SHA256 mismatch')
    print('Archive SHA256 passed: ' + expected, flush=True)
    destination = base / 'Substain_GB'
    if destination.exists() or destination.is_symlink():
        raise ValueError('Project already exists; skip extract and inspect it before apply')
    manifest_dir = base / 'Substain_GB_manifest'
    file_manifest = manifest_dir / 'FILES.sha256'
    if manifest_dir.exists() and not file_manifest.is_file():
        raise ValueError('Manifest directory exists but FILES.sha256 is missing')
    with tarfile.open(str(archive), 'r:gz') as handle:
        members = handle.getmembers()
        names = set()
        links = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'Substain':
                raise ValueError('Unsafe archive member: ' + member.name)
            name = str(path)
            if name in names or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError('Duplicate or special archive member: ' + name)
            names.add(name)
            if member.issym() or member.islnk():
                target = member.linkname if member.islnk() else str(path.parent / member.linkname)
                normalized = PurePosixPath(os.path.normpath(target))
                if normalized.is_absolute() or not normalized.parts or normalized.parts[0] != 'Substain' or '..' in normalized.parts:
                    raise ValueError('Unsafe archive link: ' + name)
                links.add(name)
        if any(str(parent) in links for name in names for parent in PurePosixPath(name).parents):
            raise ValueError('Archive member traverses a link')
        needed = sum(member.size for member in members if member.isfile()) + 1024 ** 3
        if shutil.disk_usage(base).free < needed:
            raise ValueError('Insufficient free disk space for extraction')
        temporary = Path(tempfile.mkdtemp(prefix='.Substain_GB-extract-', dir=str(base)))
        print('Extracting into ' + str(temporary), flush=True)
        try:
            handle.extractall(str(temporary))
            project = temporary / 'Substain'
            if file_manifest.is_file():
                verified = set()
                for line in file_manifest.read_text().splitlines():
                    if not line.strip():
                        continue
                    checksum_value, name = line.split(maxsplit=1)
                    name = name.lstrip('*')
                    path = PurePosixPath(name)
                    if path.is_absolute() or '..' in path.parts or path.parts[0] != 'Substain':
                        raise ValueError('Unsafe manifest path: ' + name)
                    candidate = temporary / name
                    if name in verified or not candidate.is_file() or candidate.is_symlink() or digest(candidate) != checksum_value.lower():
                        raise ValueError('File SHA256 mismatch: ' + name)
                    verified.add(name)
                if verified != {member.name for member in members if member.isfile()}:
                    raise ValueError('Manifest does not cover all archive files')
            for name in ('config/config.yaml', 'workflow/Snakefile', 'scripts/finish_ws1_v1_0_9.sh', 'scripts/install_offline.sh'):
                if not (project / name).is_file():
                    raise ValueError('Archive lacks existing workflow: ' + name)
            (project / '.gb-minimal').mkdir(exist_ok=True)
            (project / '.gb-minimal/archive.json').write_text(json.dumps({
                'archive': str(archive), 'sha256': expected,
                'per_file_verified': file_manifest.is_file()}, indent=2) + '\n')
            project.rename(destination)
        except BaseException:
            print('Extraction not installed; staging directory retained: ' + str(temporary), flush=True)
            raise
        temporary.rmdir()
    print('Installed existing archive at ' + str(destination))
    if not file_manifest.is_file():
        print('Only whole-archive SHA256 verified; no per-file manifest was supplied')


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError('Unexpected source context: ' + old[:100])
    return text.replace(old, new, 1)


def apply(root: Path, sas_file: str | None = None) -> None:
    import yaml

    root = root.resolve()
    backup = root / '.gb-minimal/original'
    ledger_path = root / '.gb-minimal/changes.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    originals = {}

    def source(name: str) -> str:
        path = root / name
        if name in ledger and digest(path) != ledger[name]['after']:
            raise ValueError('File changed since patch; refusing overwrite: ' + name)
        baseline = backup / name if (backup / name).is_file() else path
        originals[name] = baseline.read_text()
        return originals[name]

    pool_path = backup / 'src/substain_features/gpu_pool.py'
    if not pool_path.is_file():
        pool_path = root / 'src/substain_features/gpu_pool.py'
    if digest(pool_path) != POOL_SHA256:
        raise ValueError('GPU source is not the latest archived baseline; no files changed')
    planned = {}
    name = 'src/substain_features/gpu_pool.py'
    text = source(name)
    start = text.index('def run_with_gpu_lock(')
    end = text.index('\ndef _command_stage(', start)
    planned[name] = text[:start] + GPU_FUNCTION + '\n\n' + text[end:]
    planned[name] = planned[name].replace('仅T1：空槽用GPU0，否则立即CPU', '仅T1：检查所有配置GPU，无可用槽位才CPU')

    name = 'src/substain_features/input_prep.py'
    planned[name] = replace_once(source(name), '    mode = str(input_config.get("mode", "")).strip().lower()',
        '    if input_config.get("cohort") == "GB":\n        from .gb_inputs import prepare_gb_inputs\n        return prepare_gb_inputs(config)\n    mode = str(input_config.get("mode", "")).strip().lower()')
    adapter_name = 'src/substain_features/gb_inputs.py'
    if (root / adapter_name).exists() and adapter_name not in ledger:
        raise ValueError('Existing GB adapter would be overwritten')
    if adapter_name in ledger and digest(root / adapter_name) != ledger[adapter_name]['after']:
        raise ValueError('GB adapter changed since patch')
    planned[adapter_name] = GB_INPUT_SOURCE

    name = 'src/substain_features/audit.py'
    planned[name] = replace_once(source(name), '    if len(matches) != 1:',
        '    if path.name.endswith("_lesion_mask_to_MNI152_1mm.nii.gz") and [item[0] if isinstance(item, (tuple, list)) else item for item in matches] != ["1mm"]:\n'
        '        errors.append("GB lesion filename requires the physical FSL MNI152 1mm grid")\n'
        '    if len(matches) != 1:')

    name = 'config/config.yaml'
    config = yaml.safe_load(source(name))
    config.update(project_root='..', participants='config/participants_gb.tsv', derivatives=RESULTS, qc_dir=RESULTS + '/qc')
    config['input'].update(cohort='GB', mode='bids', bids_root=BIDS, t1_root=BIDS, flair_root=BIDS,
        lesion_root=LESIONS, lesion_suffix='_lesion_mask_to_MNI152_1mm.nii.gz',
        metadata_tsv='config/metadata_gb.tsv', bids_links='inputs/gb_bids_links', complete_cases=True)
    if sas_file:
        config['input']['sas_file'] = sas_file
    elif 'config/config.yaml' in ledger:
        active_sas = yaml.safe_load((root / 'config/config.yaml').read_text()).get('input', {}).get('sas_file')
        if active_sas:
            config['input']['sas_file'] = active_sas
    config['execution'].update(cpu_cores=50, cpu_threads_per_job=4, max_parallel_jobs=200,
        gpu_devices=['0', '1'], gpu_slots_per_device=2, gpu_policy='t1_shared2_wmh_exclusive',
        gpu_lock_dir=RESULTS + '/.gpu-locks', t1_gpu_min_free_memory_mib=12288, wmh_gpu_min_free_memory_mib=20480)
    planned[name] = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)

    name = 'src/substain_features/cli.py'
    text = source(name)
    text = replace_once(text, 'default=96, show_default=True)', 'default=50, show_default=True)')
    text = replace_once(text, '    gpu_ids = detect_gpu_ids()',
        '    slots_per_device = int(config.get("execution", {}).get("gpu_slots_per_device", 1))\n'
        '    gpu_ids = detect_gpu_ids()\n'
        '    if config.get("input", {}).get("cohort") == "GB":\n'
        '        if cores > 50:\n            raise click.ClickException("GB CPU budget is 50 cores")\n'
        '        gpu_ids = ["0", "1"]\n')
    text = replace_once(text, '"gpu={}".format(len(gpu_ids) if selected_profile == "gpu" else 0)',
        '"gpu={}".format(len(gpu_ids) * slots_per_device if selected_profile == "gpu" else 0)')
    text = replace_once(text, '            "finish_cpu={}".format(finish_cpu_slots),',
        '            "wmh_exclusive={}".format(len(gpu_ids) if selected_profile == "gpu" else 0),\n            "finish_cpu={}".format(finish_cpu_slots),')
    text = replace_once(text, '            "gpu_devices={}".format(",".join(gpu_ids)),',
        '            "gpu_devices={}".format(",".join(gpu_ids)),\n            "gpu_slots_per_device={}".format(slots_per_device),')
    planned[name] = text

    name = 'workflow/Snakefile'
    text = source(name)
    if 'ROLLING_ENABLED' not in text or 'threads: 8' not in text:
        raise ValueError('Archive does not contain the latest rolling scheduler')
    text = replace_once(text, '        wmh_exclusive=1 if PROFILE == "gpu" else 0\n    shell:',
        '        wmh_exclusive=1 if PROFILE == "gpu" else 0\n    threads: CPU_HEAVY_THREADS\n    shell:')
    text = text.replace('WMH-SynthSeg占满全部GPU槽位，不能与T1并行', 'WMH-SynthSeg占满所选单卡的两个槽位')
    text = text.replace('T1启动时尝试GPU0一槽', 'T1启动时尝试GPU0、GPU1各自的一槽')
    planned[name] = text
    name = 'src/substain_features/wmh.py'
    planned[name] = replace_once(source(name), '        "--threads",\n        "1",', '        "--threads",\n        "4",')
    name = 'src/substain_features/registration.py'
    planned[name] = replace_once(source(name), '    completed = subprocess.run(command,',
        '    if Path(command[0]).name == "antsRegistrationSyNQuick.sh":\n'
        '        command = list(command)\n'
        '        if "-n" in command:\n            command[command.index("-n") + 1] = "8"\n'
        '        else:\n            command.extend(["-n", "8"])\n'
        '    completed = subprocess.run(command,')

    name = 'scripts/finish_ws1_v1_0_9.sh'
    text = source(name).replace('cores="${1:-96}"', 'cores="${1:-50}"')
    text = replace_once(text, 'gpu_slots=2', 'gpu_slots=4\nslots_per_gpu=2')
    text = text.replace('logs/ws1_v1.0.9_rerun_manifest.path', 'logs/gb/rerun_manifest.path').replace('logs/ws1_v1.0.9_queue', 'logs/gb/queue')
    text = replace_once(text, 'derivatives="${root}/derivatives/substain_features"',
        'derivatives="$(${core_python} -c \'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["derivatives"])\' "${config}")"')
    text = text.replace('CUDA_VISIBLE_DEVICES:-0}', 'CUDA_VISIBLE_DEVICES:-0,1}').replace('wmh_exclusive=1', 'wmh_exclusive=2')
    text = replace_once(text, 'payload.get("interrupted_participants", [])', 'payload.get("queue_participants", payload.get("interrupted_participants", []))')
    text = text.replace('"gpu_devices=0"', '"gpu_devices=0,1"').replace('"gpu_slots_per_device=${gpu_slots}"', '"gpu_slots_per_device=${slots_per_gpu}"')
    text = replace_once(text, 'cd "${root}"',
        'cd "${root}"\n(( cores <= 50 )) || { echo "GB CPU budget is 50 cores" >&2; exit 2; }\n'
        '[[ "${cpu_threads}" == 4 && "${batch_size}" == 200 ]] || { echo "GB requires 4 threads and window 200" >&2; exit 2; }\n'
        'PYTHONPATH="${root}/src" "${core_python}" -c \'import sys; from pathlib import Path; from substain_features.schema import load_config; from substain_features.gb_inputs import check_pilot; check_pilot(load_config(Path(sys.argv[1])))\' "${config}"')
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        text = text.replace('export ' + variable + '="${' + variable + ':-${cpu_threads}}"', 'export ' + variable + '=4')
    text = replace_once(text, 'export OMP_NUM_THREADS=4', 'export OMP_NUM_THREADS=4\nexport OPENBLAS_NUM_THREADS=4\nexport NUMEXPR_NUM_THREADS=4')
    text += '\nPYTHONPATH="${root}/src" "${core_python}" -c \'import sys; from pathlib import Path; from substain_features.schema import load_config; from substain_features.gb_inputs import summarize; summarize(load_config(Path(sys.argv[1])))\' "${config}"\n'
    planned[name] = text
    name = 'scripts/start_ws1_v1_0_9.sh'
    text = source(name).replace('logs/full_run_v1.0.9', 'logs/gb/full_run').replace('logs/ws1_v1.0.9_rerun_manifest.path', 'logs/gb/rerun_manifest.path')
    text = text.replace('mkdir -p "${root}/logs"', 'mkdir -p "${root}/logs/gb"').replace('CUDA_VISIBLE_DEVICES=0 \\', 'CUDA_VISIBLE_DEVICES=0,1 \\')
    text = replace_once(text, '"${root}/scripts/finish_ws1_v1_0_9.sh" 96', '"${root}/scripts/finish_ws1_v1_0_9.sh" 50')
    text = text.replace('请先执行 prepare_ws1_v1_0_9_rerun.py --apply', '请先执行 run_pipeline.sh prepare 和 audit，再验证 c001p00016')
    planned[name] = text
    name = 'scripts/steps/02_features.sh'
    planned[name] = source(name).replace('cores="${3:-96}"', 'cores="${3:-50}"').replace('scripts/finish_backlog_then_all.sh', 'scripts/finish_ws1_v1_0_9.sh')
    name = 'run_pipeline.sh'
    text = source(name)
    if len(re.findall(r'(?m)^[ \t]*cores=', text)) != 1:
        raise ValueError('Expected one cores assignment in run_pipeline.sh')
    text, core_defaults = re.subn(
        r'(?m)^([ \t]*cores=)("?)\$\{4:-(?:[1-9][0-9]*|\$\{[A-Z_][A-Z0-9_]*\})\}\2([ \t]*(?:#[^\r\n]*)?)$',
        r'\1"${4:-50}"\3', text)
    if core_defaults != 1:
        raise ValueError('Unsupported cores default in run_pipeline.sh; no files changed')
    text = replace_once(text, 'case "${mode}" in', 'export CUDA_VISIBLE_DEVICES=0,1\nexport OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4\ncase "${mode}" in')
    planned[name] = text
    for name, destination in (('scripts/finish_backlog_then_all.sh', 'finish_ws1_v1_0_9.sh'), ('scripts/start_full_run.sh', 'start_ws1_v1_0_9.sh')):
        if (root / name).is_file():
            source(name)
            planned[name] = '#!/usr/bin/env bash\nset -euo pipefail\nroot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"\nexec bash "${root}/scripts/' + destination + '" "$@"\n'
    for name in ('scripts/monitor_ws1.py', 'scripts/monitor_ws1_progress_resources.py', 'scripts/verify_outputs.py'):
        if not (root / name).is_file():
            continue
        text = source(name)
        text = text.replace('root / "derivatives" / "substain_features"', 'root / __import__("yaml").safe_load((root / "config/config.yaml").read_text())["derivatives"]')
        text = text.replace('ROOT / "derivatives" / "substain_features"', 'ROOT / __import__("yaml").safe_load((ROOT / "config/config.yaml").read_text())["derivatives"]')
        text = text.replace('root / "logs" / "full_run_v1.0.9.pid"', 'root / "logs/gb/full_run.pid"')
        text = text.replace('str(item.relative_to(ROOT))', 'str(item.relative_to(derivatives))')
        text = text.replace('ROOT / "config" / "participants.tsv"', 'ROOT / "config" / "participants_gb.tsv"')
        text = text.replace('root / "config" / "participants.tsv"', 'root / "config" / "participants_gb.tsv"')
        text = text.replace('gpu_policy: auto_one_job_per_device', 'gpu_policy: t1_shared2_wmh_exclusive')
        text = text.replace('"gpu_policy": "auto_one_job_per_device"', '"gpu_policy": "t1_shared2_wmh_exclusive"')
        planned[name] = text

    differences = []
    for name, content in planned.items():
        if name.endswith('.py'):
            compile(content, name, 'exec')
        differences.extend(difflib.unified_diff(originals.get(name, '').splitlines(True), content.splitlines(True), fromfile='a/' + name, tofile='b/' + name))
    (root / '.gb-minimal').mkdir(exist_ok=True)
    (root / '.gb-minimal/changes.patch').write_text(''.join(differences))
    for name, content in planned.items():
        path = root / name
        if path.is_file() and not (backup / name).exists():
            (backup / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(backup / name))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text() != content:
            path.write_text(content)
        ledger[name] = {'before': digest(backup / name) if (backup / name).is_file() else None, 'after': digest(path)}
    ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    print('Updated existing project; backup/diff: ' + str(root / '.gb-minimal'))
    print('CPU=50; heavy=4; registration=8; GPU=0,1; slots/card=2; rolling=200')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('extract', 'apply'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--root', type=Path, default=BASE / 'Substain_GB')
    parser.add_argument('--sas-file')
    args = parser.parse_args()
    if args.action == 'extract':
        extract(args.base.resolve())
    else:
        apply(args.root, args.sas_file)


# Embedded source below is written as one GB input adapter. All inference,
# feature extraction, QC, cleanup and rolling-admission functions stay in place.
GPU_FUNCTION = r'''def run_with_gpu_lock(
    command: Sequence[str], gpu_ids: Sequence[str], lock_dir: Path,
    slots_per_gpu: int = 1, slots_required: int = 1,
    cpu_fallback: bool = False, min_free_memory_mib: int = 0,
) -> int:
    ids = [str(value).strip() for value in gpu_ids]
    if not command or not ids or len(ids) != len(set(ids)) or any(not value.isdigit() for value in ids):
        raise ValueError("A command and unique physical GPU IDs are required")
    if slots_per_gpu < 1 or not 1 <= slots_required <= slots_per_gpu or min_free_memory_mib < 0:
        raise ValueError("Invalid GPU slots or memory threshold")
    if cpu_fallback and (_command_stage(command) != "t1" or slots_per_gpu != 2 or slots_required != 1):
        raise ValueError("Busy-slot CPU fallback requires one-slot T1 on two-slot cards")
    cpu_command = _cpu_profile_command(command) if cpu_fallback or min_free_memory_mib else None
    exclusive = slots_required == slots_per_gpu and not cpu_fallback
    lock_dir.mkdir(parents=True, exist_ok=True)
    pending_gate = pending_id = None

    def acquire_gate(gpu_id):
        gate = (lock_dir / "gpu-{}-admission.lock".format(gpu_id)).open("a+")
        try:
            fcntl.flock(gate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return gate
        except BlockingIOError:
            gate.close()
            return None

    try:
        while True:
            reasons = {}
            candidates = []
            busy = False
            for gpu_id in ids:
                gate = pending_gate if gpu_id == pending_id else acquire_gate(gpu_id)
                if gate is None:
                    reasons[gpu_id] = "admission_busy"
                    busy = True
                    continue
                handles = []
                retain_gate = gate is pending_gate
                try:
                    handles = _try_acquire_slots(gpu_id, lock_dir, slots_per_gpu, slots_required)
                    if not handles:
                        reasons[gpu_id] = "slots_busy"
                        candidates.append(gpu_id)
                        busy = True
                        continue
                    # As in the archived code, WMH checks memory after project
                    # slots become free. It can probe the other card meanwhile.
                    if gate is pending_gate:
                        pending_gate = pending_id = None
                        retain_gate = False
                    free = _gpu_free_memory_mib(gpu_id) if min_free_memory_mib else 0
                    if free is None or free < min_free_memory_mib:
                        reasons[gpu_id] = "memory_query_failed" if free is None else "free_memory_{}_lt_{}".format(free, min_free_memory_mib)
                        continue
                    if pending_gate is not None:
                        _release_locks([pending_gate])
                        pending_gate = pending_id = None
                    if not exclusive:
                        _release_locks([gate])
                    return _run_selected(command, gpu_id, slots_required, "slot_acquired")
                finally:
                    _release_locks(handles)
                    if not retain_gate and not gate.closed:
                        _release_locks([gate])
            if cpu_fallback or (not busy and cpu_command is not None):
                reason = next(iter(reasons.values())) if len(ids) == 1 else "all_devices_unavailable:" + repr(reasons)
                return _run_selected(cpu_command, None, 0, reason)
            if exclusive and pending_gate is None:
                for gpu_id in candidates:
                    gate = acquire_gate(gpu_id)
                    if gate is not None:
                        pending_gate, pending_id = gate, gpu_id
                        break
            time.sleep(0.25)
    finally:
        if pending_gate is not None:
            _release_locks([pending_gate])
'''
GB_INPUT_SOURCE = r'''"""GB BIDS roster, SAS clinical fields and MNI lesion matching."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

ID_PATTERN = re.compile(r"c(\d+)p(\d+)", re.IGNORECASE)
IMAGE_PATTERN = re.compile(r"sub-([A-Za-z0-9]+)(?:_ses-([A-Za-z0-9]+))?(?:_[A-Za-z0-9]+-[A-Za-z0-9]+)*_(T1w|FLAIR)\.nii(?:\.gz)?")
LESION_SUFFIX = "_lesion_mask_to_MNI152_1mm.nii.gz"
METADATA_COLUMNS = ["participant_id", "age", "sex", "site_id"]
PARTICIPANT_COLUMNS = METADATA_COLUMNS + ["t1w", "flair", "lesion_mask"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ValueError("Refusing to write a symlink: {}".format(path))
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def id_key(value: str) -> Tuple[int, int]:
    text = value[4:] if value.startswith("sub-") else value
    match = ID_PATTERN.fullmatch(text)
    if not match:
        raise ValueError("Invalid GB ID: {!r}".format(value))
    return int(match[1]), int(match[2])


def code_number(value: object, prefix: str) -> int:
    if pd.isna(value):
        raise ValueError("Missing {}_code".format(prefix))
    text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    text = text.strip()
    if text[:1].lower() == prefix.lower():
        text = text[1:]
    if not re.fullmatch(r"\d+(?:\.0+)?", text):
        raise ValueError("Invalid {}_code: {!r}".format(prefix, value))
    return int(text.split(".")[0])


def find_sas(root: Path, configured: Optional[str] = None) -> Path:
    if configured:
        path = Path(configured).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file() or path.suffix.lower() != ".sas7bdat":
            raise ValueError("SAS file does not exist: {}".format(path))
        return path.resolve()
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".sas7bdat")
    if len(files) != 1:
        raise ValueError("Expected one project-root SAS file, found {}; use --sas-file".format(len(files)))
    return files[0].resolve()


def clinical_rows(source: pd.DataFrame, roster: Mapping[Tuple[int, int], str]) -> Tuple[Dict[str, dict], List[str], int]:
    required = ["AGE", "GENDER", "C_code", "P_code"]
    missing = [name for name in required if name not in source.columns]
    if missing:
        return {}, ["Missing SAS columns: " + repr(missing)], 0
    result: Dict[str, dict] = {}
    errors: List[str] = []
    extras = 0
    for index, row in source[required].iterrows():
        try:
            key = (code_number(row["C_code"], "c"), code_number(row["P_code"], "p"))
        except (ValueError, UnicodeDecodeError) as error:
            errors.append("SAS row {}: {}".format(index, error))
            continue
        if key not in roster:
            extras += 1
            continue
        subject = roster[key]
        if subject in result:
            errors.append("Duplicate SAS mapping: " + subject)
            continue
        try:
            age = float(row["AGE"])
            gender = float(row["GENDER"])
            if not math.isfinite(age) or not 0 < age <= 120 or gender not in {1, 2}:
                raise ValueError("AGE must be finite in (0,120]; GENDER must be 1 or 2")
        except (TypeError, ValueError) as error:
            errors.append("{}: {}".format(subject, error))
            continue
        result[subject] = {"participant_id": subject, "age": age,
                           "sex": {1: "male", 2: "female"}[gender],
                           "site_id": "c{:03d}".format(key[0])}
    return result, errors, extras


def scan_bids(root: Path) -> Tuple[Dict[Tuple[int, int], str], Dict[str, dict], List[str]]:
    if not root.is_dir():
        raise ValueError("BIDS directory does not exist: " + str(root))
    roster: Dict[Tuple[int, int], str] = {}
    images: Dict[str, dict] = {}
    errors: List[str] = []
    for directory in sorted(root.glob("sub-*")):
        if not directory.is_dir():
            continue
        subject = directory.name[4:]
        try:
            key = id_key(subject)
        except ValueError as error:
            errors.append(str(error))
            continue
        if key in roster:
            errors.append("BIDS IDs collapse to the same center/patient: {} / {}".format(roster[key], subject))
            continue
        roster[key] = subject
        modalities: Dict[str, list] = {"T1w": [], "FLAIR": []}
        sessions = {item.name[4:] for item in directory.glob("ses-*") if item.is_dir()}
        for path in sorted(directory.rglob("*.nii*")):
            if path.parent.name != "anat" or "derivatives" in path.relative_to(directory).parts:
                continue
            match = IMAGE_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            if match[1] != subject:
                errors.append("BIDS folder/file ID mismatch: " + str(path))
                continue
            folder_session = next((part[4:] for part in path.relative_to(directory).parts if part.startswith("ses-")), "")
            if match[2] and folder_session and match[2] != folder_session:
                errors.append("BIDS folder/file session mismatch: " + str(path))
            sessions.add(match[2] or folder_session)
            modalities[match[3]].append(path.resolve())
        if len(sessions) > 1:
            errors.append("Multiple BIDS sessions: " + subject)
        images[subject] = {}
        for modality, paths in modalities.items():
            if len(paths) != 1:
                errors.append("{} {} count={} (expected 1)".format(subject, modality, len(paths)))
            else:
                images[subject][modality] = paths[0]
    if not roster:
        errors.append("No GB BIDS subjects")
    return roster, images, errors


def _absolute(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()



def select_complete_cases(roster, matched_rows, errors):
    """Exclude identifiable incomplete/ambiguous records and retain every reason."""
    reasons = {subject: [] for subject in roster.values()}
    blocking = []
    token = re.compile(r"(?<![A-Za-z0-9])c\d+p\d+(?![A-Za-z0-9])", re.IGNORECASE)
    for error in errors:
        affected = set()
        if not error.startswith(("BIDS IDs collapse", "Invalid GB ID")):
            for value in token.findall(error):
                subject = roster.get(id_key(value))
                if subject is not None:
                    affected.add(subject)
        if not affected:
            blocking.append(error)
        for subject in affected:
            reasons[subject].append(error)
    included = [row for row in matched_rows if not reasons[row["participant_id"]]]
    excluded = [{"participant_id": subject, "reasons": " | ".join(values)}
                for subject, values in sorted(reasons.items()) if values]
    if not included:
        blocking.append("No complete, uniquely matched GB participants")
    return included, excluded, blocking


def prepare_gb_inputs(config: Mapping[str, object]) -> Dict[str, object]:
    root = Path(str(config["project_root"])).resolve()
    inputs = config["input"]
    output = _absolute(root, config["derivatives"]) / "audit"
    errors: List[str] = []
    try:
        roster, images, errors = scan_bids(_absolute(root, inputs["bids_root"]))
    except ValueError as error:
        roster, images = {}, {}
        errors.append(str(error))
    lesions: Dict[str, Path] = {}
    lesion_root = _absolute(root, inputs["lesion_root"])
    extra_lesions = 0
    if not lesion_root.is_dir():
        errors.append("Lesion directory does not exist: " + str(lesion_root))
    else:
        for path in sorted(lesion_root.rglob("*" + LESION_SUFFIX)):
            try:
                key = id_key(path.name[:-len(LESION_SUFFIX)])
            except ValueError as error:
                errors.append(str(error))
                continue
            if key not in roster:
                extra_lesions += 1
            elif roster[key] in lesions:
                errors.append("Duplicate lesion mapping: " + roster[key])
            else:
                lesions[roster[key]] = path.resolve()
    clinical: Dict[str, dict] = {}
    extra_clinical = 0
    sas_path = None
    try:
        sas_path = find_sas(root, inputs.get("sas_file"))
        clinical, clinical_errors, extra_clinical = clinical_rows(pd.read_sas(sas_path, format="sas7bdat"), roster)
        errors.extend(clinical_errors)
    except (OSError, ValueError, UnicodeError) as error:
        errors.append(str(error))
    matched_rows = []
    for subject in sorted(roster.values()):
        if subject not in lesions:
            errors.append("Missing lesion: " + subject)
        if subject not in clinical:
            errors.append("Missing valid SAS record: " + subject)
        matched_rows.append({"participant_id": subject, **clinical.get(subject, {}),
                             "t1w": str(images.get(subject, {}).get("T1w", "")),
                             "flair": str(images.get(subject, {}).get("FLAIR", "")),
                             "lesion_mask": str(lesions.get(subject, ""))})
    report = {"status": "fail" if errors else "pass", "bids_subject_count": len(roster),
              "matched_clinical_count": len(clinical), "matched_lesion_count": len(lesions),
              "extra_sas_rows": extra_clinical, "extra_lesion_files": extra_lesions,
              "sas_file": str(sas_path) if sas_path else None,
              "sas_sha256": sha256(sas_path) if sas_path else None,
              "cohort_filter": "BIDS roster; no C3 disease filters", "errors": errors,
              "raw_inputs_modified": False}
    atomic_text(output / "gb_input_matching.tsv", pd.DataFrame(matched_rows).to_csv(sep="\t", index=False))
    report["input_errors"] = list(errors)
    report["selection_policy"] = "complete_cases" if inputs.get("complete_cases", False) else "strict_all"
    report["source_hashes_deferred"] = os.environ.get("GB_PREPARE_METADATA_ONLY") == "1"
    report["t1_matched_subjects"] = sum("T1w" in images.get(s, {}) for s in roster.values())
    report["flair_matched_subjects"] = sum("FLAIR" in images.get(s, {}) for s in roster.values())
    report["t1_flair_matched_subjects"] = sum(all(m in images.get(s, {}) for m in ("T1w", "FLAIR")) for s in roster.values())
    excluded = []
    if inputs.get("complete_cases", False):
        matched_rows, excluded, errors = select_complete_cases(roster, matched_rows, errors)
        selected = {row["participant_id"] for row in matched_rows}
        roster = {key: subject for key, subject in roster.items() if subject in selected}
    report.update(status="fail" if errors else "pass", errors=errors,
                  selected_subject_count=len(matched_rows), excluded_subject_count=len(excluded))
    atomic_text(output / "gb_excluded_subjects.tsv",
                pd.DataFrame(excluded, columns=["participant_id", "reasons"]).to_csv(sep="\t", index=False))
    atomic_json(output / "gb_input_matching.json", report)
    counts = {key: report[key] for key in ("bids_subject_count", "t1_matched_subjects", "flair_matched_subjects",
              "t1_flair_matched_subjects", "matched_clinical_count", "matched_lesion_count",
              "selected_subject_count", "excluded_subject_count")}
    atomic_json(output / "gb_cohort_counts.json", counts)
    print("GB_COUNTS " + json.dumps(counts, ensure_ascii=False), flush=True)
    if errors:
        raise ValueError("GB input matching failed; see {}\n{}".format(output / "gb_input_matching.json", "\n".join(errors[:20])))
    # Validate all link collisions before creating any views or replacing tables.
    links = []
    view = _absolute(root, inputs.get("bids_links", "inputs/bids_links"))
    for row in matched_rows:
        for column, modality in (("t1w", "T1w"), ("flair", "FLAIR")):
            source = Path(row[column])
            extension = ".nii.gz" if source.name.endswith(".nii.gz") else ".nii"
            link = view / ("sub-" + row["participant_id"]) / "anat" / ("sub-" + row["participant_id"] + "_" + modality + extension)
            if os.path.lexists(str(link)) and (not link.is_symlink() or link.resolve() != source):
                report["errors"].append("Input view collision: " + str(link))
            links.append((row, column, source, link))
    if report["errors"]:
        report["status"] = "fail"
        atomic_json(output / "gb_input_matching.json", report)
        raise ValueError("Input view collision; see " + str(output / "gb_input_matching.json"))
    manifest = []
    for row, column, source, link in links:
        manifest.append({"participant_id": row["participant_id"], "modality": column,
                         "source": str(source), "source_sha256": sha256(source) if not report["source_hashes_deferred"] else "", "link": str(link)})
    for row in matched_rows:
        source = Path(row["lesion_mask"])
        manifest.append({"participant_id": row["participant_id"], "modality": "lesion_MNI152",
                         "source": str(source), "source_sha256": sha256(source) if not report["source_hashes_deferred"] else "", "link": ""})
    for row, column, source, link in links:
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(source)
        row[column] = str(link)
    frame = pd.DataFrame(matched_rows, columns=PARTICIPANT_COLUMNS)
    atomic_text(_absolute(root, inputs["metadata_tsv"]), frame[METADATA_COLUMNS].to_csv(sep="\t", index=False))
    atomic_text(_absolute(root, config["participants"]), frame.to_csv(sep="\t", index=False))
    atomic_text(view / "input_manifest.tsv", pd.DataFrame(manifest).to_csv(sep="\t", index=False))
    print("GB_INPUTS subjects={} clinical={} lesions={} source_files_hashed={}".format(
        len(roster), len(roster), len(roster), sum(bool(item["source_sha256"]) for item in manifest)), flush=True)
    queue = root / "logs/gb"
    queue.mkdir(parents=True, exist_ok=True)
    manifest = queue / "rerun_manifest.json"
    atomic_json(manifest, {"project_root": str(root), "cohort": "GB", "queue_participants": sorted(roster.values()),
        "grid_failure_participants": [], "representative_participants": [],
        "cpu_cores": 50, "cpu_threads_per_job": 4, "registration_threads": 8,
        "gpu_devices": [0, 1], "slots_per_gpu": 2, "rolling_window": 200})
    atomic_text(queue / "rerun_manifest.path", str(manifest) + "\n")
    return report
'''

INPUT_EXTENSION = r'''

def check_pilot(config):
    """Verify the computed pilot before opening the existing rolling queue."""
    import nibabel as nib
    import numpy as np
    from .schema import load_participants

    root = Path(str(config["project_root"])).resolve()
    output = _absolute(root, config["derivatives"])
    record = output / "audit/gb_pilot_validation.json"
    atomic_json(record, {"status": "fail", "participant_id": "c001p00016"})
    people = {person.participant_id: person for person in load_participants(_absolute(root, config["participants"]), root)}
    person = people.get("c001p00016")
    if person is None:
        raise ValueError("Pilot c001p00016 is absent from the BIDS cohort")
    subject = output / person.bids_id
    states = {}
    for stage in ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc", "cleanup"):
        state = json.loads((subject / "status" / (stage + ".json")).read_text())
        if state.get("status") != "pass" or state.get("participant_id") != person.participant_id:
            raise ValueError("Pilot stage/identity failed: " + stage)
        states[stage] = state
    wmh = states["wmh"]["details"]
    t1 = states["t1"]["details"]
    grid = wmh.get("wmh_atlas_grid", {})
    if not grid.get("matches") or float(grid.get("max_corner_displacement_mm", 1)) >= 0.05:
        raise ValueError("Pilot WMH atlas physical grid failed")
    paths = [Path(wmh["corrected_wmh"]), Path(t1["segmentation"]), Path(t1["macro20_segmentation"])]
    for path in paths:
        if not path.is_file() or output.resolve() not in path.resolve().parents:
            raise ValueError("Pilot final output missing/outside result directory: " + str(path))
        data = np.asanyarray(nib.load(str(path)).dataobj)
        if data.ndim != 3 or not np.isfinite(data).all():
            raise ValueError("Pilot segmentation invalid: " + str(path))
        if path != paths[0] and not np.any(data > 0):
            raise ValueError("Pilot T1 segmentation is empty: " + str(path))
        if path == paths[0] and not set(np.unique(data)).issubset({0, 1}):
            raise ValueError("Pilot corrected WMH is not binary")
    for details, keys in ((wmh, ("raw_ml", "z_chung")), (t1, ("macro20_ml", "macro20_atrophy_z"))):
        feature_path = Path(details["feature_json"])
        feature = json.loads(feature_path.read_text())
        if feature.get("participant_id") != person.participant_id:
            raise ValueError("Pilot feature ID mismatch")
        for key in keys:
            values = feature.get(key, {})
            if len(values) != 20 or not all(np.isfinite(float(value)) for value in values.values()):
                raise ValueError("Pilot does not have finite 20 + 20 features: " + key)
    figures = list((output / "qc").glob(person.participant_id + "_*.png"))
    if states["qc"]["details"].get("figure_count") != 4 or len(figures) != 4 or any(path.stat().st_size == 0 for path in figures):
        raise ValueError("Pilot requires four nonempty QC figures")
    from PIL import Image
    for path in figures:
        with Image.open(str(path)) as image:
            image.verify()
    atomic_json(record, {"status": "pass", "participant_id": person.participant_id,
        "wmh_feature_count": 20, "t1_feature_count": 20, "qc_figures": [str(path) for path in figures],
        "final_outputs": [str(path) for path in paths], "manual_qc_performed": False})
    print("GB pilot c001p00016 passed: final masks, 20+20 features, 4 QC figures, 8 stages", flush=True)


def summarize(config):
    root = Path(str(config["project_root"])).resolve()
    output = _absolute(root, config["derivatives"])
    participants = pd.read_csv(_absolute(root, config["participants"]), sep="\t", dtype=str)["participant_id"].tolist()
    counts = {"cohort": len(participants), "success": 0, "failed": 0, "missing": 0}
    rows = []
    for subject in participants:
        states = []
        for stage in ("skullstrip", "wmh_seg", "registration", "lesion", "wmh", "t1", "qc", "cleanup"):
            path = output / ("sub-" + subject) / "status" / (stage + ".json")
            states.append(json.loads(path.read_text()).get("status") if path.is_file() else "missing")
        status = "success" if all(value == "pass" for value in states) else "failed" if "fail" in states else "missing"
        counts[status] += 1
        rows.append({"participant_id": subject, "status": status})
    table = pd.read_csv(output / "tables/features_computed40.tsv", sep="\t", dtype={"participant_id": str})
    if len(table.columns) != 41 or table["participant_id"].duplicated().any() or not set(table["participant_id"]).issubset(participants):
        raise ValueError("Computed40 table identity/column contract failed")
    counts["computed40"] = len(table)
    atomic_text(output / "tables/gb_run_summary.tsv", pd.DataFrame(rows).to_csv(sep="\t", index=False))
    atomic_json(output / "tables/gb_run_summary.json", counts)
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    return counts
'''

GB_INPUT_SOURCE += INPUT_EXTENSION

if __name__ == '__main__':
    main()
