#!/usr/bin/env python3
"""Isolated LS orchestration; never rewrite the installed GB analysis framework."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import yaml

DEFAULT_ROOT = Path('/data02/linzhenzong/Substain_GB')
DEFAULT_INPUT = Path('/data02/linzhenzong/LS')
DEFAULT_OUTPUT = Path('/mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_LS/derivatives')
STAGES = ('skullstrip', 'wmh_seg', 'registration', 'lesion', 'wmh', 't1', 'qc', 'cleanup')


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def write_same(path, content):
    """Atomic, content-stable write; do not follow destination symlinks."""
    path = Path(path)
    data = content.encode('utf-8') if isinstance(content, str) else content
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Refusing output through a symlink: ' + str(path))
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix='.LS-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    os.replace(str(temporary), str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def absolute(root, value):
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def process_identity(pid, proc=Path('/proc')):
    folder = proc / str(pid)
    try:
        raw = (folder / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': int(pid), 'pgid': int(fields[2]), 'start_ticks': int(fields[19]),
                'state': fields[0], 'argv': (folder / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],
                'cwd': str((folder / 'cwd').resolve())}
    except FileNotFoundError:
        return None


def require_idle(root, own_pid=None, proc=Path('/proc')):
    """Read process identity, including relative CLI invocations; never kill anything."""
    root = Path(root).resolve()
    own_pid = os.getpid() if own_pid is None else own_pid
    if not proc.is_dir():
        raise RuntimeError('Linux /proc is required for safe workstation process checks')
    busy = []
    for folder in proc.iterdir():
        if not folder.name.isdigit() or int(folder.name) == own_pid:
            continue
        try:
            item = process_identity(int(folder.name), proc)
        except PermissionError:
            # Other users are not ours to manage; our own process metadata must be readable.
            if folder.stat().st_uid == os.getuid():
                raise RuntimeError('Cannot inspect same-user process ' + folder.name)
            continue
        if not item or item['state'] == 'Z':
            continue
        argv = item['argv']
        joined = ' '.join(argv)
        in_project = str(root) in joined or item['cwd'] == str(root) or item['cwd'].startswith(str(root) + '/')
        if not in_project:
            continue
        pipeline = (any(Path(token).name == 'snakemake' for token in argv) or 'substain_features.cli' in argv or
                    any(Path(token).name in ('finish_ws1_v1_0_9.sh', 'gb_verify_then_all.sh', 'run_pipeline.sh') for token in argv))
        ls_worker = any(Path(token).name == 'runner_LS.py' for token in argv) and 'worker' in argv
        if pipeline or ls_worker:
            busy.append({'pid': item['pid'], 'argv': argv})
    if busy:
        raise RuntimeError('Existing GB/LS analysis is active; nothing stopped: ' + canonical(busy))


def config_for(root: Path, input_root: Path, derivatives: Path) -> dict:
    root = root.resolve(strict=True)
    base_path = root / 'config/config.yaml'
    base = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    old_root = absolute(base_path.parent, base['project_root'])
    if old_root != root:
        raise ValueError('GB project_root differs from the requested installed project')
    cfg = copy.deepcopy(base)
    execution = cfg['execution']
    required = ('cpu_cores', 'cpu_threads_per_job', 'max_parallel_jobs', 'gpu_devices', 'gpu_slots_per_device')
    missing = [key for key in required if key not in execution]
    if missing:
        raise ValueError('Cannot infer current workstation resource policy; missing: ' + ', '.join(missing))
    devices = [str(x) for x in execution['gpu_devices']]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Invalid inherited GPU devices')
    for key in ('cpu_cores', 'cpu_threads_per_job', 'max_parallel_jobs', 'gpu_slots_per_device'):
        if int(execution[key]) < 1:
            raise ValueError('Nonpositive resource setting: ' + key)
    gb_output = absolute(root, base['derivatives'])
    derivatives = derivatives.resolve()
    input_root = input_root.resolve()
    if (derivatives == gb_output or gb_output in derivatives.parents or derivatives in gb_output.parents
            or derivatives == input_root or input_root in derivatives.parents or derivatives in input_root.parents):
        raise ValueError('LS output must not overlap GB output or raw LS inputs')
    cfg.update(project_root=str(root), participants=str(root / 'config/participants_LS.tsv'),
               derivatives=str(derivatives), qc_dir=str(derivatives / 'qc_LS'), cohort_suffix='_LS')
    cfg['input'] = {'cohort': 'LS', 'mode': 'ls_healthy', 'bids_links': str(root / 'inputs/cohort_LS'),
                    'metadata_tsv': str(root / 'inputs/cohort_LS/metadata_LS.tsv')}
    # Resolve against the GB root before changing any working directory.
    execution['gpu_lock_dir'] = str(absolute(root, execution['gpu_lock_dir']))
    for key in ('core_python', 'wmh_python', 't1_python'):
        # Resolving a venv's python symlink selects the system environment instead.
        candidate = Path(str(execution[key]))
        execution[key] = os.path.abspath(str(candidate if candidate.is_absolute() else root / candidate))
    execution['ants_bin'] = str(absolute(root, execution['ants_bin']))
    cfg['ls'] = {
        'input_root': str(input_root), 'clinical_file': str(input_root / 'LS_INFO.xlsx'),
        'preparation_dir': str(root / 'inputs/cohort_LS'),
        'template': str(absolute(root, cfg['wmh']['fsl_mni152_1mm'])),
        'log_dir': str(root / 'logs/LS'), 'runtime_dir': str(root / 'runtime_LS'),
        'audit_guard_path': str(root / 'logs/LS/audit_guard_LS.json'),
        'resource_manifest_path': str(derivatives / 'tables/resource_manifest_LS.tsv'),
        'healthy_no_acute_lesion': True, 'lesion_source_kind': 'synthetic_zero_mask',
        'lesion_correction_applied': False, 'site_id': 'LS', 'qc_port': 8766,
        'source_config_sha256': digest(base_path),
    }
    return cfg


def load_config(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = Path(cfg['project_root'])
    if path != root / 'config/config_LS.yaml' or cfg.get('cohort_suffix') != '_LS':
        raise ValueError('Only the installed absolute config_LS.yaml is accepted')
    if cfg.get('input', {}).get('cohort') != 'LS' or not cfg['ls'].get('healthy_no_acute_lesion'):
        raise ValueError('Expected explicitly confirmed healthy LS cohort')
    if Path(cfg['participants']) != root / 'config/participants_LS.tsv':
        raise ValueError('Invalid LS participants path')
    receipt = root / 'logs/LS/installation_LS.json'
    if receipt.is_file() and read_json(receipt).get('config_sha256') != digest(path):
        raise ValueError('Installed LS configuration was modified; no automatic adoption')
    return cfg


def env_for(cfg: dict) -> dict:
    runtime = Path(cfg['ls']['runtime_dir'])
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env['PYTHONPATH'] = str(runtime / 'src') + os.pathsep + str(HERE / 'vendor')
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in cfg['execution']['gpu_devices'])
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    threads = str(cfg['execution']['cpu_threads_per_job'])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        env[key] = threads
    for key in ('TMPDIR', 'MPLCONFIGDIR'):
        target = runtime / ('tmp_LS' if key == 'TMPDIR' else 'mpl_cache_LS')
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    ants_lib = Path(cfg['execution']['ants_bin']).parent / 'lib'
    env['LD_LIBRARY_PATH'] = str(ants_lib) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['HF_HUB_DISABLE_TELEMETRY'] = '1'
    env['SUBSTAIN_OFFLINE_VERIFY_ROOT'] = str(runtime / 'verification_LS')
    return env


def environment_record(cfg: dict) -> dict:
    code = """import importlib.metadata as m,importlib.util as u,json,sys
record={'python':sys.version,'executable':sys.executable,'packages':dict(sorted((d.metadata['Name'],d.version) for d in m.distributions() if d.metadata.get('Name'))),'models':[]}
for name,dataset in [('DLMUSE','Dataset903_Task903_DLMUSEV2'),('DLICV','Dataset901_Task901_dlicv')]:
 s=u.find_spec(name)
 if s and s.origin:
  from pathlib import Path
  folder=Path(s.origin).parent/'nnunet_results'/dataset/'nnUNetTrainer__nnUNetPlans__3d_fullres'
  record['models'] += [str(folder/p) for p in ['dataset.json','plans.json','fold_0/checkpoint_final.pth']]
if u.find_spec('torch'):
 import torch
 record.update(torch_version=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())
print(json.dumps(record,sort_keys=True))
"""
    result = {}
    for label in ('core', 'wmh', 't1'):
        executable = cfg['execution'][label + '_python']
        run = subprocess.run([executable, '-c', code], env=env_for(cfg), capture_output=True, text=True, check=True)
        item = json.loads(run.stdout.strip().splitlines()[-1])
        if label in ('wmh', 't1') and (not item.get('cuda_build') or not item.get('cuda_available')):
            print('LS environment: {} CUDA unavailable; retaining the original stage fallback policy'.format(label), flush=True)
        if label == 't1' and len(item['models']) != 6:
            raise RuntimeError('Cannot locate both installed DLICV and DLMUSE model files')
        result[label] = item
    return result


def resource_paths(cfg, environment):
    root = Path(cfg['project_root'])
    paths = set()
    for group in ('wmh', 't1', 'synthstrip'):
        for value in cfg.get(group, {}).values():
            if isinstance(value, str):
                path = absolute(root, value)
                if path.is_file():
                    paths.add(path)
    # Actual runtime code, installed templates/models/mappings; never GB config/participants.
    for name in ('resources/runtime', 'resources/models', 'resources/templates', 'resources/mappings'):
        folder = root / name
        if folder.is_dir():
            paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for item in environment.values():
        paths.update(Path(p) for p in item['models'])
    for name in ('antsRegistrationSyNQuick.sh', 'antsRegistration', 'antsApplyTransforms'):
        paths.add(Path(cfg['execution']['ants_bin']) / name)
    paths.add(root / 'scripts/rolling_admission.py')
    return sorted(paths, key=str)


def freeze_resources(cfg, environment):
    path = Path(cfg['ls']['resource_manifest_path'])
    rows = []
    root = Path(cfg['project_root'])
    for item in resource_paths(cfg, environment):
        if not item.is_file():
            raise FileNotFoundError('Required shared resource: ' + str(item))
        rows.append({'relative_path': os.path.relpath(str(item), str(root)),
                     'size_bytes': item.stat().st_size, 'sha256': digest(item)})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=('relative_path', 'size_bytes', 'sha256'), delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue()
    if path.exists() and path.read_text() != data:
        raise RuntimeError('Resource baseline changed; existing LS manifest was preserved')
    write_same(path, data)
    write_same(Path(cfg['ls']['log_dir']) / 'environment_LS.json', canonical(environment))
    return rows


def verify_resources(cfg):
    root = Path(cfg['project_root'])
    with Path(cfg['ls']['resource_manifest_path']).open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    errors = [row['relative_path'] for row in rows if not (root / row['relative_path']).is_file()
              or digest(root / row['relative_path']) != row['sha256']]
    if errors:
        raise RuntimeError('Shared resource identity changed: ' + ', '.join(errors[:20]))


def runtime_sources() -> list:
    """Only runtime code and vendored dependencies enter the installed identity."""
    paths = [HERE / name for name in ('inputs_LS.py', 'monitor_LS.py', 'overlay_LS.py', 'runner_LS.py')]
    vendor = HERE / 'vendor'
    if not vendor.is_dir() or vendor.is_symlink():
        raise ValueError('Missing regular vendor directory: ' + str(vendor))
    paths.extend(path for path in vendor.rglob('*') if path.is_file()
                 and not any(part in ('__pycache__', '.pytest_cache', '.ruff_cache') for part in path.parts)
                 and path.suffix not in ('.pyc', '.pyo'))
    for path in paths:
        if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError('Missing or symlinked LS runtime source: ' + str(path))
    return sorted(paths, key=lambda path: path.relative_to(HERE).as_posix())


def install(root: Path, input_root: Path, derivatives: Path) -> None:
    from overlay_LS import build_overlay
    require_idle(root)
    cfg = config_for(root, input_root, derivatives)
    config_path = root.resolve() / 'config/config_LS.yaml'
    serialized = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    if config_path.exists() and config_path.read_text() != serialized:
        raise RuntimeError('Existing config_LS.yaml differs; no overwrite')
    for key in ('input_root', 'clinical_file', 'template'):
        if not Path(cfg['ls'][key]).exists():
            raise FileNotFoundError(cfg['ls'][key])
    destination = root.resolve() / 'scripts/cohort_LS'
    sources = runtime_sources()
    for source in sources:
        target = destination / source.relative_to(HERE)
        if target.exists() and digest(target) != digest(source):
            raise RuntimeError('Installed LS adapter differs; no overwrite: ' + str(target))
    # Only LS paths are created. Shared GB source and configuration remain byte-identical.
    print('LS install: capture current GB configuration and create isolated source overlay', flush=True)
    write_same(config_path, serialized)
    build_overlay(root.resolve(), config_path, Path(cfg['ls']['runtime_dir']))
    print('LS install: inspect existing Python/CUDA environments; no package installation', flush=True)
    environment = environment_record(cfg)
    print('LS install: hash shared models/templates and freeze the LS resource baseline', flush=True)
    freeze_resources(cfg, environment)
    for source in sources:
        write_same(destination / source.relative_to(HERE), source.read_bytes())
    wrappers = {'run_LS.sh': 'first', 'resume_LS.sh': 'resume', 'prepare_LS.sh': 'prepare',
                'verify_LS.sh': 'verify', 'export_LS.sh': 'export', 'qc_LS.sh': 'qc'}
    import shlex
    for name, action in wrappers.items():
        body = '#!/usr/bin/env bash\nset -euo pipefail\nexec ' + ' '.join(shlex.quote(x) for x in (
            cfg['execution']['core_python'], str(destination / 'runner_LS.py'), action,
            '--config', str(config_path))) + ' "$@"\n'
        write_same(root / 'scripts' / name, body)
    body = '#!/usr/bin/env bash\nset -euo pipefail\nexec ' + ' '.join(shlex.quote(x) for x in (
        cfg['execution']['core_python'], str(destination / 'monitor_LS.py'), '--config', str(config_path))) + ' "$@"\n'
    write_same(root / 'scripts/monitor_LS.sh', body)
    write_same(Path(cfg['ls']['log_dir']) / 'installation_LS.json', canonical({
        'status': 'installed', 'config': str(config_path), 'config_sha256': digest(config_path),
        'source_config_sha256': cfg['ls']['source_config_sha256'], 'shared_code_modified': False,
        'execution': cfg['execution'], 'derivatives': cfg['derivatives']}))
    print('LS_INSTALLED config=' + str(config_path), flush=True)
    print('Next: bash ' + str(root / 'scripts/run_LS.sh'), flush=True)


def participants(cfg):
    with Path(cfg['participants']).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def capture_identity(cfg: dict, config_path: Path) -> dict:
    root = Path(cfg['project_root'])
    files = {Path(config_path), Path(cfg['participants']), Path(cfg['ls']['clinical_file']),
             Path(cfg['ls']['resource_manifest_path']), Path(cfg['ls']['log_dir']) / 'environment_LS.json'}
    for row in participants(cfg):
        files.update(Path(row[key]) for key in ('t1w', 'flair', 'lesion_mask'))
    runtime = Path(cfg['ls']['runtime_dir'])
    prep = Path(cfg['ls']['preparation_dir'])
    files.update(p for p in prep.glob('*_LS.*') if p.is_file())
    files.add(runtime / 'overlay_manifest_LS.json')
    for folder in (runtime / 'src', runtime / 'workflow', root / 'scripts/cohort_LS'):
        files.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    # Record the actual installed source too: never silently adopt a later algorithm update.
    files.update((root / 'src/substain_features').glob('*.py'))
    files.add(root / 'workflow/Snakefile')
    snapshot = {str(path): digest(path) for path in sorted(files, key=str)}
    discovered = {}
    for folder in sorted(Path(cfg['ls']['input_root']).glob('P*')):
        if folder.is_dir():
            discovered[folder.name] = sorted(str(p.relative_to(folder)) for pattern in ('*T1w*.nii.gz', '*T2wFLAIR*.nii.gz')
                                             for p in folder.glob(pattern))
    return {'schema': 1, 'files': snapshot, 'discovery': discovered, 'cohort': 'LS', 'healthy_no_acute_lesion': True}


def check_identity(cfg: dict, config_path: Path) -> str:
    recorded = read_json(Path(cfg['ls']['log_dir']) / 'run_manifest_LS.json')
    current = capture_identity(cfg, config_path)
    if recorded != current:
        before, after = recorded.get('files', {}), current['files']
        differences = [name for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)]
        if recorded.get('discovery') != current.get('discovery'):
            differences.append('LS image-directory inventory')
        raise RuntimeError('LS run identity changed; existing outputs preserved: ' + ', '.join(differences[:20]))
    verify_resources(cfg)
    return digest(Path(cfg['ls']['log_dir']) / 'run_manifest_LS.json')


def cli(cfg, config_path, command, *extra):
    args = [cfg['execution']['core_python'], '-m', 'substain_features.cli'] + command.split()
    args += ['--config-file', str(config_path)] + list(extra)
    subprocess.run(args, cwd=cfg['ls']['runtime_dir'], env=env_for(cfg), check=True)


def prepare(cfg: dict, config_path: Path) -> None:
    from inputs_LS import prepare_inputs
    log_dir = Path(cfg['ls']['log_dir'])
    manifest = log_dir / 'run_manifest_LS.json'
    if manifest.exists():
        check_identity(cfg, config_path)
        print('LS preparation already frozen; unchanged roster reused', flush=True)
        return
    if any(Path(cfg['derivatives']).glob('sub-*/status/*.json')):
        raise RuntimeError('Existing LS subject results have no run manifest; refusing to adopt them')
    print('LS prepare: read Excel, match all image directories and hash inputs', flush=True)
    result = prepare_inputs(cfg)
    print('LS prepare: participants={} eligible={} ineligible={}'.format(
        result['participants_count'], result['normative_eligible_count'], result['normative_ineligible_count']), flush=True)
    verify_resources(cfg)
    write_same(manifest, canonical(capture_identity(cfg, config_path)))


def ensure_audit(cfg, config_path, identity):
    guard = Path(cfg['ls']['audit_guard_path'])
    if guard.exists():
        record = read_json(guard)
        if record.get('status') == 'pass' and record.get('run_manifest_sha256') == identity:
            report = Path(record['report'])
            if report.is_file() and digest(report) == record['report_sha256']:
                print('LS audit reused: input/config identity unchanged', flush=True)
                return
        raise RuntimeError('LS audit guard no longer matches; no automatic re-audit or overwrite')
    print('LS audit: first full input/grid audit; completion report will be reusable', flush=True)
    cli(cfg, config_path, 'audit', '--participant-id', 'all')
    report = Path(cfg['derivatives']) / 'audit/audit_report_LS.json'
    if read_json(report).get('status') != 'pass':
        raise RuntimeError('LS audit failed')
    write_same(guard, canonical({'status': 'pass', 'run_manifest_sha256': identity,
                               'report': str(report), 'report_sha256': digest(report)}))


def snakemake_command(cfg, config_path, targets, rolling=None, dry_run=False):
    runtime = Path(cfg['ls']['runtime_dir'])
    execution = cfg['execution']
    devices = [str(value) for value in execution['gpu_devices']]
    slots = int(execution['gpu_slots_per_device'])
    args = [execution['core_python'], '-m', 'snakemake', '--snakefile', str(runtime / 'workflow/Snakefile_LS'),
            '--configfile', str(config_path), '--cores', str(execution['cpu_cores']),
            '--keep-going', '--printshellcmds']
    if dry_run:
        args.append('--dry-run')
    args += list(targets)
    args += ['--resources', 'gpu=' + str(len(devices) * slots), 'wmh_exclusive=' + str(len(devices))]
    if 'max_compute_jobs' in execution:
        args.append('compute_jobs=' + str(execution['max_compute_jobs']))
    args += ['--config', 'active_config_file=' + str(config_path), 'selected_participant=all',
             'profile=gpu', 'gpu_devices=' + ','.join(devices), 'gpu_slots_per_device=' + str(slots)]
    if rolling:
        args += ['rolling_order_file=' + str(rolling[0]), 'rolling_token_dir=' + str(rolling[1]),
                 'rolling_window=' + str(execution['max_parallel_jobs']), 'rolling_poll_seconds=10']
    return args


def checked_status(cfg, pid, stage):
    path = Path(cfg['derivatives']) / ('sub-' + pid) / 'status' / (stage + '.json')
    if not path.exists():
        return None
    item = read_json(path)
    if item.get('participant_id') != pid or item.get('stage') != stage or item.get('status') not in ('pass', 'fail'):
        raise ValueError('Invalid LS status identity: ' + str(path))
    return item


def healthy_check(cfg: dict, pid: str, require_cleanup: bool = True) -> dict:
    import nibabel as nib
    import numpy as np
    states = {stage: checked_status(cfg, pid, stage) for stage in STAGES}
    required = STAGES if require_cleanup else STAGES[:-1]
    if any(not states[stage] or states[stage]['status'] != 'pass' for stage in required):
        raise ValueError('Required analysis stages did not pass: ' + pid)
    details = states['wmh']['details']
    for key in ('lesion_t1', 'lesion_flair'):
        image = nib.load(states['lesion']['details'][key])
        if np.any(np.asanyarray(image.dataobj) != 0):
            raise ValueError('Nonzero synthetic lesion after transform: ' + pid)
    original = nib.load(details['original_wmh'])
    corrected = nib.load(details['corrected_wmh'])
    segmentation = nib.load(states['wmh_seg']['details']['segmentation'])
    expected = np.rint(np.asanyarray(segmentation.dataobj)).astype(np.int32) == int(cfg['wmh']['lesion_label'])
    if not np.array_equal(np.asanyarray(original.dataobj) > 0, expected):
        raise ValueError('Original WMH differs from segmentation label: ' + pid)
    if (original.shape != corrected.shape or not np.allclose(original.affine, corrected.affine, atol=1e-5)
            or not np.array_equal(np.asanyarray(original.dataobj), np.asanyarray(corrected.dataobj))):
        raise ValueError('Healthy WMH identity failed: ' + pid)
    zero_keys = ('wmh_lesion_overlap_ml', 'wmh_contralateral_donor_ml', 'wmh_replacement_added_ml',
                 'wmh_original_overlap_removed_ml', 'wmh_bilateral_lesion_conflict_removed_ml',
                 'wmh_out_of_brain_donor_removed_ml')
    for key in zero_keys:
        if key not in details or float(details[key]) != 0:
            raise ValueError('Expected zero correction ' + key + ': ' + pid)
    for key in ('lesion_component_count', 'triggered_lesion_component_count'):
        if details.get(key) != 0:
            raise ValueError('Expected zero lesion components: ' + pid)
    wmh = read_json(details['feature_json'])
    t1 = read_json(states['t1']['details']['feature_json'])
    for key, data in (('raw_ml', wmh), ('z_chung', wmh), ('macro20_ml', t1), ('macro20_atrophy_z', t1)):
        values = data.get(key, {})
        if len(values) != 20 or any(not math.isfinite(float(x)) for x in values.values()):
            raise ValueError('Invalid 20-feature block ' + key + ': ' + pid)
    return {'participant_id': pid, 'status': 'pass', 'lesion_source_kind': 'synthetic_zero_mask',
            'lesion_correction_applied': False, 'wmh_identity': True, 'features': 40}


def choose_pilot(cfg):
    table = Path(cfg['ls']['preparation_dir']) / 'normative_eligibility_LS.tsv'
    with table.open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    eligible = sorted(row['participant_id'] for row in rows if str(row['eligible']).lower() in ('true', '1'))
    print('LS normative eligibility: eligible={} ineligible={}; all remain in roster'.format(len(eligible), len(rows) - len(eligible)), flush=True)
    if not eligible:
        raise RuntimeError('No normative-eligible LS participant; see normative_eligibility_LS.tsv')
    return 'P0001' if 'P0001' in eligible else eligible[0]


def run_queue(cfg, config_path):
    pending = []
    for row in participants(cfg):
        pid = row['participant_id']
        if checked_status(cfg, pid, 'cleanup') is None:
            pending.append(pid)
    if not pending:
        print('LS all participants already ended; no inference queued', flush=True)
        return
    queue = Path(cfg['ls']['log_dir']) / 'queue_LS'
    queue.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    order = queue / (stamp + '_order_LS.txt')
    tokens = queue / (stamp + '_admission_LS')
    write_same(order, ''.join(pid + '\n' for pid in pending))
    initialize = [cfg['execution']['core_python'], str(Path(cfg['project_root']) / 'scripts/rolling_admission.py'),
                  'initialize', '--order-file', str(order), '--token-dir', str(tokens),
                  '--window', str(cfg['execution']['max_parallel_jobs'])]
    subprocess.run(initialize, env=env_for(cfg), check=True)
    # Token filenames belong to the existing private per-job protocol, like subject status files.
    target = str(tokens / 'all_complete_LS.json')
    subprocess.run(snakemake_command(cfg, config_path, [target], (order, tokens)),
                   cwd=cfg['ls']['runtime_dir'], env=env_for(cfg), check=True)
    missing = [pid for pid in pending if checked_status(cfg, pid, 'cleanup') is None]
    if missing:
        raise RuntimeError('Queue returned without cleanup: ' + ', '.join(missing))


def summarize(cfg: dict) -> dict:
    records, good = [], []
    for row in participants(cfg):
        pid = row['participant_id']
        states = {stage: checked_status(cfg, pid, stage) for stage in STAGES}
        failures = [stage for stage, state in states.items() if state and state['status'] == 'fail']
        missing = [stage for stage, state in states.items() if state is None]
        reason = []
        for stage in failures:
            state = states[stage]
            reason.append(stage + ':' + str(state.get('details', {}).get('error', state.get('error', 'stage failed'))))
        status = 'failed' if failures else 'incomplete' if missing else 'pass'
        # Preserve the original aggregate contract: cleanup is operational,
        # whereas the first seven stages determine feature eligibility.
        computed_eligible = all(states[stage] and states[stage]['status'] == 'pass' for stage in STAGES[:-1])
        if computed_eligible:
            healthy_check(cfg, pid, require_cleanup=False)
            good.append(pid)
        records.append({'participant_id': pid, 'status': status, 'failed_stages': '|'.join(failures),
                        'missing_stages': '|'.join(missing), 'reason': ' | '.join(reason),
                        'computed40_exported': bool(computed_eligible)})
    tables = Path(cfg['derivatives']) / 'tables'
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=('participant_id', 'status', 'failed_stages', 'missing_stages', 'reason', 'computed40_exported'),
                            delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    write_same(tables / 'run_summary_LS.tsv', stream.getvalue())
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=('participant_id', 'status', 'failed_stages', 'missing_stages', 'reason', 'computed40_exported'),
                            delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(row for row in records if not row['computed40_exported'])
    write_same(tables / 'not_exported_LS.tsv', stream.getvalue())
    names = ('features_computed40_LS.tsv', 'wmh20_raw_LS.tsv', 'wmh20_z_chung_LS.tsv',
             't1_macro20_raw_LS.tsv', 't1_macro20_z_genmind_LS.tsv', 't1_muse145_raw_LS.tsv', 't1_gm119_raw_LS.tsv')
    counts = {}
    for name in names:
        with (tables / name).open() as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            values = list(reader)
            if [row['participant_id'] for row in values] != good:
                raise ValueError('Cohort table IDs/order differ from successful subjects: ' + name)
            if name == 'features_computed40_LS.tsv':
                if len(reader.fieldnames) != 41:
                    raise ValueError('Expected ID + exactly 40 features')
                for row in values:
                    if any(not math.isfinite(float(value)) for key, value in row.items() if key != 'participant_id'):
                        raise ValueError('Nonfinite/blank computed feature')
            counts[name] = len(values)
    summary = {'cohort': 'LS', 'total': len(records), 'success': sum(row['status'] == 'pass' for row in records),
               'computed40_rows': len(good),
               'computed_with_cleanup_issue': sum(row['computed40_exported'] and row['status'] != 'pass' for row in records),
               'failed': sum(row['status'] == 'failed' for row in records),
               'incomplete': sum(row['status'] == 'incomplete' for row in records),
               'table_rows': counts, 'healthy_wmh_identity_checked': len(good),
               'manual_qc_required_for_primary': True, 'derivatives': cfg['derivatives']}
    write_same(tables / 'run_summary_LS.json', canonical(summary))
    print(canonical(summary), flush=True)
    return summary


@contextlib.contextmanager
def locked(path, wait_seconds=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Another LS operation holds ' + str(path)) from exc
                time.sleep(0.1)
        yield


def worker(cfg, config_path, mode):
    log_dir = Path(cfg['ls']['log_dir'])
    with locked(log_dir / 'operation_LS.lock', wait_seconds=15):
        require_idle(Path(cfg['project_root']))
        if mode == 'first':
            prepare(cfg, config_path)
        print('LS resume identity: checking frozen inputs, code and resources (no inference or re-audit)', flush=True)
        identity = check_identity(cfg, config_path)
        ensure_audit(cfg, config_path, identity)
        pilot_file = log_dir / 'pilot_verification_LS.json'
        if not pilot_file.exists():
            if mode == 'resume':
                raise RuntimeError('Pilot has not passed; use run_LS.sh to continue first-run validation')
            pilot = choose_pilot(cfg)
            target = str(Path(cfg['derivatives']) / ('sub-' + pilot) / 'status/cleanup.json')
            subprocess.run(snakemake_command(cfg, config_path, [target]), cwd=cfg['ls']['runtime_dir'],
                           env=env_for(cfg), check=True)
            result = healthy_check(cfg, pilot)
            result['run_manifest_sha256'] = identity
            write_same(pilot_file, canonical(result))
        else:
            result = read_json(pilot_file)
            if result.get('status') != 'pass' or result.get('run_manifest_sha256') != identity:
                raise RuntimeError('Pilot record identity mismatch')
            healthy_check(cfg, result['participant_id'])
        print('LS pilot passed; starting/reusing full queue. Resources=' + canonical(cfg['execution']), flush=True)
        all_ended = all(checked_status(cfg, row['participant_id'], 'cleanup') is not None for row in participants(cfg))
        if all_ended and (Path(cfg['derivatives']) / 'tables/run_summary_LS.json').is_file():
            summarize(cfg)
            print('LS complete and verified; existing manual-QC export preserved', flush=True)
            return
        run_queue(cfg, config_path)
        primary = Path(cfg['derivatives']) / 'tables/features_primary40_LS.tsv'
        if primary.is_file():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup = Path(cfg['derivatives']) / 'exports_LS' / stamp / primary.name
            write_same(backup, primary.read_bytes())
            print('Preserved previous manual-QC export: ' + str(backup), flush=True)
        cli(cfg, config_path, 'stage aggregate', '--participant-id', 'all')
        summarize(cfg)


def launch(cfg, config_path, mode):
    log_dir = Path(cfg['ls']['log_dir'])
    with locked(log_dir / 'operation_LS.lock'):
        require_idle(Path(cfg['project_root']))
        if mode == 'resume' and not (log_dir / 'pilot_verification_LS.json').is_file():
            raise RuntimeError('No passed pilot record; use first-run entrypoint')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        log = log_dir / (stamp + '_full_run_LS.log')
        installed = Path(cfg['project_root']) / 'scripts/cohort_LS/runner_LS.py'
        args = [cfg['execution']['core_python'], str(installed), 'worker', '--config', str(config_path), '--mode', mode]
        with log.open('xb') as output:
            process = subprocess.Popen(args, cwd=cfg['ls']['runtime_dir'], env=env_for(cfg),
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        item = process_identity(process.pid)
        if item is None:
            raise RuntimeError('LS worker exited immediately; see ' + str(log))
        pointer = {key: item[key] for key in ('pid', 'pgid', 'start_ticks')}
        pointer.update(config=str(config_path), runner=str(installed), mode=mode, started_at=stamp)
        write_same(log_dir / 'full_run_LS.json', canonical(pointer))
        write_same(log_dir / 'full_run_LS.pid', str(process.pid) + '\n')
        write_same(log_dir / 'full_run_LS.logpath', str(log) + '\n')
        print('LS_STARTED PID={} log={}'.format(process.pid, log), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'prepare', 'first', 'resume', 'worker', 'verify', 'export', 'qc'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--derivatives', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--mode', choices=('first', 'resume'), default='first')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args(argv)
    cfg = None
    try:
        os.umask(0o022)
        if args.action == 'install':
            with locked(args.root / 'logs/LS/operation_LS.lock'):
                install(args.root, args.input_root, args.derivatives)
            return 0
        config_path = (args.config or args.root / 'config/config_LS.yaml').resolve()
        cfg = load_config(config_path)
        if args.action in ('first', 'resume'):
            if args.foreground:
                worker(cfg, config_path, args.action)
            else:
                launch(cfg, config_path, args.action)
        elif args.action == 'worker':
            worker(cfg, config_path, args.mode)
            write_same(Path(cfg['ls']['log_dir']) / 'full_run_LS.exitcode', '0\n')
        elif args.action == 'prepare':
            with locked(Path(cfg['ls']['log_dir']) / 'operation_LS.lock'):
                require_idle(Path(cfg['project_root']))
                prepare(cfg, config_path)
        elif args.action == 'verify':
            with locked(Path(cfg['ls']['log_dir']) / 'operation_LS.lock'):
                require_idle(Path(cfg['project_root']))
                check_identity(cfg, config_path)
                summarize(cfg)
        elif args.action in ('export', 'qc'):
            with locked(Path(cfg['ls']['log_dir']) / 'operation_LS.lock'):
                require_idle(Path(cfg['project_root']))
                check_identity(cfg, config_path)
                if args.action == 'qc':
                    cli(cfg, config_path, 'qc', '--participant-id', 'all', '--port', str(cfg['ls']['qc_port']), '--no-browser')
                else:
                    cli(cfg, config_path, 'export', '--participant-id', 'all')
        return 0
    except Exception as error:
        if args.action == 'worker' and cfg is not None:
            write_same(Path(cfg['ls']['log_dir']) / 'full_run_LS.exitcode', '1\n')
        print('LS_ERROR: {}: {}'.format(type(error).__name__, error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
