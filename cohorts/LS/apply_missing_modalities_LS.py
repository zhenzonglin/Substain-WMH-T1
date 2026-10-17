#!/usr/bin/env python3
"""Patch only the input adapter after a failed, unfrozen LS preparation."""
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
KNOWN_PREVIOUS_SHA256 = frozenset({
    '591c9999d8cc3d5207bb90b0e08b39c28a66ad8275593422b4479f195063ac1d',
    'f40d700337ec306acb5c243ea3aef4f7a32f30c92af1f39f23345c715a7dbe17',
})


def apply(root: Path, runner=None) -> dict:
    root = root.resolve(strict=True)
    if runner is None:
        spec = importlib.util.spec_from_file_location('installed_runner_LS', root / 'scripts/cohort_LS/runner_LS.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    cfg = runner.load_config(root / 'config/config_LS.yaml')
    logs = root / 'logs/LS'
    prep = root / 'inputs/cohort_LS'
    if Path(cfg['ls']['log_dir']) != logs or Path(cfg['ls']['preparation_dir']) != prep:
        raise ValueError('Unexpected LS preparation/log paths; no files changed')
    target = root / 'scripts/cohort_LS/inputs_LS.py'
    new_data = (HERE / 'inputs_LS.py').read_bytes()
    expected = {line.split('  ', 1)[1]: line.split('  ', 1)[0]
                for line in (HERE / 'SHA256SUMS_LS').read_text().splitlines()}['inputs_LS.py']
    if hashlib.sha256(new_data).hexdigest() != expected:
        raise ValueError('Git checkout inputs_LS.py checksum mismatch')
    with runner.locked(logs / 'operation_LS.lock'):
        runner.require_idle(root)
        if target.is_symlink():
            raise ValueError('Refusing symlinked input adapter')
        old_data = target.read_bytes()
        if old_data == new_data:
            return {'status': 'already_updated', 'target': str(target)}
        old_sha256 = hashlib.sha256(old_data).hexdigest()
        if old_sha256 not in KNOWN_PREVIOUS_SHA256:
            raise ValueError('Input adapter differs from the known installed version; preserved')
        frozen = [Path(cfg['participants'])] + [logs / name for name in (
            'run_manifest_LS.json', 'audit_guard_LS.json', 'pilot_verification_LS.json')]
        if any(path.exists() or path.is_symlink() for path in frozen):
            raise ValueError('LS roster/run is already frozen; this preparation-only patch did not change it')
        derivatives = Path(cfg['derivatives'])
        if any(derivatives.glob('sub-*/status/*.json')):
            raise ValueError('Existing analysis status found; no files changed')
        saved = logs / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_missing_modalities_backup_LS')
        saved.mkdir(parents=True, exist_ok=False)
        shutil.copy2(str(target), str(saved / target.name))
        for report in prep.glob('*_LS.*'):
            if report.is_file() and not report.is_symlink():
                shutil.copy2(str(report), str(saved / report.name))
        if target.read_bytes() != old_data:
            raise RuntimeError('Input adapter changed during backup; no replacement performed')
        runner.write_same(target, new_data)
        result = {'status': 'updated', 'target': str(target), 'backup': str(saved),
                  'before': old_sha256, 'after': expected, 'changed_runtime_files': 1}
        runner.write_same(saved / 'update_LS.json', json.dumps(result, indent=2) + '\n')
        return result


if __name__ == '__main__':
    try:
        result = apply(Path(sys.argv[1]) if len(sys.argv) == 2 else Path('/data02/linzhenzong/Substain_GB'))
        print('LS_MISSING_MODALITIES_READY ' + json.dumps(result), flush=True)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print('LS_PATCH_ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
