"""The preparation-only patch changes one module and retains previous evidence."""
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import apply_missing_modalities_LS as patch
import runner_LS as runner


@pytest.fixture
def state(tmp_path, monkeypatch):
    root = tmp_path / 'GB'
    source = tmp_path / 'checkout'
    source.mkdir()
    new = b'# new input adapter\n'
    (source / 'inputs_LS.py').write_bytes(new)
    (source / 'SHA256SUMS_LS').write_text(hashlib.sha256(new).hexdigest() + '  inputs_LS.py\n')
    monkeypatch.setattr(patch, 'HERE', source)
    old = b'# known previous input adapter\n'
    monkeypatch.setattr(patch, 'PREVIOUS_SHA256', hashlib.sha256(old).hexdigest())
    target = root / 'scripts/cohort_LS/inputs_LS.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(old)
    cfg = {'participants': str(root / 'config/participants_LS.tsv'), 'derivatives': str(tmp_path / 'LS_results'),
           'ls': {'log_dir': str(root / 'logs/LS'), 'preparation_dir': str(root / 'inputs/cohort_LS')}}
    prep = Path(cfg['ls']['preparation_dir'])
    prep.mkdir(parents=True)
    (prep / 'input_report_LS.json').write_text('{"status":"fail","errors":238}\n')
    fake = types.SimpleNamespace(load_config=lambda path: cfg, locked=runner.locked,
                                 require_idle=lambda path: None, write_same=runner.write_same)
    return root, cfg, fake, target, old, new


def test_single_module_update_preserves_reports_and_is_idempotent(state):
    root, cfg, fake, target, old, new = state
    report = Path(cfg['ls']['preparation_dir']) / 'input_report_LS.json'
    original_report = report.read_bytes()
    result = patch.apply(root, fake)
    assert target.read_bytes() == new
    assert report.read_bytes() == original_report
    backup = Path(result['backup'])
    assert (backup / target.name).read_bytes() == old
    assert (backup / report.name).read_bytes() == original_report
    assert json.loads((backup / 'update_LS.json').read_text())['changed_runtime_files'] == 1
    mtime = target.stat().st_mtime_ns
    assert patch.apply(root, fake)['status'] == 'already_updated'
    assert target.stat().st_mtime_ns == mtime


@pytest.mark.parametrize('marker', ('participants', 'run_manifest_LS.json', 'audit_guard_LS.json', 'pilot_verification_LS.json', 'status'))
def test_existing_roster_or_analysis_prevents_patch(state, marker):
    root, cfg, fake, target, old, _ = state
    if marker == 'participants':
        path = Path(cfg['participants'])
    elif marker == 'status':
        path = Path(cfg['derivatives']) / 'sub-P0001/status/t1.json'
    else:
        path = root / 'logs/LS' / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
    with pytest.raises(ValueError):
        patch.apply(root, fake)
    assert target.read_bytes() == old
    assert not list((root / 'logs/LS').glob('*_backup_LS'))


def test_active_job_unknown_code_and_lock_all_prevent_patch(state):
    root, _, fake, target, old, _ = state
    def busy(path):
        raise RuntimeError('active job')
    fake.require_idle = busy
    with pytest.raises(RuntimeError, match='active job'):
        patch.apply(root, fake)
    fake.require_idle = lambda path: None
    with runner.locked(root / 'logs/LS/operation_LS.lock'):
        with pytest.raises(RuntimeError, match='Another LS operation'):
            patch.apply(root, fake)
    target.write_bytes(old + b'# manual edit\n')
    with pytest.raises(ValueError, match='known installed'):
        patch.apply(root, fake)
    assert target.read_bytes().endswith(b'# manual edit\n')
