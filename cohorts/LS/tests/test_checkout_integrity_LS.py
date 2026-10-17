"""Git source verification and the short workstation installation entrypoint."""
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('checkout_verifier_LS', PACKAGE / 'verify_checkout_LS.py')
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def source_tree(tmp_path):
    root = tmp_path / 'cloned repo/cohorts/LS'
    root.mkdir(parents=True)
    for name in ('install_LS.sh', 'verify_checkout_LS.py'):
        shutil.copyfile(str(PACKAGE / name), str(root / name))
    for name in ('inputs_LS.py', 'monitor_LS.py', 'overlay_LS.py', 'runner_LS.py',
                 'vendor/openpyxl/__init__.py', 'vendor/et_xmlfile/__init__.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic test source\n', encoding='utf-8')
    write_manifest(root)
    return root


def write_manifest(root):
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS_LS':
            rows.append('{}  {}\n'.format(hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(root).as_posix()))
    (root / 'SHA256SUMS_LS').write_text(''.join(rows), encoding='utf-8')


def test_checksum_verifier_rejects_modified_runtime_before_install(tmp_path):
    root = source_tree(tmp_path)
    assert verifier.verify(root) == 8
    (root / 'runner_LS.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='SHA256 mismatch: runner_LS.py'):
        verifier.verify(root)


def test_checkout_rejects_parent_path_and_missing_required_file(tmp_path):
    root = source_tree(tmp_path)
    manifest = root / 'SHA256SUMS_LS'
    original = manifest.read_text()
    manifest.write_text(original + '0' * 64 + '  ../outside\n')
    with pytest.raises(ValueError, match='Unsafe'):
        verifier.verify(root)
    manifest.write_text(''.join(line for line in original.splitlines(True) if not line.endswith('  runner_LS.py\n')))
    with pytest.raises(ValueError, match='Incomplete'):
        verifier.verify(root)


@pytest.mark.skipif(os.name == 'nt', reason='Workstation entrypoint is Linux Bash')
def test_shell_install_from_unrelated_directory_verifies_then_invokes_existing_python(tmp_path):
    root = source_tree(tmp_path)
    project = tmp_path / 'existing GB'
    python = project / 'envs/core-venv/bin/python'
    python.parent.mkdir(parents=True)
    record = tmp_path / 'invocation.txt'
    # A fake existing interpreter captures argv; no project state or models are used.
    python.write_text('#!' + sys.executable + '\nimport os,sys\nfrom pathlib import Path\n'
                      'Path(os.environ["LS_TEST_INVOCATION"]).write_text("\\n".join(sys.argv[1:]))\n')
    python.chmod(0o755)
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    env = dict(os.environ, SUBSTAIN_PROJECT_ROOT=str(project), LS_TEST_INVOCATION=str(record))
    result = subprocess.run(['bash', str(root / 'install_LS.sh')], cwd=str(unrelated), env=env,
                            text=True, capture_output=True, check=True)
    assert 'LS_CHECKOUT_VERIFIED files=8' in result.stdout
    assert record.read_text().splitlines() == [str(root / 'runner_LS.py'), 'install', '--root', str(project)]
    record.unlink()
    (root / 'inputs_LS.py').write_text('# tampered\n')
    failure = subprocess.run(['bash', str(root / 'install_LS.sh')], cwd=str(unrelated), env=env,
                             text=True, capture_output=True)
    assert failure.returncode != 0 and not record.exists()


@pytest.mark.skipif(os.name == 'nt', reason='Workstation monitor entrypoint is Linux Bash')
@pytest.mark.parametrize('entry,extra', [('monitor_LS', ['--interval', '30']), ('warmup80_LS', [])])
def test_checkout_monitor_invokes_external_code_without_changing_installed_runtime(tmp_path, entry, extra):
    root = source_tree(tmp_path)
    shutil.copyfile(str(PACKAGE / (entry + '.sh')), str(root / (entry + '.sh')))
    project = tmp_path / 'existing GB'
    python = project / 'envs/core-venv/bin/python'
    python.parent.mkdir(parents=True)
    record = tmp_path / 'monitor_invocation.txt'
    python.write_text('#!' + sys.executable + '\nimport os,sys\nfrom pathlib import Path\n'
                      'Path(os.environ["LS_TEST_INVOCATION"]).write_text("\\n".join(sys.argv[1:]))\n')
    python.chmod(0o755)
    installed = project / 'scripts/cohort_LS/monitor_LS.py'
    installed.parent.mkdir(parents=True)
    installed.write_text('# previously installed frozen monitor\n')
    before = {str(p): p.read_bytes() for p in project.rglob('*') if p.is_file()}
    env = dict(os.environ, SUBSTAIN_PROJECT_ROOT=str(project), LS_TEST_INVOCATION=str(record))
    subprocess.run(['bash', str(root / (entry + '.sh'))] + extra, cwd=str(tmp_path), env=env,
                   text=True, capture_output=True, check=True)
    expected = [str(root / (entry + '.py'))] + (['start'] if entry == 'warmup80_LS' else [])
    assert record.read_text().splitlines() == expected + ['--config', str(project / 'config/config_LS.yaml')] + extra
    assert {str(p): p.read_bytes() for p in project.rglob('*') if p.is_file()} == before
