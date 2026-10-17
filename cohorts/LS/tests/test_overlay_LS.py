"""Controlled overlay tests use real archived sources and no model inference."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parent / 'fixtures'
BASELINE = WORKSPACE / 'Substain-v1.0.9-remote-sim'
sys.path.insert(0, str(PACKAGE))
from overlay_LS import build_overlay, digest, TABLES


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture_project(tmp_path, version='original'):
    root = tmp_path / 'Substain_GB'
    for directory in ('src/substain_features', 'scripts/steps'):
        shutil.copytree(str(BASELINE / directory), str(root / directory), ignore=shutil.ignore_patterns('__pycache__'))
    names = ('config/config.yaml', 'workflow/Snakefile', 'run_pipeline.sh',
             'scripts/finish_ws1_v1_0_9.sh', 'scripts/start_ws1_v1_0_9.sh',
             'scripts/rolling_admission.py', 'scripts/install_offline.sh')
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(BASELINE / name), str(path))
    if version != 'original':
        patcher = load_file('fixture_gb_minimal', WORKSPACE / 'Substain-ws1-rolling-patch-release/gb_minimal.py')
        with contextlib.redirect_stdout(io.StringIO()):
            patcher.apply(root)
    if version == 'gb18':
        # Produce exactly the CPU64 configuration then apply the shipped 18-job
        # pure planner. Only shell syntax checks are stubbed on Windows.
        for name in ('config/config.yaml', 'src/substain_features/cli.py',
                     'scripts/finish_ws1_v1_0_9.sh', 'scripts/start_ws1_v1_0_9.sh',
                     'run_pipeline.sh', 'scripts/steps/02_features.sh'):
            path = root / name
            text = path.read_text(encoding='utf-8')
            for old, new in (('cpu_cores: 50', 'cpu_cores: 64'), ('default=50,', 'default=64,'),
                             ('cores > 50', 'cores > 64'), ('cores <= 50', 'cores <= 64'),
                             ('50 cores', '64 cores'), ('${1:-50}', '${1:-64}'),
                             ('${3:-50}', '${3:-64}'), ('${4:-50}', '${4:-64}'),
                             ('finish_ws1_v1_0_9.sh" 50', 'finish_ws1_v1_0_9.sh" 64')):
                text = text.replace(old, new)
            path.write_text(text, encoding='utf-8')
        manifest = root / 'logs/gb/rerun_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'project_root': str(root), 'queue_participants': ['c001p00016']}))
        load_file('gb_resume_audit', WORKSPACE / 'deliverables/gb_resume_audit.py')
        patcher = load_file('fixture_gb18', WORKSPACE / 'deliverables/gb_concurrency18_patch.py')
        for path in root.rglob('*'):
            if path.is_file() and (path.suffix in ('.py', '.sh', '.yaml', '.json') or path.name == 'Snakefile'):
                path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        from unittest.mock import patch
        with patch.object(patcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)):
            _, planned = patcher.planned_edits(root)
        for name, data in planned.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
    config = yaml.safe_load((root / 'config/config.yaml').read_text(encoding='utf-8'))
    output = tmp_path / 'Sustain_LS/derivatives'
    output.mkdir(parents=True)
    config.update(participants='config/participants_LS.tsv', derivatives=str(output), qc_dir=str(output / 'qc'), cohort_suffix='_LS')
    config['input']['cohort'] = 'LS'
    config['ls'] = {'audit_guard_path': str(output / 'audit/input_guard_LS.json'),
                    'resource_manifest_path': str(output / 'tables/resource_manifest_LS.tsv')}
    config_path = root / 'config/config_LS.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (root / 'config/participants_LS.tsv').write_text('participant_id\tage\tsex\tsite_id\tt1w\tflair\tlesion_mask\nP0001\t60\tmale\tLS\t/a\t/b\t/c\n')
    guard = Path(config['ls']['audit_guard_path'])
    guard.parent.mkdir(parents=True)
    guard.write_text('{"status":"pass","config_fingerprint":"fixture","input_fingerprint":"fixture"}')
    return root, config_path, output


def snapshot(root):
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*')
            if p.is_file() and 'runtime_LS' not in p.relative_to(root).parts and '__pycache__' not in p.parts}


@pytest.mark.parametrize('version', ['original', 'gb', 'gb18'])
def test_overlay_known_baselines_preserve_gb_and_resources(tmp_path, version):
    root, config, output = fixture_project(tmp_path, version)
    before = snapshot(root)
    destination = root / 'runtime_LS'
    plan = build_overlay(root, config, destination, dry_run=True)
    assert not destination.exists()
    actual = build_overlay(root, config, destination)
    assert plan == actual
    assert snapshot(root) == before
    assert build_overlay(root, config, destination) == actual
    snake = (destination / 'workflow/Snakefile_LS').read_text(encoding='utf-8')
    assert 'GB_AUDIT_INPUT' not in snake and 'gb_resume_audit' not in snake
    assert 'LS_AUDIT_INPUT = ancient(audit_guard_LS(config))' in snake
    assert 'audit=LS_AUDIT_INPUT,' in snake
    assert 'configfile: ' + repr(str(config)) in snake
    assert 'root=SOURCE_ROOT,' in snake
    assert 'PYTHONPATH={ROOT}/src' not in snake
    assert 'all_complete_LS.json' in snake
    for module in ('gpu_pool.py', 't1.py', 'wmh.py', 'registration.py', 'synthstrip.py', 'symmetry.py', 'normative.py', 'images.py'):
        assert (destination / 'src/substain_features' / module).read_bytes() == (root / 'src/substain_features' / module).read_bytes()
    if version == 'gb18':
        assert snake.count('compute_jobs=1,') == 10
        assert '    threads: 3\n' in snake
        assert 'compute_jobs=18' in (destination / 'src/substain_features/cli.py').read_text(encoding='utf-8')
    for path in (destination / 'src').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))


def test_rejects_unknown_source_before_destination_write(tmp_path):
    root, config, _ = fixture_project(tmp_path)
    snake = root / 'workflow/Snakefile'
    snake.write_text(snake.read_text(encoding='utf-8').replace('audit=rules.audit.output,', 'audit=unknown_input,'), encoding='utf-8')
    with pytest.raises(ValueError, match='Unrecognized'):
        build_overlay(root, config, root / 'runtime_LS')
    assert not (root / 'runtime_LS').exists()


def test_never_overwrites_modified_overlay_or_new_baseline(tmp_path):
    root, config, _ = fixture_project(tmp_path)
    destination = root / 'runtime_LS'
    build_overlay(root, config, destination)
    changed = destination / 'src/substain_features/t1.py'
    changed.write_bytes(changed.read_bytes() + b'\n# human edit\n')
    before = changed.read_bytes()
    with pytest.raises(ValueError, match='refusing overwrite'):
        build_overlay(root, config, destination)
    assert changed.read_bytes() == before


def test_all_cohort_filenames_are_suffixed_and_individual_contracts_unchanged(tmp_path):
    root, config, _ = fixture_project(tmp_path)
    destination = root / 'runtime_LS'
    build_overlay(root, config, destination)
    seen = set()
    for name in ('pipeline.py', 'audit.py', 'qc_review.py', 'cli.py'):
        text = (destination / 'src/substain_features' / name).read_text(encoding='utf-8')
        values = {node.value for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not set(TABLES).intersection(values)
        seen.update(values)
    assert 'qc_reviews_LS.sqlite' in seen
    assert 'features_primary40_LS.tsv' in seen
    pipeline = (destination / 'src/substain_features/pipeline.py').read_text(encoding='utf-8')
    assert '"{}.json".format(stage)' in pipeline
    assert '"wmh_features.json"' in pipeline
    assert 'existing_manifest_LS(root, config, manifest_path)' in (destination / 'src/substain_features/cli.py').read_text(encoding='utf-8')


def test_crosscwd_root_and_guard_refuse_failed_or_gb_guard(tmp_path, monkeypatch):
    root, config_path, output = fixture_project(tmp_path)
    destination = root / 'runtime_LS'
    build_overlay(root, config_path, destination)
    helper = load_file('runtime_overlay_fixture', destination / 'src/substain_features/runtime_LS.py')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert helper.audit_guard_LS(config) == config['ls']['audit_guard_path']
    with pytest.raises(ValueError, match='only'):
        helper.validate_config_path_LS(root / 'config/config.yaml')
    config['ls']['audit_guard_path'] = str(root / 'logs/gb/resume_audit_cache.json')
    with pytest.raises(ValueError, match='regular'):
        helper.audit_guard_LS(config)
    config['ls']['audit_guard_path'] = str(output / 'audit/input_guard_LS.json')
    Path(config['ls']['audit_guard_path']).write_text('{"status":"fail"}')
    with pytest.raises(ValueError, match='not passed'):
        helper.audit_guard_LS(config)


def test_scoped_resource_manifest_is_read_only_and_detects_changes(tmp_path):
    root, config_path, output = fixture_project(tmp_path)
    destination = root / 'runtime_LS'
    build_overlay(root, config_path, destination)
    helper = load_file('runtime_manifest_fixture', destination / 'src/substain_features/runtime_LS.py')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    resource = root / 'resources/model.fixture'
    resource.parent.mkdir()
    resource.write_bytes(b'immutable model')
    manifest = Path(config['ls']['resource_manifest_path'])
    manifest.parent.mkdir()
    manifest.write_text('relative_path\tsha256\nresources/model.fixture\t' + digest(resource.read_bytes()) + '\n')
    before = manifest.read_bytes()
    assert helper.existing_manifest_LS(root, config, manifest)['checked_files'] == 1
    resource.write_bytes(b'changed model')
    with pytest.raises(ValueError, match='verification failed'):
        helper.existing_manifest_LS(root, config, manifest)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize('version', ['original', 'gb18'])
def test_snake_python_preamble_and_formatted_commands_use_overlay_from_any_cwd(tmp_path, monkeypatch, version):
    root, config_path, _ = fixture_project(tmp_path, version)
    destination = root / 'runtime_LS'
    build_overlay(root, config_path, destination)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    snake = (destination / 'workflow/Snakefile_LS').read_text(encoding='utf-8')
    preamble = '\n'.join(line for line in snake.split('rule all:', 1)[0].splitlines() if not line.startswith('configfile:'))
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config.update(profile='gpu', gpu_devices='0,1', gpu_slots_per_device=2)
    namespace = {'config': config, 'ancient': lambda value: value}
    exec(compile(preamble, '<actual Snakefile preamble>', 'exec'), namespace)
    assert namespace['ROOT'] == root
    assert namespace['CONFIG_FILE'] == config_path
    direct = namespace['execution_command']('/python', '-m stage')
    assert str(destination) + '/src' in direct
    prefix = namespace['gpu_prefix'](2)
    assert str(destination) + '/src' in prefix
    assert str(root / config['execution']['gpu_lock_dir']) in prefix
    for line in snake.splitlines():
        if line.strip().startswith('recorded_subject_failure_allowed('):
            command = eval(line.strip(), namespace)
            assert str(destination) + '/src' in command


def test_aggregate_only_writes_ls_tables_and_never_deletes_gb_primary(tmp_path):
    root, config_path, output = fixture_project(tmp_path)
    destination = root / 'runtime_LS'
    build_overlay(root, config_path, destination)
    text = (destination / 'src/substain_features/pipeline.py').read_text(encoding='utf-8')
    functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_absolute', '_records_to_table', 'aggregate_outputs')]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'Path': Path, 'pd': pd, 'np': np, 'WMH_FEATURES': ['wmh_{}_ml'.format(i) for i in range(20)], 'display_name_map': lambda path: {}}
    exec(compile(module, '<isolated actual aggregate function>', 'exec'), namespace)
    mapping = root / 'resources/mappings/macro.tsv'
    mapping.parent.mkdir(parents=True)
    pd.DataFrame({'macro_index': range(1, 21), 'macro_id': ['region{}'.format(i) for i in range(20)]}).to_csv(mapping, sep='\t', index=False)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['project_root'] = str(root)
    config['t1']['macro_mapping'] = str(mapping)
    tables = output / 'tables'
    tables.mkdir()
    gb_primary = tables / 'features_primary40.tsv'
    gb_primary.write_text('sentinel GB or prior naming file')
    (tables / 'features_primary40_LS.tsv').write_text('stale LS export')
    result = namespace['aggregate_outputs'](config, [])
    assert result['primary40_withheld_pending_manual_qc'] is True
    assert gb_primary.read_text() == 'sentinel GB or prior naming file'
    assert not (tables / 'features_primary40_LS.tsv').exists()
    assert (tables / 'features_computed40_LS.tsv').is_file()
    assert all('_LS' in path.stem for path in tables.iterdir() if path != gb_primary)


@pytest.mark.parametrize('version', ['original', 'gb18'])
@pytest.mark.parametrize('rolling', [False, True])
def test_real_snakemake_dry_run_uses_ls_config_outputs_and_private_workdir(tmp_path, version, rolling):
    pytest.importorskip('snakemake')
    root, config_path, output = fixture_project(tmp_path, version)
    destination = root / 'runtime_LS'
    build_overlay(root, config_path, destination)
    work = destination / 'work_LS'
    work.mkdir()
    overrides = ['profile=cpu', 'selected_participant=all', 'active_config_file=' + str(config_path)]
    if rolling:
        order = root / 'logs/LS/order_LS.txt'
        order.parent.mkdir(parents=True)
        order.write_text('P0001\n')
        tokens = root / 'logs/LS/admission_LS'
        target = str(tokens / 'all_complete_LS.json')
        overrides += ['rolling_order_file=' + str(order), 'rolling_token_dir=' + str(tokens), 'rolling_window=200']
    else:
        target = 'all'
    # The existing test runtime is a wrapper around system Python plus core-site.
    # Preserve its dependency path while putting only the LS package first.
    env = dict(os.environ, PYTHONPATH=str(destination / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-m', 'snakemake', '--snakefile', str(destination / 'workflow/Snakefile_LS'),
        '--configfile', str(config_path), '--cores', '8', '--dry-run', '--printshellcmds', target,
        '--config'] + overrides, cwd=str(work), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout
    assert str(config_path) in result.stdout
    assert str(destination / 'src') in result.stdout
    assert str(output / 'sub-P0001') in result.stdout
    assert not (root / '.snakemake').exists()
    assert 'resume_audit_cache.json' not in result.stdout
    if rolling:
        assert 'all_complete_LS.json' in result.stdout
    else:
        assert 'features_computed40_LS.tsv' in result.stdout
