"""Integration checks with the real GB patch lineage and synthetic LS inputs."""
import sys
import json
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / 'vendor'))
import openpyxl
import runner_LS as runner
from test_overlay_LS import fixture_project


def image(path, values=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((4, 5, 6), dtype=np.uint8) if values is None else values
    affine = np.diag([-1., 1., 1., 1.])
    affine[:3, 3] = [3., -2., -2.]
    obj = nib.Nifti1Image(data, affine)
    obj.set_qform(affine, 1)
    obj.set_sform(affine, 1)
    nib.save(obj, str(path))
    return str(path)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    root, old_ls, output = fixture_project(tmp_path, 'gb18')
    # The overlay fixture's LS config is deliberately replaced before installation.
    old_ls.unlink()
    (root / 'config/participants_LS.tsv').unlink()
    raw = tmp_path / 'LS'
    for pid in ('P0001', 'P0005'):
        image(raw / pid / 'source_T1w_run1.nii.gz')
        image(raw / pid / 'source_T2wFLAIR_run1.nii.gz')
    book = openpyxl.Workbook()
    book.active.append(['BIGDATA_CODE', 'AGE', 'SEX', 'OTHER'])
    book.active.append(['P0001', 50.3, 1, 'kept'])
    book.active.append(['P0005', 50.7, 2, 'kept too'])
    book.save(str(raw / 'LS_INFO.xlsx'))
    image(root / 'resources/templates/fsl/MNI152_T1_1mm.nii.gz')
    reference = root / 'resources/normative/genmind_dataset.csv'
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text('Age,Sex\n' + '50,M\n' * 300 + '50,F\n' * 300)
    monkeypatch.setattr(runner, 'require_idle', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'environment_record', lambda cfg: {'core': {'models': [], 'packages': {'test': '1'}}})
    monkeypatch.setattr(runner, 'resource_paths', lambda cfg, env: [reference, root / 'resources/templates/fsl/MNI152_T1_1mm.nii.gz'])
    before = {str(p): runner.digest(p) for folder in ('src', 'workflow') for p in (root / folder).rglob('*') if p.is_file()}
    before[str(root / 'config/config.yaml')] = runner.digest(root / 'config/config.yaml')
    gb_sentinel = root / 'logs/gb/full_run.logpath'
    gb_sentinel.write_text('old GB log\n')
    before[str(gb_sentinel)] = runner.digest(gb_sentinel)
    runner.install(root, raw, output)
    cfg_path = root / 'config/config_LS.yaml'
    cfg = runner.load_config(cfg_path)
    return root, cfg_path, cfg, before


def test_install_prepare_freeze_and_reuse_never_modify_gb(installed, monkeypatch):
    root, config_path, cfg, before = installed
    runner.prepare(cfg, config_path)
    people = runner.participants(cfg)
    assert [(x['participant_id'], x['age'], x['sex']) for x in people] == [
        ('P0001', '50.3', 'male'), ('P0005', '50.7', 'female')]
    assert cfg['execution']['cpu_cores'] == 64
    assert cfg['execution']['cpu_threads_per_job'] == 2
    assert cfg['execution']['registration_threads'] == 3
    assert cfg['execution']['max_compute_jobs'] == 18
    assert cfg['derivatives'].endswith('Sustain_LS/derivatives')
    assert not Path(cfg['derivatives']).joinpath('substain_features').exists()
    roster = Path(cfg['participants'])
    mtime = roster.stat().st_mtime_ns
    runner.prepare(cfg, config_path)
    assert roster.stat().st_mtime_ns == mtime
    identity = runner.check_identity(cfg, config_path)
    calls = []
    def fake_audit(*args):
        calls.append(args)
        runner.write_same(Path(cfg['derivatives']) / 'audit/audit_report_LS.json', '{"status":"pass"}\n')
    monkeypatch.setattr(runner, 'cli', fake_audit)
    runner.ensure_audit(cfg, config_path, identity)
    runner.ensure_audit(cfg, config_path, identity)
    assert len(calls) == 1
    assert all(runner.digest(Path(path)) == checksum for path, checksum in before.items())
    assert (root / 'scripts/run_LS.sh').is_file()
    assert (root / 'scripts/monitor_LS.sh').is_file()


def test_reinstall_is_stable_and_preserves_modified_installed_code(installed):
    root, config_path, cfg, before = installed
    destination = root / 'scripts/cohort_LS'
    snapshot = {str(path): (runner.digest(path), path.stat().st_mtime_ns)
                for path in destination.rglob('*') if path.is_file()}
    runner.install(root, Path(cfg['ls']['input_root']), Path(cfg['derivatives']))
    assert {str(path): (runner.digest(path), path.stat().st_mtime_ns)
            for path in destination.rglob('*') if path.is_file()} == snapshot
    assert not (destination / 'tests').exists()
    assert not (destination / 'README_LS.md').exists()
    assert not (destination / 'SHA256SUMS_LS').exists()
    assert not (destination / 'test_results_LS.xml').exists()
    changed = destination / 'runner_LS.py'
    changed.write_text(changed.read_text() + '\n# local edit must be preserved\n')
    edited_digest = runner.digest(changed)
    config_digest = runner.digest(config_path)
    with pytest.raises(RuntimeError, match='Installed LS adapter differs'):
        runner.install(root, Path(cfg['ls']['input_root']), Path(cfg['derivatives']))
    assert runner.digest(changed) == edited_digest
    assert runner.digest(config_path) == config_digest
    assert all(runner.digest(Path(path)) == checksum for path, checksum in before.items())


def successful_subject(cfg, pid):
    subject = Path(cfg['derivatives']) / ('sub-' + pid)
    zero = np.zeros((4, 5, 6), dtype=np.uint8)
    wmh = zero.copy()
    wmh[1, 2, 3] = 1
    segmentation = image(subject / 'wmh/seg.nii.gz', wmh * 77)
    original = image(subject / 'wmh/original.nii.gz', wmh)
    corrected = image(subject / 'wmh/final.nii.gz', wmh)
    mask = image(subject / 'lesion/zero.nii.gz', zero)
    wmh_features = subject / 'wmh/wmh_features.json'
    t1_features = subject / 't1/t1_features.json'
    values = {str(i): float(i) for i in range(20)}
    runner.write_same(wmh_features, runner.canonical({'raw_ml': values, 'z_chung': values}))
    runner.write_same(t1_features, runner.canonical({'macro20_ml': values, 'macro20_atrophy_z': values}))
    details = {'original_wmh': original, 'corrected_wmh': corrected, 'feature_json': str(wmh_features),
               'lesion_component_count': 0, 'triggered_lesion_component_count': 0}
    details.update({key: 0. for key in ('wmh_lesion_overlap_ml', 'wmh_contralateral_donor_ml',
        'wmh_replacement_added_ml', 'wmh_original_overlap_removed_ml', 'wmh_bilateral_lesion_conflict_removed_ml',
        'wmh_out_of_brain_donor_removed_ml')})
    by_stage = {'wmh': details, 'wmh_seg': {'segmentation': segmentation},
                'lesion': {'lesion_t1': mask, 'lesion_flair': mask}, 't1': {'feature_json': str(t1_features)}}
    for stage in runner.STAGES:
        runner.write_same(subject / 'status' / (stage + '.json'), runner.canonical({
            'participant_id': pid, 'stage': stage, 'status': 'pass', 'details': by_stage.get(stage, {})}))
    return subject


def test_healthy_validation_and_counted_partial_export(installed):
    _, config_path, cfg, _ = installed
    runner.prepare(cfg, config_path)
    subject = successful_subject(cfg, 'P0001')
    assert runner.healthy_check(cfg, 'P0001')['features'] == 40
    failed = Path(cfg['derivatives']) / 'sub-P0005/status'
    for stage in runner.STAGES:
        runner.write_same(failed / (stage + '.json'), runner.canonical({
            'participant_id': 'P0005', 'stage': stage, 'status': 'fail' if stage == 't1' else 'pass',
            'details': {'error': 'synthetic failure'} if stage == 't1' else {}}))
    for name in ('features_computed40', 'wmh20_raw', 'wmh20_z_chung', 't1_macro20_raw',
                 't1_macro20_z_genmind', 't1_muse145_raw', 't1_gm119_raw'):
        runner.write_same(Path(cfg['derivatives']) / 'tables' / (name + '_LS.tsv'),
            'participant_id\t' + '\t'.join('f' + str(i) for i in range(40)) + '\nP0001\t' + '\t'.join(['1'] * 40) + '\n')
    summary = runner.summarize(cfg)
    assert (summary['success'], summary['failed'], summary['total']) == (1, 1, 2)
    report = (Path(cfg['derivatives']) / 'tables/not_exported_LS.tsv').read_text()
    assert 'P0005' in report and 'synthetic failure' in report
    runner.write_same(subject / 'status/cleanup.json', runner.canonical({
        'participant_id': 'P0001', 'stage': 'cleanup', 'status': 'fail', 'details': {'error': 'cleanup-only failure'}}))
    summary = runner.summarize(cfg)
    assert summary['success'] == 0 and summary['computed40_rows'] == 1
    assert summary['computed_with_cleanup_issue'] == 1
    assert 'P0001' not in (Path(cfg['derivatives']) / 'tables/not_exported_LS.tsv').read_text()
    with pytest.raises(ValueError, match='Required analysis'):
        runner.healthy_check(cfg, 'P0001')
    # A changed corrected mask must be detected, even if every stage reports pass.
    image(subject / 'wmh/final.nii.gz', np.zeros((4, 5, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match='identity'):
        runner.healthy_check(cfg, 'P0001', require_cleanup=False)


def test_additional_image_or_changed_source_cannot_reuse_results(installed):
    _, config_path, cfg, _ = installed
    runner.prepare(cfg, config_path)
    image(Path(cfg['ls']['input_root']) / 'P0001/another_T1w.nii.gz')
    with pytest.raises(RuntimeError, match='inventory'):
        runner.check_identity(cfg, config_path)


def test_missing_p0001_is_recorded_and_old_runner_selects_complete_pilot(installed):
    root, config_path, cfg, before = installed
    (Path(cfg['ls']['input_root']) / 'P0001/source_T1w_run1.nii.gz').unlink()
    runner.prepare(cfg, config_path)
    assert [row['participant_id'] for row in runner.participants(cfg)] == ['P0005']
    assert runner.choose_pilot(cfg) == 'P0005'
    exclusions = root / 'inputs/cohort_LS/excluded_subjects_LS.tsv'
    assert 'P0001' in exclusions.read_text()
    identity = runner.capture_identity(cfg, config_path)
    assert identity['files'][str(exclusions)] == runner.digest(exclusions)
    assert all(runner.digest(Path(path)) == checksum for path, checksum in before.items())


def test_existing_cpu_fallback_environment_is_recorded_without_reinstallation(monkeypatch, capsys):
    cfg = {'execution': {name + '_python': '/' + name + '/python' for name in ('core', 'wmh', 't1')}}
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        return types.SimpleNamespace(stdout=json.dumps({'python': '3.8', 'packages': {},
            'models': ['model' + str(i) for i in range(6)] if argv[0] == '/t1/python' else [],
            'torch_version': '2.2.1+cpu', 'cuda_build': None, 'cuda_available': False}) + '\n')
    monkeypatch.setattr(runner, 'env_for', lambda config: {})
    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    record = runner.environment_record(cfg)
    assert record['t1']['torch_version'] == '2.2.1+cpu'
    assert all(argv[1] == '-c' for argv in calls)
    assert 'original stage fallback policy' in capsys.readouterr().out
