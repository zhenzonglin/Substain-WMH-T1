"""One-time warmup scheduling, graceful handover, and reuse of completed work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import pytest

pytest.importorskip('fcntl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warmup80_LS as warmup
from test_runner_LS import (runner, project as project_fixture, gb18_baseline as baseline_fixture, write_roster, write_state,
                           frozen_identity, build_overlay)

project = project_fixture
gb18_baseline = baseline_fixture


def prepass(cfg, person):
    for stage in warmup.PRE_STAGES:
        write_state(cfg, person, stage)


def test_progress_excludes_failures_and_prefers_partial_subjects(project):
    _, cfg, _ = project
    write_roster(cfg, ('P0001', 'P0005', 'P0009', 'P0010'))
    prepass(cfg, 'P0001')
    write_state(cfg, 'P0005', 'skullstrip')
    write_state(cfg, 'P0009', 'wmh_seg', 'fail')
    assert warmup.progress(runner, cfg) == (['P0001'], ['P0005', 'P0010'], ['P0009'])
    write_state(cfg, 'P0010', 't1')
    with pytest.raises(ValueError, match='missing upstream'):
        warmup.progress(runner, cfg)


def test_collect_tops_up_to_80_and_retains_failures_and_completed_outputs(project, monkeypatch):
    _, cfg, config = project
    people = ['P{:04d}'.format(i) for i in range(83)]
    write_roster(cfg, people)
    for person in people[:78]:
        prepass(cfg, person)
    existing = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(cfg['derivatives']).rglob('*.json')}
    calls = []
    def execute(command, **kwargs):
        selected = [Path(arg).parts[-3][4:] for arg in command if arg.endswith('/status/wmh.json')]
        calls.append(selected)
        for person in selected:
            prepass(cfg, person)
            if person == people[78]:
                write_state(cfg, person, 'wmh', 'fail')
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(warmup.subprocess, 'run', execute)
    phases = []
    warmup.collect(runner, cfg, config, lambda phase, **fields: phases.append((phase, fields)))
    assert calls == [people[78:80], [people[80]]]
    assert len(warmup.progress(runner, cfg)[0]) == 80
    assert runner.checked_status(cfg, people[78], 'wmh')['status'] == 'fail'
    assert phases[-1][0] == 'threshold_met'
    assert all((Path(path).read_bytes(), Path(path).stat().st_mtime_ns) == saved for path, saved in existing.items())
    assert not list(Path(cfg['derivatives']).glob('sub-*/status/t1.json'))


def test_exhausted_or_missing_outputs_do_not_release_t1(project, monkeypatch):
    _, cfg, config = project
    write_roster(cfg, ('P0001',))
    write_state(cfg, 'P0001', 'skullstrip', 'fail')
    with pytest.raises(RuntimeError, match='Fewer than 80'):
        warmup.collect(runner, cfg, config, lambda *a, **k: None)
    write_state(cfg, 'P0001', 'skullstrip', 'pass')
    monkeypatch.setattr(warmup.subprocess, 'run', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='without all selected'):
        warmup.collect(runner, cfg, config, lambda *a, **k: None)


def test_preflight_reuses_saved_audit_and_rejects_modified_control_file(project):
    root, cfg, config = project
    frozen_identity(cfg, config)
    logs = Path(cfg['ls']['log_dir'])
    report = Path(cfg['derivatives']) / 'audit_report_LS.json'
    report.write_text('{"status":"pass"}')
    fingerprint = runner.digest(logs / 'run_manifest_LS.json')
    for name in ('audit_guard_LS.json', 'pilot_verification_LS.json'):
        (logs / name).write_text(json.dumps(dict(status='pass', run_manifest_sha256=fingerprint,
            report=str(report), report_sha256=runner.digest(report))))
    module = root / 'scripts/cohort_LS/runner_LS.py'
    module.write_text('# frozen runner')
    manifest_path = logs / 'run_manifest_LS.json'
    manifest = runner.read_json(manifest_path)
    manifest['files'][str(module)] = runner.digest(module)
    manifest_path.write_text(runner.canonical(manifest))
    fingerprint = runner.digest(manifest_path)
    for name in ('audit_guard_LS.json', 'pilot_verification_LS.json'):
        value = runner.read_json(logs / name)
        value['run_manifest_sha256'] = fingerprint
        (logs / name).write_text(runner.canonical(value))
    assert warmup.preflight(runner, cfg, config) == fingerprint
    before = manifest_path.read_bytes()
    module.write_text('# unknown edit')
    with pytest.raises(ValueError, match='control file changed'):
        warmup.preflight(runner, cfg, config)
    assert manifest_path.read_bytes() == before


def proc(pid, argv, ppid=50):
    return dict(pid=pid, ppid=ppid, pgid=50, start_ticks=123, state='S', argv=argv)


def test_drain_signals_only_scheduler_and_its_admission_waiter(project, monkeypatch):
    root, cfg, config = project
    scheduler = proc(60, ['python', '-m', 'snakemake', '--configfile', str(config), '--snakefile', str(Path(cfg['ls']['runtime_dir']) / 'workflow/Snakefile_LS')])
    waiter = proc(70, ['python', str(root / 'scripts/rolling_admission.py'), 'wait', '--output', str(root / 'logs/LS/queue_LS/old_admission_LS/P0200.json')], ppid=60)
    t1 = proc(80, ['python', '-m', 'substain_features.cli', 'stage', 't1', '--config-file', str(config)], ppid=60)
    frames = iter([([scheduler, waiter, t1], dict(status='running', pid=50), {50, 60, 70, 80}),
                   ([scheduler, t1], dict(status='running', pid=50), {50, 60, 80}),
                   ([], dict(status='exited', pid=50), set())])
    monkeypatch.setattr(warmup, 'graceful_supported', lambda: None)
    monkeypatch.setattr(warmup, 'live_context', lambda *a: next(frames))
    signals = []
    monkeypatch.setattr(warmup, 'signal_exact', lambda r, p, **k: signals.append((p['pid'], k.get('require_handler', False))))
    monkeypatch.setattr(runner, 'require_idle', lambda *a: None)
    warmup.drain(runner, cfg, config, lambda *a, **k: None, pause=lambda _: None)
    assert signals == [(60, True), (70, False)]


def test_signal_rejects_reused_pid(monkeypatch):
    expected = proc(60, ['python', '-m', 'snakemake'])
    fake = types.SimpleNamespace(process_identity=lambda _: dict(expected, start_ticks=124))
    monkeypatch.setattr(warmup.os, 'kill', lambda *a: pytest.fail('Signalled reused PID'))
    with pytest.raises(RuntimeError, match='identity changed'):
        warmup.signal_exact(fake, expected)


def test_foreign_gb_scheduler_blocks_handover(project, monkeypatch):
    root, cfg, config = project
    gb = proc(91, ['python', '-m', 'snakemake', '--configfile', str(root / 'config/config.yaml')], ppid=90)
    monkeypatch.setattr(warmup.monitor, 'read_processes', lambda: ([gb], True, []))
    monkeypatch.setattr(warmup.monitor, '_runner_state', lambda *a: (dict(status='running', pid=50), []))
    with pytest.raises(RuntimeError, match='Another GB/LS'):
        warmup.live_context(cfg, config)


def test_worker_restores_original_parallel_flow_once_without_install_edits(project, monkeypatch):
    _, cfg, config = project
    logs = Path(cfg['ls']['log_dir'])
    state = logs / 'warmup80_LS.json'
    state.write_text(json.dumps(dict(pid=os.getpid(), config=str(config))))
    old_pointer = logs / 'full_run_LS.json'
    old_pointer.write_text('{"pid":1234}')
    events = []
    monkeypatch.setattr(warmup, 'preflight', lambda *a: 'frozen')
    monkeypatch.setattr(warmup, 'progress', lambda *a: ([], ['P0001'], []))
    monkeypatch.setattr(warmup, 'live_context', lambda *a: ([], {'status': 'exited'}, set()))
    monkeypatch.setattr(warmup, 'drain', lambda *a: events.append('drain'))
    monkeypatch.setattr(warmup, 'collect', lambda *a: events.append('warmup'))
    monkeypatch.setattr(runner, 'require_idle', lambda *a: None)
    original = runner.snakemake_command
    def parallel(current, path, mode):
        assert mode == 'resume'
        command = runner.snakemake_command(current, path, ['all'])
        assert command[command.index('--rerun-triggers') + 1] == 'mtime'
        events.append('parallel')
    monkeypatch.setattr(runner, 'worker', parallel)
    warmup.worker(runner, cfg, config)
    assert events == ['drain', 'warmup', 'parallel']
    assert runner.snakemake_command is original
    assert json.loads(state.read_text())['phase'] == 'complete'
    assert old_pointer.read_text() == '{"pid":1234}'


def test_already_80_ready_leaves_running_queue_untouched(project, monkeypatch):
    _, cfg, config = project
    state = Path(cfg['ls']['log_dir']) / 'warmup80_LS.json'
    state.write_text(json.dumps(dict(pid=os.getpid(), config=str(config))))
    monkeypatch.setattr(warmup, 'preflight', lambda *a: 'frozen')
    monkeypatch.setattr(warmup, 'progress', lambda *a: (list(range(80)), [], []))
    monkeypatch.setattr(warmup, 'live_context', lambda *a: ([], {'status': 'running'}, set()))
    monkeypatch.setattr(warmup, 'drain', lambda *a: pytest.fail('Unnecessary handover'))
    warmup.worker(runner, cfg, config)
    assert json.loads(state.read_text())['phase'] == 'already_satisfied'


def test_real_upstream_dry_run_excludes_t1_qc_cleanup_and_keeps_limits(project):
    _, cfg, config = project
    write_roster(cfg)
    runtime = Path(cfg['ls']['runtime_dir'])
    build_overlay(Path(cfg['project_root']), config, runtime)
    Path(cfg['ls']['audit_guard_path']).write_text('{"status":"pass"}')
    command = warmup.warmup_command(runner, cfg, config, ['P0001'])
    assert all(item in command for item in ('64', 'compute_jobs=18', 'gpu=4', 'wmh_exclusive=2'))
    assert not any(item.startswith('rolling_order_file=') for item in command)
    command[0] = sys.executable
    command.insert(3, '--dry-run')
    env = dict(os.environ, PYTHONPATH=str(runtime / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(command, cwd=str(runtime), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=40)
    assert result.returncode == 0, result.stdout
    for name in ('skullstrip', 'wmh_segmentation', 'registration', 'lesion_processing', 'wmh'):
        assert 'rule ' + name + ':' in result.stdout
    for name in ('t1', 'qc', 'cleanup', 'audit', 'rolling_admission'):
        assert 'rule ' + name + ':' not in result.stdout


def test_real_snakemake_graceful_signal_finishes_current_job_only(tmp_path):
    warmup.graceful_supported()
    snake = tmp_path / 'Snakefile'
    snake.write_text('rule b:\n    input: "a.ok"\n    output: "b.ok"\n    shell: "touch {output}"\n'
                     'rule a:\n    output: "a.ok"\n    shell: "touch started; sleep 1; touch {output}"\n')
    with (tmp_path / 'log.txt').open('w') as log:
        job = subprocess.Popen([sys.executable, '-m', 'snakemake', '--cores', '1', 'b.ok'], cwd=str(tmp_path), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / 'started').exists() and job.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert (tmp_path / 'started').exists(), (tmp_path / 'log.txt').read_text()
            identity = runner.process_identity(job.pid)
            warmup.signal_exact(runner, identity, require_handler=True)
            assert job.wait(timeout=15) != 0
            assert (tmp_path / 'a.ok').exists()
            assert not (tmp_path / 'b.ok').exists()
        finally:
            if job.poll() is None:
                job.terminate()
                job.wait(timeout=10)


def test_scheduler_admission_change_does_not_rerun_completed_step(tmp_path):
    snake = tmp_path / 'Snakefile'
    snake.write_text('rule task:\n    input: ancient(config.get("token", []))\n    output: "done.ok"\n    shell: "touch {output}"\n')
    for name in ('old.json', 'new.json'):
        (tmp_path / name).write_text('{}')
    base = [sys.executable, '-m', 'snakemake', '--cores', '1', 'done.ok', '--config']
    subprocess.run(base + ['token=old.json'], cwd=str(tmp_path), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    before = (tmp_path / 'done.ok').stat().st_mtime_ns
    for overrides in ([], ['token=new.json']):
        result = subprocess.run(warmup.preserve_completed(base + overrides), cwd=str(tmp_path), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert 'Nothing to be done' in result.stdout
        assert (tmp_path / 'done.ok').stat().st_mtime_ns == before
