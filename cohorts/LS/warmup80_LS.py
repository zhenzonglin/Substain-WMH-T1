#!/usr/bin/env python3
"""One-time 80-person upstream warmup, using the frozen installed LS workflow."""
import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import monitor_LS as monitor

HERE = Path(__file__).resolve().parent
GOAL = 80
PRE_STAGES = monitor.STAGES[:5]


def installed_runner(config_path):
    root = config_path.parent.parent
    path = root / 'scripts/cohort_LS/runner_LS.py'
    spec = importlib.util.spec_from_file_location('installed_runner_LS', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner, runner.load_config(config_path)


def preflight(runner, cfg, config_path):
    """Check frozen control files only; original resume retains its input checks."""
    logs = Path(cfg['ls']['log_dir'])
    manifest_path = logs / 'run_manifest_LS.json'
    manifest = runner.read_json(manifest_path)
    fingerprint = runner.digest(manifest_path)
    if manifest.get('cohort') != 'LS':
        raise ValueError('Expected frozen LS run manifest')
    root = Path(cfg['project_root'])
    for path in (config_path, Path(cfg['participants']), root / 'scripts/cohort_LS/runner_LS.py',
                 Path(cfg['ls']['runtime_dir']) / 'workflow/Snakefile_LS'):
        if manifest['files'].get(str(path)) != runner.digest(path):
            raise ValueError('Frozen control file changed: ' + str(path))
    for name in ('audit_guard_LS.json', 'pilot_verification_LS.json'):
        record = runner.read_json(logs / name)
        if record.get('status') != 'pass' or record.get('run_manifest_sha256') != fingerprint:
            raise ValueError('A passed, matching audit and pilot are required: ' + name)
        if name == 'audit_guard_LS.json' and runner.digest(Path(record['report'])) != record['report_sha256']:
            raise ValueError('Saved audit report changed')
    return fingerprint


def progress(runner, cfg):
    ready, candidates, failed = [], [], []
    for row in runner.participants(cfg):
        person = row['participant_id']
        states = [runner.checked_status(cfg, person, stage) for stage in PRE_STAGES]
        if all(state and state['status'] == 'pass' for state in states):
            ready.append(person)
        elif any(state and state['status'] == 'fail' for state in states):
            failed.append(person)
        elif runner.checked_status(cfg, person, 't1') or runner.checked_status(cfg, person, 'cleanup'):
            raise ValueError('Finished downstream state has missing upstream records: ' + person)
        else:
            candidates.append((sum(state is not None for state in states), person))
    # Prefer partly completed subjects, preserving all existing status outputs.
    return ready, [person for _, person in sorted(candidates, key=lambda item: (-item[0], item[1]))], failed


def descendants(processes, parent):
    related = {parent}
    while True:
        added = {p['pid'] for p in processes if p['ppid'] in related}
        if added.issubset(related):
            return related
        related.update(added)


def live_context(cfg, config_path):
    processes, visible, errors = monitor.read_processes()
    if not visible:
        raise RuntimeError('Cannot safely inspect LS process identities: ' + '; '.join(errors))
    live = [p for p in processes if p['state'] not in ('Z', 'X')]
    master, errors = monitor._runner_state(Path(cfg['ls']['log_dir']), config_path, {p['pid']: p for p in live})
    if master['status'] in ('unknown', 'stale_pid'):
        raise RuntimeError('LS runner identity is not verified: ' + '; '.join(errors))
    related = descendants(live, master['pid']) if master['status'] == 'running' else set()
    root = str(cfg['project_root'])
    for p in live:
        argv = p['argv']
        pipeline = ('snakemake' in argv or 'substain_features.cli' in argv or
                    any(Path(a).name in ('snakemake', 'runner_LS.py', 'finish_ws1_v1_0_9.sh', 'gb_verify_then_all.sh', 'run_pipeline.sh') for a in argv))
        if pipeline and p['pid'] not in related:
            cwd = str((Path('/proc') / str(p['pid']) / 'cwd').resolve())
            if root in ' '.join(argv) or cwd == root or cwd.startswith(root + '/'):
                raise RuntimeError('Another GB/LS analysis is active; no signal sent: PID=' + str(p['pid']))
    return live, master, related


def graceful_supported():
    import snakemake
    from snakemake.scheduler import JobScheduler
    if snakemake.__version__ != '7.32.4':
        raise RuntimeError('Graceful handover was validated only for Snakemake 7.32.4')
    if ('signal.signal(signal.SIGTERM, self.exit_gracefully)' not in inspect.getsource(JobScheduler.__init__)
            or 'self._user_kill = "graceful"' not in inspect.getsource(JobScheduler.exit_gracefully)):
        raise RuntimeError('Installed Snakemake graceful SIGTERM handler differs; nothing stopped')


def signal_exact(runner, process, require_handler=False):
    current = runner.process_identity(process['pid'])
    if (not current or current['state'] in ('Z', 'X') or
            any(current[key] != process[key] for key in ('start_ticks', 'pgid', 'argv'))):
        raise RuntimeError('Process identity changed before signal; no signal sent')
    if require_handler:
        fields = dict(line.split(':', 1) for line in (Path('/proc') / str(process['pid']) / 'status').read_text().splitlines())
        if not int(fields['SigCgt'].strip(), 16) & (1 << (signal.SIGTERM - 1)):
            raise RuntimeError('Scheduler has no SIGTERM handler yet; no signal sent')
    os.kill(process['pid'], signal.SIGTERM)  # One verified PID, never its process group.


def drain(runner, cfg, config_path, record, pause=time.sleep):
    graceful_supported()
    signalled = None
    cancelled_waiters = set()
    while True:
        live, master, related = live_context(cfg, config_path)
        if master['status'] != 'running':
            runner.require_idle(Path(cfg['project_root']))
            return
        schedulers = [p for p in live if p['ppid'] == master['pid'] and '-m' in p['argv'] and 'snakemake' in p['argv']
            and monitor._config_context(p['argv'], config_path) is True
            and monitor._option(p['argv'], ('--snakefile',)) == [str(Path(cfg['ls']['runtime_dir']) / 'workflow/Snakefile_LS')]]
        if len(schedulers) > 1:
            raise RuntimeError('Multiple LS schedulers found; nothing stopped')
        if schedulers:
            scheduler = schedulers[0]
            identity = (scheduler['pid'], scheduler['start_ticks'])
            if signalled is None:
                record('draining', scheduler_pid=scheduler['pid'], previous_runner_pid=master['pid'])
                signal_exact(runner, scheduler, require_handler=True)
                signalled = identity
            elif identity != signalled:
                raise RuntimeError('LS scheduler changed during handover; refusing another signal')
            # A rolling admission waiter can never finish once new computation is
            # stopped. Cancel only this lightweight token waiter, not an MRI job.
            children = descendants(live, scheduler['pid'])
            admission = str(Path(cfg['project_root']) / 'scripts/rolling_admission.py')
            for p in live:
                if (p['pid'] not in children or admission not in p['argv'] or 'wait' not in p['argv']
                        or (p['pid'], p['start_ticks']) in cancelled_waiters):
                    continue
                outputs = monitor._option(p['argv'], ('--output',))
                if len(outputs) != 1:
                    raise RuntimeError('Unexpected admission waiter arguments')
                Path(outputs[0]).resolve().relative_to(Path(cfg['ls']['log_dir']) / 'queue_LS')
                signal_exact(runner, p)
                cancelled_waiters.add((p['pid'], p['start_ticks']))
                record('draining', cancelled_admission_pids=sorted(pid for pid, _ in cancelled_waiters))
        else:
            record('waiting_scheduler' if signalled is None else 'draining')
        pause(10)


def warmup_command(runner, cfg, config_path, people):
    targets = [str(Path(cfg['derivatives']) / ('sub-' + person) / 'status/wmh.json') for person in people]
    return preserve_completed(runner.snakemake_command(cfg, config_path, targets))


def preserve_completed(args):
    args = list(args)
    # Changing only admission mode must not rerun completed steps for metadata changes.
    args[args.index('--config'):args.index('--config')] = ['--rerun-triggers', 'mtime']
    return args


def collect(runner, cfg, config_path, record):
    while True:
        ready, candidates, failed = progress(runner, cfg)
        record('collecting', ready_count=len(ready), upstream_failed_count=len(failed))
        if len(ready) >= GOAL:
            record('threshold_met', ready_count=len(ready), ready_ids=ready, upstream_failed_ids=failed)
            return
        selected = candidates[:GOAL - len(ready)]
        if not selected:
            raise RuntimeError('Fewer than 80 subjects passed upstream stages; failures retained, T1 not released')
        record('collecting', selected_ids=selected)
        subprocess.run(warmup_command(runner, cfg, config_path, selected), cwd=cfg['ls']['runtime_dir'],
                       env=runner.env_for(cfg), check=True)
        # Ordinary recorded stage failures are retained; missing output is an infrastructure failure.
        if any(runner.checked_status(cfg, person, 'wmh') is None for person in selected):
            raise RuntimeError('Warmup command returned without all selected WMH status records')


def worker(runner, cfg, config_path):
    logs = Path(cfg['ls']['log_dir'])
    state_path = logs / 'warmup80_LS.json'
    state = {}
    def record(phase, **fields):
        state.update(fields, phase=phase, updated_at=datetime.now(timezone.utc).isoformat())
        runner.write_same(state_path, runner.canonical(state))
        print('LS_WARMUP80 phase={} ready={}/80'.format(phase, state.get('ready_count', '?')), flush=True)
    with runner.locked(logs / 'warmup80_LS.lock', wait_seconds=15):
        state.update(runner.read_json(state_path))
        if state.get('pid') != os.getpid() or state.get('config') != str(config_path):
            raise RuntimeError('Warmup worker does not match its launch record')
        try:
            fingerprint = preflight(runner, cfg, config_path)
            ready, _, _ = progress(runner, cfg)
            live, master, _ = live_context(cfg, config_path)
            if len(ready) >= GOAL and master['status'] == 'running':
                record('already_satisfied', ready_count=len(ready), run_manifest_sha256=fingerprint)
                return
            record('preflight', run_manifest_sha256=fingerprint, ready_count=len(ready))
            drain(runner, cfg, config_path, record)
            with runner.locked(logs / 'operation_LS.lock'):
                runner.require_idle(Path(cfg['project_root']))
                if preflight(runner, cfg, config_path) != fingerprint:
                    raise RuntimeError('Frozen LS identity changed during handover')
                collect(runner, cfg, config_path, record)
            # Reuse the installed worker in this process. Only the scheduler's
            # metadata trigger changes: new rolling-token filenames must not
            # rerun finished MRI stages. Frozen code/config/input checks remain.
            original_command = runner.snakemake_command
            runner.snakemake_command = lambda *args, **kwargs: preserve_completed(original_command(*args, **kwargs))
            try:
                record('parallel')
                runner.worker(cfg, config_path, 'resume')
            finally:
                runner.snakemake_command = original_command
            record('complete')
        except Exception as error:
            record('error', error='{}: {}'.format(type(error).__name__, error))
            raise


def start(runner, cfg, config_path):
    logs = Path(cfg['ls']['log_dir'])
    with runner.locked(logs / 'warmup80_LS.lock'):
        preflight(runner, cfg, config_path)
        # Fail before launching if the installed scheduler cannot drain safely.
        graceful_supported()
        live_context(cfg, config_path)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        log = logs / (stamp + '_warmup80_LS.log')
        state_path = logs / 'warmup80_LS.json'
        if state_path.exists():
            prior = runner.read_json(state_path)
            old = runner.process_identity(prior.get('pid', 0))
            if old and old['state'] not in ('Z', 'X') and old['start_ticks'] == prior.get('start_ticks'):
                raise RuntimeError('An LS warmup controller is already active: PID=' + str(old['pid']))
            runner.write_same(logs / (stamp + '_previous_warmup80_LS.json'), state_path.read_bytes())
        args = [cfg['execution']['core_python'], str(HERE / 'warmup80_LS.py'), 'worker', '--config', str(config_path)]
        with log.open('xb') as output:
            process = subprocess.Popen(args, cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        identity = runner.process_identity(process.pid)
        if not identity:
            raise RuntimeError('Warmup controller exited immediately; see ' + str(log))
        state = {key: identity[key] for key in ('pid', 'pgid', 'start_ticks')}
        state.update(phase='starting', goal=GOAL, config=str(config_path), script=str(HERE / 'warmup80_LS.py'),
                     started_at=stamp, log=str(log), source_sha256=runner.digest(HERE / 'warmup80_LS.py'))
        runner.write_same(state_path, runner.canonical(state))
        runner.write_same(logs / 'warmup80_LS.pid', str(process.pid) + '\n')
        runner.write_same(logs / 'warmup80_LS.logpath', str(log) + '\n')
        print('LS_WARMUP80_STARTED PID={} log={}'.format(process.pid, log), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'worker'))
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        path = args.config.resolve(strict=True)
        runner, cfg = installed_runner(path)
        (start if args.action == 'start' else worker)(runner, cfg, path)
        return 0
    except Exception as error:
        print('LS_WARMUP80_ERROR: {}: {}'.format(type(error).__name__, error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
