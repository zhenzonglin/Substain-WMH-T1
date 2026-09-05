#!/usr/bin/env python3
"""Linux process tests without CUDA/models; --snakemake also runs the real scheduler."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CHILD = r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv
subject = args[args.index('--participant-id') + 1]
profile = args[args.index('--profile') + 1]
root = Path(os.environ['HYBRID_TEST_ROOT'])
data = dict(subject=subject, profile=profile, pid=os.getpid(),
            cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
            assigned={k:v for k,v in os.environ.items() if k.startswith('SUBSTAIN_ASSIGNED_GPU')},
            threads=os.environ.get('OMP_NUM_THREADS'), started=time.time())
(root / (subject + '.started')).write_text(json.dumps(data))
if subject == 'fail':
    raise SystemExit(7)
while not (root / (subject + '.release')).exists() and not (root / 'release_all').exists():
    time.sleep(.02)
data['finished'] = time.time()
(root / (subject + '.finished')).write_text(json.dumps(data))
output = root / 'out' / (subject + '.json')
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(data))
'''


def wait_for(predicate, message, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError(message)


class Harness:
    def __init__(self, root, source):
        self.root, self.source = root, source
        self.procs = []
        package = root / 'src' / 'substain_features'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        (package / 'cli.py').write_text(CHILD)
        (package / 'gpu_pool.py').write_bytes((source / 'src/substain_features/gpu_pool.py').read_bytes())
        self.pool = package / 'gpu_pool.py'
        self.env = dict(os.environ, PYTHONPATH=str(root / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''), HYBRID_TEST_ROOT=str(root),
                        CUDA_VISIBLE_DEVICES='0', SUBSTAIN_ASSIGNED_GPU='stale',
                        SUBSTAIN_ASSIGNED_GPU_SLOTS='stale', OMP_NUM_THREADS='4')

    def launch(self, name, wmh=False):
        cmd = [sys.executable, str(self.pool), '--gpu-ids', '0', '--lock-dir', str(self.root / 'locks'),
               '--slots-per-gpu', '2', '--slots-required', '2' if wmh else '1']
        if not wmh:
            cmd.append('--cpu-fallback')
        cmd += ['--', sys.executable, '-m', 'substain_features.cli', 'stage', 'wmh-seg' if wmh else 't1',
                '--participant-id', name, '--profile', 'gpu']
        process = subprocess.Popen(cmd, env=self.env)
        self.procs.append(process)
        return process

    def started(self, name, profile):
        path = self.root / (name + '.started')
        wait_for(path.exists, 'child did not start: ' + name)
        data = json.loads(path.read_text())
        assert data['profile'] == profile, data
        if profile == 'cpu':
            assert data['cuda'] == '' and not data['assigned'], data
        else:
            assert data['cuda'] == '0' and data['assigned']['SUBSTAIN_ASSIGNED_GPU'] == '0', data
        return data

    def release(self, name, proc):
        (self.root / (name + '.release')).touch()
        assert proc.wait(timeout=10) == 0

    def close(self):
        (self.root / 'release_all').touch()
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=10)


def process_tests(h):
    a = h.launch('a'); h.started('a', 'gpu')
    b = h.launch('b'); h.started('b', 'gpu')
    c = h.launch('c'); h.started('c', 'cpu')
    h.release('a', a)
    d = h.launch('d'); h.started('d', 'gpu')
    # The whole-card waiter owns admission before either occupied slot is released.
    w = h.launch('wmh', True)
    gate = h.root / 'locks/gpu-0-admission.lock'
    def gate_busy():
        with gate.open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True
    wait_for(gate_busy, 'WMH did not reserve admission')
    e = h.launch('e'); h.started('e', 'cpu')
    assert not (h.root / 'wmh.started').exists()
    h.release('b', b)
    assert not (h.root / 'wmh.started').exists()
    h.release('d', d)
    wm = h.started('wmh', 'gpu')
    assert wm['assigned']['SUBSTAIN_ASSIGNED_GPU_SLOTS'] == '2'
    f = h.launch('f'); h.started('f', 'cpu')
    h.release('wmh', w)
    g = h.launch('g'); h.started('g', 'gpu'); h.release('g', g)
    for name, proc in [('c', c), ('e', e), ('f', f)]:
        h.release(name, proc)
    fail = h.launch('fail'); h.started('fail', 'gpu')
    assert fail.wait(timeout=10) == 7
    term = h.launch('term'); td = h.started('term', 'gpu')
    term.terminate()  # wrapper-only TERM also ends its child, before releasing locks
    assert term.wait(timeout=10) == 143
    assert not Path('/proc/{}'.format(td['pid'])).exists()
    x = h.launch('x'); h.started('x', 'gpu')
    y = h.launch('y'); h.started('y', 'gpu')
    waiter = h.launch('term_waiter', True)
    wait_for(gate_busy, 'WMH waiter did not reserve gate')
    waiter.terminate(); waiter.wait(timeout=10)
    h.release('x', x); h.release('y', y)
    after = h.launch('after'); h.started('after', 'gpu'); h.release('after', after)
    # Concurrent race: never allocate more than two GPU slots.
    pairs = [(str(i), h.launch('race' + str(i))) for i in range(12)]
    wait_for(lambda: len(list(h.root.glob('race*.started'))) == 12, 'race launches stalled')
    devices = [json.loads(p.read_text())['profile'] for p in h.root.glob('race*.started')]
    assert 1 <= devices.count('gpu') <= 2 and devices.count('cpu') >= 10, devices
    for name, proc in pairs:
        h.release('race' + name, proc)
    print('PASS: two GPU T1, immediate CPU fallback, WMH admission/exclusion, failure/TERM, races', flush=True)


def scheduler_test(h):
    snake = (h.source / 'workflow/Snakefile').read_text()
    helpers = snake[snake.index('def gpu_prefix('):snake.index('\nrule all:')]
    rule = re.search(r'^rule t1:\n.*?(?=^rule |\Z)', snake, re.M | re.S).group()
    assert 'gpu=0' in rule and 'threads: CPU_HEAVY_THREADS' in rule and 'cpu_fallback=True' in rule
    names = ['sched' + str(i) for i in range(26)]
    (h.root / 'input').mkdir()
    for name in names:
        (h.root / 'input' / (name + '.json')).touch()
    preamble = '''from pathlib import Path
ROOT = Path({root!r})
CORE = T1_PYTHON = {python!r}
GPU_LOCK_DIR = str(ROOT / 'locks')
GPU_DEVICES = '0'
GPU_SLOTS_PER_DEVICE = 2
CPU_HEAVY_THREADS = 4
PROFILE = 'gpu'
CONFIG_FILE = ROOT / 'config.yaml'
PARTICIPANTS = {names!r}
def stage_pattern(stage):
    return str(ROOT / ('input' if stage == 'wmh' else 'out') / '{{participant}}.json')
rule all:
    input: [str(ROOT / 'out' / (name + '.json')) for name in PARTICIPANTS]
'''.format(root=str(h.root), python=sys.executable, names=names)
    (h.root / 'Snakefile').write_text(preamble + '\n' + helpers + '\n' + rule)
    command = [sys.executable, '-m', 'snakemake', '--snakefile', str(h.root / 'Snakefile'),
               '--cores', '96', '--resources', 'gpu=2', 'wmh_exclusive=1', '--printshellcmds', '--scheduler', 'greedy']
    with (h.root / 'snakemake.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=h.root, env=h.env, stdout=log, stderr=subprocess.STDOUT)
        h.procs.append(proc)
        wait_for(lambda: len(list(h.root.glob('sched*.started'))) == 24 or proc.poll() is not None,
                 '96-core scheduler did not admit 24 four-thread T1 tasks', 45)
        assert proc.poll() is None, (h.root / 'snakemake.log').read_text()
        data = [json.loads(p.read_text()) for p in h.root.glob('sched*.started')]
        assert len(data) == 24 and all(d['threads'] == '4' for d in data), data
        assert sum(d['profile'] == 'gpu' for d in data) <= 2
        assert sum(d['profile'] == 'cpu' for d in data) >= 22
        (h.root / 'release_all').touch()
        assert proc.wait(timeout=45) == 0, (h.root / 'snakemake.log').read_text()
    assert len(list((h.root / 'out').glob('sched*.json'))) == 26
    print('PASS: real Snakemake, 24 simultaneous T1 x 4 threads = 96 cores, gpu=2 not a T1 cap', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_root', type=Path)
    parser.add_argument('--snakemake', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='ws1-hybrid-test-') as name:
        h = Harness(Path(name), args.project_root.resolve())
        try:
            process_tests(h)
        finally:
            h.close()
    if args.snakemake:
        with tempfile.TemporaryDirectory(prefix='ws1-hybrid-scheduler-') as name:
            h = Harness(Path(name), args.project_root.resolve())
            try:
                scheduler_test(h)
            finally:
                h.close()
    print('LOCAL VALIDATION PASS (no image/model/A100 execution)')


if __name__ == '__main__':
    main()
