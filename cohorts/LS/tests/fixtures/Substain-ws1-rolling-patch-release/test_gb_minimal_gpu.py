import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import gb_minimal as patch
BASELINE = Path(__file__).resolve().parents[1] / "Substain-v1.0.9-remote-sim/src/substain_features/gpu_pool.py"


class Harness:
    def __init__(self, root):
        self.root = root
        original = BASELINE.read_text()
        start = original.index("def run_with_gpu_lock(")
        end = original.index("\ndef _command_stage(", start)
        self.pool_file = root / "actual_archived_gpu_pool.py"
        self.pool_file.write_text(original[:start] + patch.GPU_FUNCTION + "\n\n" + original[end:])
        self.processes = []
        package = root / "src/substain_features"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "cli.py").write_text('''import json,os,sys,time
from pathlib import Path
args=sys.argv
name=args[args.index('--participant-id')+1]
root=Path(os.environ['GB_GPU_TEST_ROOT'])
data=dict(device=os.environ['CUDA_VISIBLE_DEVICES'], profile=args[args.index('--profile')+1],
          assigned={k:v for k,v in os.environ.items() if k.startswith('SUBSTAIN_ASSIGNED_GPU')})
(root/(name+'.started')).write_text(json.dumps(data))
if name=='error': raise SystemExit(7)
while not (root/(name+'.release')).exists(): time.sleep(.01)
''')
        binary = root / "bin"
        binary.mkdir()
        nvidia = binary / "nvidia-smi"
        nvidia.write_text('''#!/usr/bin/env python3
import json,os,sys
gpu=next(v.split('=',1)[1] for v in sys.argv if v.startswith('--id='))
value=json.loads(os.environ['GB_GPU_MEMORY'])[gpu]
if value=='fail': raise SystemExit(1)
print(value)
''')
        nvidia.chmod(0o755)
        self.env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                        PYTHONPATH=str(root / "src"), GB_GPU_TEST_ROOT=str(root),
                        SUBSTAIN_ASSIGNED_GPU="stale", GB_GPU_MEMORY=json.dumps({"0": 40000, "1": 40000}))

    def launch(self, name, wmh=False, memory=None):
        command = [sys.executable, str(self.pool_file), "--gpu-ids", "0,1", "--lock-dir", str(self.root / "locks"),
                   "--slots-per-gpu", "2", "--slots-required", "2" if wmh else "1",
                   "--min-free-memory-mib", "20480" if wmh else "12288"]
        if not wmh:
            command.append("--cpu-fallback")
        command += ["--", sys.executable, "-m", "substain_features.cli", "stage", "wmh-seg" if wmh else "t1",
                    "--participant-id", name, "--profile", "gpu"]
        env = dict(self.env)
        if memory:
            env['GB_GPU_MEMORY'] = json.dumps(memory)
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(child)
        return child

    def started(self, name):
        path = self.root / (name + '.started')
        deadline = time.monotonic() + 15
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert path.exists(), name
        return json.loads(path.read_text())

    def release(self, name, child):
        (self.root / (name + '.release')).touch()
        out, err = child.communicate(timeout=15)
        assert child.returncode == 0, (out, err)

    def close(self):
        for child in self.processes:
            if child.poll() is None:
                child.terminate()
            child.communicate(timeout=15)


@pytest.fixture
def harness(tmp_path):
    instance = Harness(tmp_path)
    yield instance
    instance.close()


def test_gpu0_busy_uses_gpu1_then_four_t1_slots(harness):
    children = []
    for name, expected in (("a", "0"), ("b", "0"), ("c", "1"), ("d", "1"), ("e", "")):
        child = harness.launch(name)
        data = harness.started(name)
        assert data['device'] == expected
        assert data['profile'] == ('cpu' if expected == '' else 'gpu')
        if expected == '':
            assert data['assigned'] == {}
        children.append((name, child))
    for name, child in children:
        harness.release(name, child)


@pytest.mark.parametrize("wmh", [False, True])
def test_gpu0_memory_failure_still_tries_gpu1(harness, wmh):
    child = harness.launch('memory', wmh=wmh, memory={'0': 'fail', '1': 40000})
    assert harness.started('memory')['device'] == '1'
    harness.release('memory', child)


@pytest.mark.parametrize("wmh", [False, True])
def test_all_devices_low_memory_cpu(harness, wmh):
    child = harness.launch('cpu', wmh=wmh, memory={'0': 100, '1': 'fail'})
    data = harness.started('cpu')
    assert data['device'] == '' and data['profile'] == 'cpu' and data['assigned'] == {}
    harness.release('cpu', child)


def test_wmh_on_gpu0_does_not_block_t1_on_gpu1(harness):
    wm = harness.launch('wm', wmh=True)
    assert harness.started('wm')['device'] == '0'
    t1 = harness.launch('t1')
    assert harness.started('t1')['device'] == '1'
    second = harness.launch('wm2', wmh=True, memory={'0': 0, '1': 40000})
    time.sleep(.3)
    assert not (harness.root / 'wm2.started').exists()
    fallback = harness.launch('new_t1')
    assert harness.started('new_t1')['profile'] == 'cpu'
    harness.release('t1', t1)
    assert harness.started('wm2')['device'] == '1'
    harness.release('new_t1', fallback)
    harness.release('wm2', second)
    harness.release('wm', wm)


def test_two_wmh_cards_and_error_release_locks(harness):
    a = harness.launch('wm0', wmh=True)
    assert harness.started('wm0')['device'] == '0'
    b = harness.launch('wm1', wmh=True)
    assert harness.started('wm1')['device'] == '1'
    harness.release('wm0', a)
    harness.release('wm1', b)
    error = harness.launch('error')
    harness.started('error')
    error.communicate(timeout=10)
    assert error.returncode == 7
    after = harness.launch('after')
    assert harness.started('after')['device'] == '0'
    after.terminate()
    after.communicate(timeout=10)
    assert after.returncode == 143
    next_child = harness.launch('next')
    assert harness.started('next')['device'] == '0'
    harness.release('next', next_child)


def test_wmh_probes_second_gpu_before_waiting_on_first(harness):
    t1 = harness.launch('occupied')
    assert harness.started('occupied')['device'] == '0'
    wm = harness.launch('free_card', wmh=True)
    assert harness.started('free_card')['device'] == '1'
    harness.release('free_card', wm)
    harness.release('occupied', t1)


def test_wmh_waits_for_project_slots_before_memory_fallback(harness):
    t1 = harness.launch('occupant')
    assert harness.started('occupant')['device'] == '0'
    wm = harness.launch('wait_then_cpu', wmh=True, memory={'0': 100, '1': 100})
    time.sleep(.5)
    assert not (harness.root / 'wait_then_cpu.started').exists()
    new = harness.launch('on_other_card')
    assert harness.started('on_other_card')['device'] == '1'
    harness.release('occupant', t1)
    time.sleep(.4)
    assert not (harness.root / 'wait_then_cpu.started').exists()
    harness.release('on_other_card', new)
    assert harness.started('wait_then_cpu')['profile'] == 'cpu'
    harness.release('wait_then_cpu', wm)


@pytest.mark.parametrize('wmh', [False, True])
def test_both_gpu_memory_queries_fail_to_cpu(harness, wmh):
    child = harness.launch('failed_queries', wmh=wmh, memory={'0': 'fail', '1': 'fail'})
    assert harness.started('failed_queries')['profile'] == 'cpu'
    harness.release('failed_queries', child)
