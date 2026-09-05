"""Focused deployment safety tests; no access to real WS1 processes or data."""
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import deploy_ws1_t1_hybrid as deploy


BUNDLE = Path(__file__).resolve().parent


class DeployTests(unittest.TestCase):
    def test_term_requires_explicit_marker_and_this_stop_window(self):
        before = dict(before_status=None)
        def payload(error, stamp=101):
            return dict(status='fail', timestamp_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
                        details=dict(error=error))
        for error in ('died with <Signals.SIGTERM: 15>', 'returncode -15', 'exit status 143'):
            self.assertTrue(deploy.interrupted(payload(error), before, 100, 102, 101))
        for error in ('CUDA out of memory', 'terminated unexpectedly', 'bad affine'):
            self.assertFalse(deploy.interrupted(payload(error), before, 100, 102, 101))
        self.assertFalse(deploy.interrupted(payload('SIGTERM', 99), before, 100, 102, 101))
        self.assertFalse(deploy.interrupted(payload('SIGTERM'), before, 100, 102, 99))
        self.assertFalse(deploy.interrupted(payload('SIGTERM'), dict(before_status=dict(status='pass')), 100, 102, 101))

    def test_only_confirmed_new_term_status_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deriv, archive = root / 'deriv', root / 'archive'
            deriv.mkdir(); archive.mkdir()
            active = {}
            for subject, status, error, stamp in [('pass', 'pass', '', 101), ('old', 'fail', 'SIGTERM', 90),
                                                  ('term', 'fail', 'SIGTERM', 101)]:
                source = deriv / ('sub-' + subject) / 'status/t1.json'
                source.parent.mkdir(parents=True)
                source.write_text(json.dumps(dict(status=status, details=dict(error=error),
                    timestamp_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat())))
                os.utime(source, (stamp, stamp))
                active[str(source)] = dict(before_sha256=deploy.digest(source) if subject == 'old' else None,
                                          before_status=None)
            deploy.archive_interrupted(active, deriv, archive, 100, 102)
            self.assertTrue((deriv / 'sub-pass/status/t1.json').exists())
            self.assertTrue((deriv / 'sub-old/status/t1.json').exists())
            self.assertFalse((deriv / 'sub-term/status/t1.json').exists())
            self.assertTrue((archive / 'interrupted-status/sub-term/status/t1.json').exists())

    def test_unknown_new_failure_stays_and_blocks_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deriv, archive = root / 'deriv', root / 'archive'
            deriv.mkdir(); archive.mkdir()
            source = deriv / 't1.json'
            source.write_text(json.dumps(dict(status='fail', details=dict(error='OOM'))))
            with self.assertRaisesRegex(RuntimeError, '无法确认'):
                deploy.archive_interrupted({str(source): dict(before_sha256=None, before_status=None)},
                                           deriv, archive, 0, 1)
            self.assertTrue(source.exists())

    def test_safe_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises((ValueError, RuntimeError)):
                deploy.safe(root / '../outside', root, False)

    def test_python_existing_symlink_is_accepted(self):
        python = Path('/usr/bin/python3')
        if not python.is_symlink():
            self.skipTest('requires Linux system Python symlink')
        self.assertTrue(python.is_file() and os.access(python, os.X_OK))

    def test_process_identity_ignores_running_sleeping_state(self):
        self.assertEqual(deploy.identity(dict(pid=1, birth='1', state='R')),
                         deploy.identity(dict(pid=1, birth='1', state='S')))
        self.assertNotEqual(deploy.identity(dict(pid=1, birth='1')),
                            deploy.identity(dict(pid=1, birth='2')))

    def make_fixture(self, root):
        # Reconstruct a supported old tree by reversing the published increment.
        source = Path(os.environ['HYBRID_SOURCE_ROOT'])
        for rel in deploy.FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((source / rel).read_text(encoding='utf-8').encode('utf-8'))
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        subprocess.run(['git', 'apply', '-R', str(BUNDLE / 'ws1_t1_hybrid.patch')], cwd=root, check=True)
        for rel in ('scripts/start_ws1_v1_0_9.sh', 'scripts/finish_ws1_v1_0_9.sh',
                    'src/substain_features/pipeline.py'):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((source / rel).read_text(encoding='utf-8').encode('utf-8'))
        (root / 'config').mkdir()
        (root / 'config/config.yaml').write_text('derivatives: derivatives/substain_features\nexecution:\n  cpu_threads_per_job: 4\n')
        (root / 'derivatives/substain_features').mkdir(parents=True)
        for name in ('core-venv', 't1', 'wmh'):
            executable = root / 'envs' / name / 'bin/python'
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/sh\nexit 0\n')
            executable.chmod(0o755)
        return {rel: (root / rel).read_bytes() for rel in deploy.FILES}

    def test_both_known_patch_contexts(self):
        if not os.environ.get('HYBRID_SOURCE_ROOT'):
            self.skipTest('set HYBRID_SOURCE_ROOT to patched simulation')
        for variant in (1, 2):
            with self.subTest(wmh_tokens=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_fixture(root)
                patchfile = BUNDLE / 'ws1_gpu_throughput_first.patch'
                if variant == 1:
                    subprocess.run(['git', 'apply', '-R', str(patchfile)], cwd=root, check=True)
                deploy.patch_snakefile(root / deploy.FILES[0])
                include = '--include=' + deploy.FILES[1]
                subprocess.run(['git', 'apply', '--check', include,
                                str(BUNDLE / 'ws1_t1_hybrid.patch')], cwd=root, check=True)
                subprocess.run(['git', 'apply', include,
                                str(BUNDLE / 'ws1_t1_hybrid.patch')], cwd=root, check=True)
                snake = (root / deploy.FILES[0]).read_text()
                self.assertIn('cpu_fallback=True', snake)
                self.assertIn('gpu=GPU_SLOTS_PER_DEVICE if PROFILE == "gpu" else 0', snake)
                self.assertEqual(deploy.text_digest(root / deploy.FILES[1]),
                                 json.loads((BUNDLE / 'ws1_t1_hybrid_manifest.json').read_text())['new_pool_sha256'])

    def test_observed_ws1_snakefile_hash_is_allowlisted(self):
        manifest = json.loads((BUNDLE / 'ws1_t1_hybrid_manifest.json').read_text())
        self.assertIn('f29d004d86a1d27859272b6326f98ffd2b26a576c8608aa7232595c084e4b437',
                      manifest['old_snake_sha256'])

    def test_semantic_snakefile_patch_rejects_unknown_t1_context(self):
        if not os.environ.get('HYBRID_SOURCE_ROOT'):
            self.skipTest('set HYBRID_SOURCE_ROOT to patched simulation')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            snake = root / deploy.FILES[0]
            snake.write_text(snake.read_text().replace(
                'gpu=1 if PROFILE == "gpu" else 0\n    threads: CPU_HEAVY_THREADS',
                'gpu=99\n    threads: CPU_HEAVY_THREADS'))
            with self.assertRaisesRegex(RuntimeError, 'T1调度令牌'):
                deploy.patch_snakefile(snake)

    def test_validation_failure_restores_source_without_restart(self):
        if not os.environ.get('HYBRID_SOURCE_ROOT'):
            self.skipTest('set HYBRID_SOURCE_ROOT to patched simulation')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = self.make_fixture(root)
            leader = dict(pid=99999999, pgid=99999999, birth='test', args=[])
            member = dict(args=['snakemake', 'gpu_devices=0', 'rolling_window=200', 'gpu_slots_per_device=2'])
            calls = []
            def fake_run(args, cwd, log=None, timeout=120):
                args = [str(a) for a in args]
                calls.append(args)
                if args[0] == 'git':
                    subprocess.run(args, cwd=cwd, check=True, capture_output=True)
                if '--list' in args:
                    raise RuntimeError('injected parse failure')
                return ''
            with patch.object(deploy, 'ROOT', root), patch.object(deploy.sys, 'argv', ['deploy', str(root)]), \
                 patch.object(deploy, 'validate_leader', return_value=leader), \
                 patch.object(deploy, 'group', side_effect=[[member], [], [], []]), \
                 patch.object(deploy, 'project_workers', return_value=[]), \
                 patch.object(deploy, 'run', side_effect=fake_run), patch.object(deploy.os, 'killpg', create=True) as kill:
                with self.assertRaisesRegex(RuntimeError, 'injected parse failure'):
                    deploy.main()
                kill.assert_called_once()
            for rel in deploy.FILES:
                self.assertEqual((root / rel).read_bytes(), before[rel])
            self.assertFalse(any(args[0] == 'bash' and args[-1].endswith('start_ws1_v1_0_9.sh') and '-n' not in args
                                 for args in calls))

    def test_unknown_pool_aborts_before_stop(self):
        if not os.environ.get('HYBRID_SOURCE_ROOT'):
            self.skipTest('set HYBRID_SOURCE_ROOT to patched simulation')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            with (root / deploy.FILES[1]).open('a') as handle:
                handle.write('\n# unknown local edit\n')
            with patch.object(deploy, 'ROOT', root), patch.object(deploy.sys, 'argv', ['deploy', str(root)]), \
                 patch.object(deploy, 'run', return_value=''), patch.object(deploy.os, 'killpg', create=True) as kill:
                with self.assertRaisesRegex(RuntimeError, '未知GPU锁'):
                    deploy.main()
                kill.assert_not_called()

    def test_term_timeout_does_not_patch_or_kill(self):
        if not os.environ.get('HYBRID_SOURCE_ROOT'):
            self.skipTest('set HYBRID_SOURCE_ROOT to patched simulation')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = self.make_fixture(root)
            leader = dict(pid=99999999, pgid=99999999, birth='test', args=[])
            member = dict(args=['snakemake', 'gpu_devices=0', 'rolling_window=200', 'gpu_slots_per_device=2'])
            def fake_run(args, cwd, log=None, timeout=120):
                if str(args[0]) == 'git':
                    subprocess.run([str(a) for a in args], cwd=cwd, check=True, capture_output=True)
                return ''
            with patch.object(deploy, 'ROOT', root), patch.object(deploy.sys, 'argv', ['deploy', str(root)]), \
                 patch.object(deploy, 'validate_leader', return_value=leader), \
                 patch.object(deploy, 'group', return_value=[member]), \
                 patch.object(deploy, 'project_workers', return_value=[]), \
                 patch.object(deploy, 'run', side_effect=fake_run), \
                 patch.object(deploy.time, 'monotonic', side_effect=[0, 100]), \
                 patch.object(deploy.os, 'killpg', create=True) as kill:
                with self.assertRaisesRegex(RuntimeError, '60秒仍有活动进程'):
                    deploy.main()
                kill.assert_called_once_with(99999999, deploy.signal.SIGTERM)
            for rel in deploy.FILES:
                self.assertEqual((root / rel).read_bytes(), before[rel])

    def test_success_restarts_only_after_validation(self):
        if not os.environ.get('HYBRID_SOURCE_ROOT'):
            self.skipTest('set HYBRID_SOURCE_ROOT to patched simulation')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            leader = dict(pid=99999999, pgid=99999999, birth='test', args=[])
            member = dict(args=['snakemake', 'gpu_devices=0', 'rolling_window=200', 'gpu_slots_per_device=2', 'profile=gpu'])
            state, calls = dict(stopped=False, restarted=False), []
            def stop(*args):
                state['stopped'] = True
            def members(*args):
                return [member] if not state['stopped'] or state['restarted'] else []
            def fake_run(args, cwd, log=None, timeout=120):
                args = [str(a) for a in args]
                calls.append(args)
                if args[0] == 'git':
                    subprocess.run(args, cwd=cwd, check=True, capture_output=True)
                if args[0] == 'bash' and len(args) == 2:
                    self.assertTrue(any('--list' in cmd for cmd in calls))
                    self.assertTrue(any('--unlock' in cmd for cmd in calls))
                    state['restarted'] = True
                    (root / 'logs').mkdir()
                    logpath = root / 'logs/new.log'
                    logpath.write_text('gpu_slots=2\n')
                    (root / 'logs/full_run_v1.0.9.logpath').write_text(str(logpath))
                return ''
            with patch.object(deploy, 'ROOT', root), patch.object(deploy.sys, 'argv', ['deploy', str(root)]), \
                 patch.object(deploy, 'validate_leader', return_value=leader), patch.object(deploy, 'process', return_value=leader), \
                 patch.object(deploy, 'group', side_effect=members), patch.object(deploy, 'project_workers', return_value=[]), \
                 patch.object(deploy, 'run', side_effect=fake_run), patch.object(deploy.os, 'killpg', side_effect=stop, create=True):
                deploy.main()
            self.assertTrue(state['restarted'])
            self.assertIn('cpu_fallback=True', (root / deploy.FILES[0]).read_text())
            self.assertEqual(len(list((root / 'archive').glob('*/deployment.json'))), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
