#!/usr/bin/env python3
"""WS1 only: validate before TERM, stage two-file patch, preserve finished results."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/data/usersdir/linzhenzong/Substain')
FILES = ('workflow/Snakefile', 'src/substain_features/gpu_pool.py')
STAGES = ('skullstrip', 'wmh_seg', 'registration', 'lesion', 'wmh', 't1', 'qc', 'cleanup')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(path):
    return hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def safe(path, base, must_exist=True):
    """Reject links in every component, not just the leaf; never traverse during moves."""
    relative = path.relative_to(base)
    current = base
    require(not base.is_symlink() and base.resolve() == base, '非规范根路径: ' + str(base))
    for part in relative.parts:
        require(part not in ('', '.', '..'), '非法路径分量')
        current = current / part
        require(not current.is_symlink(), '拒绝软链接路径: ' + str(current))
    require(not must_exist or path.is_file(), '必需文件不存在: ' + str(path))
    require(path.resolve() == path, '路径解析不一致: ' + str(path))
    return path


def run(args, root, log=None, timeout=120):
    env = dict(os.environ, PYTHONPATH=str(root / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''),
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4')
    result = subprocess.run([str(a) for a in args], cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if log is not None:
        log.write_text(result.stdout, encoding='utf-8')
    require(result.returncode == 0, '命令失败: {}\n{}'.format(args, result.stdout[-6000:]))
    return result.stdout


def process(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], pgid=int(fields[2]), birth=fields[19],
                    uid=proc.stat().st_uid, cwd=str((proc / 'cwd').resolve()),
                    args=[a.decode('utf-8', 'replace') for a in (proc / 'cmdline').read_bytes().split(b'\0') if a])
    except (OSError, IndexError, ValueError):
        return None


def group(pgid):
    return [p for item in Path('/proc').iterdir() if item.name.isdigit()
            for p in [process(int(item.name))] if p and p['pgid'] == pgid and p['state'] != 'Z']


def identity(info):
    return {key: value for key, value in info.items() if key != 'state'}


def project_workers(root):
    result = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        info = process(int(item.name))
        if not info or info['uid'] != os.getuid() or info['state'] == 'Z':
            continue
        text = ' '.join(info['args'])
        if str(root) + '/' in text and any(name in text for name in (
                'substain_features.cli stage', 'substain_features.gpu_pool', 'antsRegistration',
                'NiChart_DLMUSE', '/DLMUSE ', 'wmh_synthseg_inference.py')):
            result.append(info)
    return result


def project_analysis_processes(root):
    """Find this project's schedulers/workers while excluding this deployer."""
    result = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        info = process(int(item.name))
        if not info or info['pid'] == os.getpid() or info['uid'] != os.getuid() or info['state'] == 'Z':
            continue
        text = ' '.join(info['args'])
        if str(root) + '/' in text and any(name in text for name in (
                'finish_ws1_v1_0_9.sh', 'snakemake', 'substain_features.cli stage',
                'substain_features.gpu_pool', 'antsRegistration', 'NiChart_DLMUSE',
                '/DLMUSE ', 'wmh_synthseg_inference.py')):
            result.append(info)
    return result


def validate_leader(root):
    pointer = safe(root / 'logs/full_run_v1.0.9.pid', root)
    value = pointer.read_text().strip()
    require(re.fullmatch(r'[1-9][0-9]*', value), 'PID文件不是有效正整数')
    info = process(int(value))
    expected = str(root / 'scripts/finish_ws1_v1_0_9.sh')
    require(info and info['state'] != 'Z' and info['uid'] == os.getuid(), '主分析未运行或不属于当前用户')
    require(info['pid'] == info['pgid'] and info['pgid'] != os.getpgrp(), '拒绝停止：PID/PGID不匹配或等于当前终端')
    require(info['cwd'] == str(root) and info['args'][-2:] == [expected, '96'],
            '拒绝停止：主进程cwd/入口/96核参数不匹配: ' + str(info))
    return info


def load_resume_snapshot(root, archive_arg):
    """Validate a timeout archive after the recorded process group has exited."""
    archive = safe(Path(archive_arg), root, False)
    require(archive.parent == root / 'archive' and re.fullmatch(r't1-hybrid-[A-Za-z0-9_-]+', archive.name),
            '续部署归档不在固定archive目录')
    require(archive.is_dir() and not archive.is_symlink(), '续部署归档不存在或不是普通目录')
    require(not (archive / 'deployment.json').exists(), '该归档已经完成部署，不得重复续部署')
    before_path = safe(archive / 'before.json', archive)
    snapshot = json.loads(before_path.read_text())
    require(set(('leader', 'active', 'stopped_at', 'derivatives', 'source_sha256')) <= set(snapshot),
            '续部署归档缺少必要字段')
    leader = snapshot['leader']
    require(isinstance(leader, dict) and all(isinstance(leader.get(key), int) for key in ('pid', 'pgid')),
            '续部署归档的PID/PGID无效')
    require(leader['pid'] == leader['pgid'], '续部署归档的PID/PGID不匹配')
    pointer = safe(root / 'logs/full_run_v1.0.9.pid', root).read_text().strip()
    require(pointer == str(leader['pid']), 'PID文件已变化，拒绝使用旧归档续部署')
    require(process(leader['pid']) is None and not group(leader['pgid']),
            '归档记录的旧PID或进程组仍存在，拒绝续部署')
    require(not project_analysis_processes(root), '仍有本项目分析进程，拒绝续部署')
    require(Path(snapshot['derivatives']) == root / 'derivatives/substain_features',
            '续部署归档的derivatives路径不匹配')
    require(isinstance(snapshot['active'], dict) and isinstance(snapshot['stopped_at'], (int, float)),
            '续部署归档的活动状态或停止时间无效')
    require(isinstance(snapshot['source_sha256'], dict), '续部署归档缺少源码校验值')
    for rel in FILES:
        require(snapshot['source_sha256'].get(rel) == digest(safe(root / rel, root)),
                '停止后源码已变化，拒绝续部署: ' + rel)
        archived = safe(archive / 'before' / rel, archive)
        require(digest(archived) == snapshot['source_sha256'][rel],
                '归档备份校验失败: ' + rel)
    return archive, snapshot


def active_snapshot(members, deriv):
    active = {}
    for member in members:
        args = member['args']
        if 'substain_features.cli' not in args or 'stage' not in args or '--participant-id' not in args:
            continue
        stage = args[args.index('stage') + 1].replace('-', '_')
        subject = args[args.index('--participant-id') + 1]
        require(stage in STAGES and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', subject), '未知活动阶段或被试ID')
        path = safe(deriv / ('sub-' + subject) / 'status' / (stage + '.json'), deriv, False)
        active[str(path)] = dict(participant=subject, stage=stage,
                                 before_sha256=digest(path) if path.exists() else None,
                                 before_status=json.loads(path.read_text()) if path.exists() else None)
    return active


def interrupted(payload, before, stopped_at, ended_at, mtime):
    if before.get('before_status') and before['before_status'].get('status') == 'pass':
        return False
    if payload.get('status') != 'fail':
        return False
    try:
        stamp = datetime.fromisoformat(payload['timestamp_utc']).timestamp()
    except (KeyError, ValueError, TypeError):
        return False
    text = json.dumps(payload.get('details', {})).lower()
    term = re.search(r'sigterm|signal\D{0,12}15\b|returncode\s*[=:]?\s*-15\b|exit (?:status|code)\s*143\b', text)
    return bool(term and stopped_at <= stamp <= ended_at and stopped_at <= mtime <= ended_at)


def archive_interrupted(active, deriv, archive, stopped_at, ended_at):
    targets, uncertain = [], []
    for value, before in active.items():
        path = safe(Path(value), deriv, False)
        if not path.exists() or digest(path) == before['before_sha256']:
            continue
        payload = json.loads(path.read_text())
        if interrupted(payload, before, stopped_at, ended_at, path.stat().st_mtime):
            targets.append((path, archive / 'interrupted-status' / path.relative_to(deriv), digest(path)))
        elif payload.get('status') == 'fail':
            uncertain.append(str(path))
    # An ambiguous new failure must be inspected, not silently archived or skipped on restart.
    (archive / 'uncertain-status.json').write_text(json.dumps(uncertain, indent=2))
    require(not uncertain, '有新失败无法确认来自本次TERM，保留原状态并保持停止: ' + str(uncertain))
    for source, destination, expected in targets:
        safe(destination, archive, False)
        require(not destination.exists() and digest(source) == expected, '归档目标已存在或状态发生变化')
    print('确认本次TERM产生的失败状态: {} 个；仅移动以下文件，可从归档恢复。'.format(len(targets)), flush=True)
    for source, destination, expected in targets:
        print(str(source), flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        safe(source, deriv)
        require(digest(source) == expected, '归档前状态发生变化')
        source.rename(destination)


def check_source(root):
    snake = safe(root / FILES[0], root).read_text()
    pool = safe(root / FILES[1], root).read_text()
    start = safe(root / 'scripts/start_ws1_v1_0_9.sh', root).read_text()
    finish = safe(root / 'scripts/finish_ws1_v1_0_9.sh', root).read_text()
    pipeline = safe(root / 'src/substain_features/pipeline.py', root).read_text()
    def rule(name):
        match = re.search(r'^rule {}:\n.*?(?=^rule |\Z)'.format(name), snake, re.M | re.S)
        require(match, '缺少规则: ' + name)
        return match.group()
    require('threads: 8' in rule('registration'), 'registration并非8线程版本')
    require('threads: CPU_HEAVY_THREADS' in rule('t1'), 'T1线程声明不符')
    require('t1_to_ch2better_1Warp.nii.gz' in pipeline and 't1_to_ch2better_1InverseWarp.nii.gz' in pipeline,
            '缺少现有形变场cleanup实现')
    require('ROLLING_ENABLED' in snake and 'rolling_window=${window}' in finish, '缺少滚动窗口实现')
    require(all(s in start for s in ('CPU_THREADS_PER_JOB=4', 'BATCH_SIZE=200', 'CUDA_VISIBLE_DEVICES=0',
                                    'GPU_SLOTS_PER_DEVICE=2', 'finish_ws1_v1_0_9.sh" 96')), '启动资源设置不匹配')
    require(all(s in finish for s in ('gpu_slots=2', '"gpu_devices=0"', '"wmh_exclusive=1"')), 'GPU0/双槽启动参数不符')
    require('gpu_slots_required=GPU_SLOTS_PER_DEVICE' in rule('wmh_segmentation'), 'WMH未锁住两个槽')
    return snake, pool


def replace_once(text, old, new, label):
    require(text.count(old) == 1, '{}上下文数量不是1'.format(label))
    return text.replace(old, new, 1)


def patch_snakefile(path):
    """按已核验的语义片段更新Snakefile，避免补丁被无关行偏移或注释差异阻断。"""
    raw = path.read_bytes()
    require(not raw.startswith(b'\xef\xbb\xbf'), 'Snakefile含BOM')
    eol = '\r\n' if b'\r\n' in raw else '\n'
    text = raw.decode('utf-8').replace('\r\n', '\n')
    require('\r' not in text, 'Snakefile含混合换行符')
    replacements = (
        ('def gpu_prefix(slots_required):',
         'def gpu_prefix(slots_required, cpu_fallback=False):', 'gpu_prefix定义'),
        ('        "-- /usr/bin/env PYTHONPATH={root}/src "',
         '        "{fallback}-- /usr/bin/env PYTHONPATH={root}/src "', '包装器fallback参数'),
        ('        slots_required=slots_required,\n    )',
         '        slots_required=slots_required,\n        fallback="--cpu-fallback " if cpu_fallback else "",\n    )',
         '包装器fallback格式化'),
        ('def execution_command(python, command, needs_gpu=False, gpu_slots_required=1):',
         'def execution_command(python, command, needs_gpu=False, gpu_slots_required=1, cpu_fallback=False):',
         'execution_command定义'),
        ('        return gpu_prefix(gpu_slots_required) + "{} {}".format(python, command)',
         '        return gpu_prefix(gpu_slots_required, cpu_fallback) + "{} {}".format(python, command)',
         'execution_command调用'),
    )
    for old, new, label in replacements:
        text = replace_once(text, old, new, label)

    def transform_rule(source, name, transform):
        match = re.search(r'^rule {}:\n.*?(?=^rule |\Z)'.format(name), source, re.M | re.S)
        require(match, '缺少规则: ' + name)
        body = transform(match.group())
        return source[:match.start()] + body + source[match.end():]

    def transform_wmh(body):
        old = '        gpu=1 if PROFILE == "gpu" else 0,'
        new = '        gpu=GPU_SLOTS_PER_DEVICE if PROFILE == "gpu" else 0,'
        if old in body:
            body = replace_once(body, old, new, 'WMH调度令牌')
        else:
            require(body.count(new) == 1, 'WMH调度令牌不是已知的1槽或2槽版本')
        require(body.count('wmh_exclusive=1 if PROFILE == "gpu" else 0') == 1,
                'WMH独占资源上下文不匹配')
        require(body.count('gpu_slots_required=GPU_SLOTS_PER_DEVICE') == 1,
                'WMH物理双槽上下文不匹配')
        return body

    def transform_t1(body):
        body = replace_once(body, '        gpu=1 if PROFILE == "gpu" else 0\n',
                            '        gpu=0\n', 'T1调度令牌')
        body = replace_once(body, 'needs_gpu=True))', 'needs_gpu=True, cpu_fallback=True))',
                            'T1 CPU回退入口')
        require(body.count('threads: CPU_HEAVY_THREADS') == 1, 'T1四线程声明上下文不匹配')
        return body

    text = transform_rule(text, 'wmh_segmentation', transform_wmh)
    text = transform_rule(text, 't1', transform_t1)
    old_comment = '# 06 T1 GPU共享队列：每例占一个槽位；工作站1配置两槽，因此最多并行两例。'
    if old_comment in text:
        text = replace_once(text, old_comment,
                            '# 06 T1启动时尝试GPU0一槽；槽忙或WMH已进入准入区则立即CPU，不中途切换。',
                            'T1规则注释')
    path.write_bytes(text.replace('\n', eol).encode('utf-8'))
    return text


def main():
    require(len(sys.argv) in (2, 4) and Path(sys.argv[1]) == ROOT and ROOT.resolve() == ROOT,
            '只允许工作站1固定活动根目录')
    resume_archive = None
    if len(sys.argv) == 4:
        require(sys.argv[2] == '--resume', '续部署参数必须为--resume和归档路径')
        resume_archive = sys.argv[3]
    root, bundle = ROOT, Path(__file__).resolve().parent
    run(['sha256sum', '-c', str(bundle / 'SHA256SUMS')], bundle)
    manifest = json.loads((bundle / 'ws1_t1_hybrid_manifest.json').read_text())
    snake, pool = check_source(root)
    if text_digest(root / FILES[1]) == manifest['new_pool_sha256'] and 'cpu_fallback=True' in snake:
        t1 = re.search(r'^rule t1:\n.*?(?=^rule |\Z)', snake, re.M | re.S).group()
        wmh = re.search(r'^rule wmh_segmentation:\n.*?(?=^rule |\Z)', snake, re.M | re.S).group()
        require('gpu=0' in t1 and 'gpu=GPU_SLOTS_PER_DEVICE if PROFILE == "gpu" else 0' in wmh,
                'GPU锁已更新，但Snakefile不是已知的混合设备版本')
        print('混合设备补丁已经存在；未重复停止或重启。请检查当前PID及GPU_DISPATCH日志。')
        return
    require(text_digest(root / FILES[1]) == manifest['old_pool_sha256'], '未知GPU锁模块：停止前中止')
    known_snake = (digest(root / FILES[0]) in manifest['old_snake_sha256'] or
                   text_digest(root / FILES[0]) in manifest['old_snake_text_sha256'])
    require(known_snake,
            '未知Snakefile版本：停止前中止；实际SHA256=' + digest(root / FILES[0]))
    config_path = safe(root / 'config/config.yaml', root)
    config = yaml.safe_load(config_path.read_text())
    deriv = (root / config['derivatives']).resolve()
    require(deriv == root / 'derivatives/substain_features',
            '实际derivatives与原V1.0.9入口不一致；本补丁不迁移路径: ' + str(deriv))
    safe(deriv, root, False)
    require(deriv.is_dir(), 'derivatives不存在')
    execution = config['execution']
    require(int(execution.get('cpu_threads_per_job', 8)) == 4, '实际config每任务线程不是4')
    require(execution.get('device_cpu', 'cpu') == 'cpu' and execution.get('device_gpu', 'cuda') == 'cuda',
            'CPU/GPU设备配置与已验证入口不一致')
    require((root / execution.get('gpu_lock_dir', 'derivatives/substain_features/.gpu-locks')).resolve()
            == deriv / '.gpu-locks', 'GPU锁目录不是预期的共享锁目录')
    core = root / 'envs/core-venv/bin/python'
    for key, default in (('core_python', 'envs/core-venv/bin/python'), ('t1_python', 'envs/t1/bin/python'),
                         ('wmh_python', 'envs/wmh/bin/python')):
        require(root / execution.get(key, default) == root / default, '非预期分析环境路径: ' + key)
        require((root / default).is_file() and os.access(root / default, os.X_OK), 'Python软链接无效或不可执行')
    if resume_archive is None:
        leader = validate_leader(root)
        members = group(leader['pgid'])
        require(all(worker['pgid'] == leader['pgid'] for worker in project_workers(root)),
                '发现主分析进程组之外的本项目分析进程，停止前中止')
        require(any('snakemake' in m['args'] and 'gpu_devices=0' in m['args'] and 'rolling_window=200' in m['args']
                    and 'gpu_slots_per_device=2' in m['args'] for m in members), '未找到预期的活动滚动GPU0 Snakemake命令')
        archive = snapshot = None
    else:
        archive, snapshot = load_resume_snapshot(root, resume_archive)
        leader, members = None, []
    # Stage both known variants without touching live files or stopping analysis.
    with tempfile.TemporaryDirectory(prefix='ws1-hybrid-stage-') as temporary:
        staged = Path(temporary)
        for rel in FILES:
            (staged / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, staged / rel)
        run(['git', 'init', '-q'], staged)
        patch_snakefile(staged / FILES[0])
        pool_include = '--include={}'.format(FILES[1])
        run(['git', 'apply', '--check', '--whitespace=error', pool_include,
             str(bundle / 'ws1_t1_hybrid.patch')], staged)
        run(['git', 'apply', '--whitespace=error', pool_include,
             str(bundle / 'ws1_t1_hybrid.patch')], staged)
        require(text_digest(staged / FILES[1]) == manifest['new_pool_sha256'],
                'GPU锁模块应用后校验值不匹配')
        compile((staged / FILES[1]).read_text(), FILES[1], 'exec')
        run([core, bundle / 'verify_ws1_t1_hybrid.py', staged], root, timeout=90)
        for rel in ('scripts/start_ws1_v1_0_9.sh', 'scripts/finish_ws1_v1_0_9.sh'):
            run(['bash', '-n', root / rel], root)
        if resume_archive is None:
            require(identity(validate_leader(root)) == identity(leader), '预检查期间主进程身份发生变化')
            archive_base = root / 'archive'
            safe(archive_base, root, False)
            archive_base.mkdir(exist_ok=True)
            archive = Path(tempfile.mkdtemp(prefix='t1-hybrid-', dir=str(archive_base)))
            for rel in FILES:
                destination = archive / 'before' / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, destination)
            active = active_snapshot(group(leader['pgid']), deriv)
            stopped_at = time.time()
            snapshot = dict(leader=leader, active=active, stopped_at=stopped_at, derivatives=str(deriv),
                            source_sha256={rel: digest(root / rel) for rel in FILES})
            (archive / 'before.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
            print('上下文检查通过。根目录={}；derivatives={}；归档={}'.format(root, deriv, archive), flush=True)
            require(identity(validate_leader(root)) == identity(leader), '发送TERM前进程身份发生变化')
            os.killpg(leader['pgid'], signal.SIGTERM)
            deadline = time.monotonic() + 60
            while group(leader['pgid']) and time.monotonic() < deadline:
                time.sleep(.5)
            require(not group(leader['pgid']), 'TERM后60秒仍有活动进程；不使用KILL、未修改源码: ' + str(archive))
            require(not project_workers(root), '发现残留本项目分析进程；未修改源码、不使用KILL')
            ended_at = time.time()
        else:
            active = snapshot['active']
            stopped_at = float(snapshot['stopped_at'])
            ended_at = time.time()
            print('续部署检查通过。旧PID/PGID已退出；源码及归档校验一致；归档={}'.format(archive), flush=True)
        archive_interrupted(active, deriv, archive, stopped_at, ended_at)
        sm = [core, '-m', 'snakemake', '--snakefile', root / 'workflow/Snakefile', '--configfile', config_path]
        cfg = ['--config', 'active_config_file=' + str(config_path), 'selected_participant=all',
               'profile=gpu', 'gpu_devices=0', 'gpu_slots_per_device=2']
        modified = False
        try:
            for rel in FILES:
                safe(root / rel, root)
                require(digest(root / rel) == snapshot['source_sha256'][rel], '停止期间源码被修改，拒绝覆盖')
            modified = True
            for rel in FILES:
                shutil.copyfile(staged / rel, root / rel)
            run([core, '-m', 'py_compile', root / FILES[1]], root, archive / 'compile.txt')
            run(sm + ['--list'] + cfg, root, archive / 'snakemake-rules.txt')
            run(sm + ['--cores', '1', '--unlock'] + cfg, root, archive / 'unlock.txt')
            # Only clear incomplete metadata for this deployment's interrupted, missing outputs.
            # Existing pass outputs, historical failures and other subjects are never reset.
            missing = [value for value in active if not Path(value).exists()]
            if missing:
                (archive / 'missing-interrupted-targets.json').write_text(json.dumps(missing, indent=2))
                print('仅清理本次中断且输出缺失的Snakemake元数据: {} 个。'.format(len(missing)), flush=True)
                run(sm + ['--cleanup-metadata'] + missing + cfg, root, archive / 'cleanup-metadata.txt')
        except BaseException:
            if modified:
                for rel in FILES:
                    shutil.copyfile(archive / 'before' / rel, safe(root / rel, root))
            print('验证失败：已恢复修改的源码，分析保持停止。归档=' + str(archive), flush=True)
            raise
        # Do not restore source under a live restarted workflow on any later error.
        run(['bash', root / 'scripts/start_ws1_v1_0_9.sh'], root, archive / 'restart.txt')
        new_leader = validate_leader(root)
        log = Path(safe(root / 'logs/full_run_v1.0.9.logpath', root).read_text().strip())
        safe(log, root / 'logs')
        deadline = time.monotonic() + 30
        new_schedulers = []
        while time.monotonic() < deadline:
            require(process(new_leader['pid']), '重启主进程提前退出，检查日志: ' + str(log))
            new_schedulers = [m for m in group(new_leader['pgid']) if 'snakemake' in m['args']]
            if new_schedulers and 'gpu_slots=2' in log.read_text(errors='replace'):
                break
            time.sleep(.5)
        require('gpu_slots=2' in log.read_text(errors='replace'), '重启日志尚未确认gpu_slots=2，请检查: ' + str(log))
        require(new_schedulers and all('gpu_devices=0' in m['args'] and 'gpu_slots_per_device=2' in m['args']
                                      and 'profile=gpu' in m['args'] for m in new_schedulers),
                '尚未确认新Snakemake的GPU0双槽命令，请检查日志；不在运行中回退源码: ' + str(log))
        require(identity(validate_leader(root)) == identity(new_leader), '新主进程已退出或身份改变')
        (archive / 'deployment.json').write_text(json.dumps(dict(pid=new_leader['pid'], log=str(log),
            source_sha256={rel: digest(root / rel) for rel in FILES},
            gpu_model_acceptance='PENDING: 2 GPU T1 + 1 CPU T1 + 1 WMH and cleanup'), indent=2))
        print('部署并重启成功：PID=PGID={}；GPU0双槽，T1空槽GPU/忙则CPU；WMH独占。'.format(new_leader['pid']))
        print('日志: {}\n归档: {}\n真实病例验收尚待运行完成；窗口3无需修改。'.format(log, archive))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('部署中止: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
