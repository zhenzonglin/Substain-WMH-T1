"""本地GPU槽位调度：共享T1槽位，同时支持WMH独占整张卡。"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import IO, List, Mapping, Optional, Sequence


def detect_gpu_ids(environment: Optional[Mapping[str, str]] = None) -> List[str]:
    """使用nvidia-smi读取真实设备编号；无GPU或命令失败时返回空列表。"""

    env = dict(os.environ if environment is None else environment)
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"-1", "none", "None"}:
        return [value.strip() for value in visible.split(",") if value.strip()]
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _lock_path(lock_dir: Path, gpu_id: str, slot_index: int, slots_per_gpu: int) -> Path:
    if slots_per_gpu == 1:
        return lock_dir / "gpu-{}.lock".format(gpu_id)
    return lock_dir / "gpu-{}-slot-{}.lock".format(gpu_id, slot_index)


def _release_locks(handles: Sequence[IO[str]]) -> None:
    for handle in reversed(handles):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _try_acquire_slots(
    gpu_id: str,
    lock_dir: Path,
    slots_per_gpu: int,
    slots_required: int,
) -> List[IO[str]]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles: List[IO[str]] = []
    for slot_index in range(slots_per_gpu):
        handle = _lock_path(lock_dir, gpu_id, slot_index, slots_per_gpu).open(
            "a+", encoding="utf-8"
        )
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        handles.append(handle)
        if len(handles) == slots_required:
            return handles
    _release_locks(handles)
    return []


def run_with_gpu_lock(
    command: Sequence[str],
    gpu_ids: Sequence[str],
    lock_dir: Path,
    slots_per_gpu: int = 1,
    slots_required: int = 1,
    cpu_fallback: bool = False,
    min_free_memory_mib: int = 0,
) -> int:
    """等待任一GPU的所需槽位，持锁运行子进程并设置可见设备。"""

    if not command:
        raise ValueError("GPU包装器缺少待执行命令")
    if not gpu_ids:
        raise RuntimeError("GPU配置已启用，但没有可分配GPU")
    if slots_per_gpu < 1:
        raise ValueError("每张GPU的槽位数必须为正整数")
    if slots_required < 1 or slots_required > slots_per_gpu:
        raise ValueError("任务所需GPU槽位必须位于1到每卡槽位数之间")
    if min_free_memory_mib < 0:
        raise ValueError("GPU最低空闲显存不能为负数")
    cpu_command = None
    if cpu_fallback or min_free_memory_mib:
        cpu_command = _cpu_profile_command(command)
    if cpu_fallback:
        if _command_stage(command) != "t1":
            raise ValueError("槽位忙时CPU回退仅适用于T1")
        if list(gpu_ids) != ["0"] or slots_per_gpu != 2 or slots_required != 1:
            raise ValueError("T1 CPU回退仅支持GPU0的双槽、每例一槽")
    lock_dir.mkdir(parents=True, exist_ok=True)
    # WMH在等待旧T1释放槽位期间也持有准入锁，避免新T1抢先进入。
    # T1仅在准入和取槽这一小段持有准入锁，随后立即释放。
    while True:
        for gpu_id in gpu_ids:
            gate = (lock_dir / "gpu-{}-admission.lock".format(gpu_id)).open("a+")
            handles = []
            try:
                try:
                    fcntl.flock(gate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    gate.close()
                    if cpu_fallback:
                        return _run_selected(cpu_command, None, 0, "admission_busy")
                    continue
                while True:
                    handles = _try_acquire_slots(
                        str(gpu_id), lock_dir, slots_per_gpu, slots_required
                    )
                    if handles:
                        break
                    if cpu_fallback:
                        _release_locks([gate])
                        return _run_selected(cpu_command, None, 0, "slots_busy")
                    if slots_required != slots_per_gpu:
                        break
                    time.sleep(0.25)
                if not handles:
                    continue
                if min_free_memory_mib:
                    free_memory = _gpu_free_memory_mib(str(gpu_id))
                    if free_memory is None or free_memory < min_free_memory_mib:
                        reason = (
                            "memory_query_failed"
                            if free_memory is None
                            else "free_memory_{}_lt_{}".format(
                                free_memory, min_free_memory_mib
                            )
                        )
                        _release_locks(handles)
                        handles = []
                        _release_locks([gate])
                        return _run_selected(cpu_command, None, 0, reason)
                if slots_required != slots_per_gpu:
                    _release_locks([gate])
                return _run_selected(command, str(gpu_id), slots_required, "slot_acquired")
            finally:
                _release_locks(handles)
                if not gate.closed:
                    _release_locks([gate])
        time.sleep(0.25)


def _command_stage(command: Sequence[str]) -> str:
    args = list(command)
    marker = ["-m", "substain_features.cli", "stage"]
    for index in range(len(args) - 3):
        if args[index:index + 3] == marker:
            return args[index + 3]
    return ""


def _cpu_profile_command(command: Sequence[str]) -> List[str]:
    """仅将已知T1或WMH-SynthSeg命令从GPU profile切换为CPU。"""
    result = list(command)
    stage = _command_stage(result)
    if stage not in {"t1", "wmh-seg"}:
        raise ValueError("显存CPU回退仅适用于T1或WMH-SynthSeg")
    if result.count("--profile") != 1:
        raise ValueError("任务必须显式且仅指定一次--profile gpu")
    index = result.index("--profile") + 1
    if index >= len(result) or result[index] != "gpu":
        raise ValueError("任务必须使用--profile gpu作为首选设备")
    result[index] = "cpu"
    return result


def _gpu_free_memory_mib(gpu_id: str) -> Optional[int]:
    """查询物理GPU当前空闲显存；查询失败时返回None并安全转CPU。"""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id={}".format(gpu_id),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(values) != 1 or not values[0].isdigit():
        return None
    return int(values[0])


def _run_selected(command: Sequence[str], gpu_id: Optional[str], slots: int, reason: str) -> int:
    environment = os.environ.copy()
    # 不从父包装器继承陈旧的GPU分配标记。
    for key in list(environment):
        if key.startswith("SUBSTAIN_ASSIGNED_GPU"):
            environment.pop(key)
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu_id is None else gpu_id
    if gpu_id is not None:
        environment["SUBSTAIN_ASSIGNED_GPU"] = gpu_id
        environment["SUBSTAIN_ASSIGNED_GPU_SLOTS"] = str(slots)
    args = list(command)
    participant = args[args.index("--participant-id") + 1] if "--participant-id" in args else "NA"
    stage = args[args.index("stage") + 1] if "stage" in args else "unknown"
    print("GPU_DISPATCH stage={} participant={} device={} reason={} slots={}".format(
        stage, participant, "cpu" if gpu_id is None else "gpu" + gpu_id, reason, slots
    ), flush=True)

    # 子进程与包装器保留原进程组。只向包装器发送TERM时，也转发给
    # 子进程树并等待退出；锁不在模型仍运行时提前释放。部署仍只发组TERM。
    child = None
    received = []
    terminating = {}

    def terminate(signum, frame):
        received.append(signum)
        if child is not None:
            _terminate_tree(child.pid, signum, terminating)

    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        child = subprocess.Popen(args, env=environment)
        if received:
            _terminate_tree(child.pid, received[-1], terminating)
        returncode = child.wait()
        while any(_process_identity(pid) == identity for pid, identity in list(terminating.items())):
            time.sleep(0.1)
        return 128 + received[-1] if received else int(returncode)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _process_identity(pid: int) -> Optional[str]:
    try:
        fields = Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def _terminate_tree(pid: int, signum: int, terminating: dict) -> None:
    # Linux的children接口无需扫描其他用户进程，也不检测其他GPU。
    try:
        children = Path("/proc/{}/task/{}/children".format(pid, pid)).read_text().split()
    except OSError:
        children = []
    for value in children:
        _terminate_tree(int(value), signum, terminating)
    identity = _process_identity(pid)
    if identity is None:
        return
    terminating[pid] = identity
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="为GPU任务分配共享或独占槽位")
    parser.add_argument("--gpu-ids", required=True, help="逗号分隔的设备编号")
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--slots-required", type=int, default=1)
    parser.add_argument("--cpu-fallback", action="store_true", help="仅T1：空槽用GPU0，否则立即CPU")
    parser.add_argument("--min-free-memory-mib", type=int, default=0, help="空闲显存低于此值时使用CPU")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    raise SystemExit(
        run_with_gpu_lock(
            command,
            [item for item in args.gpu_ids.split(",") if item],
            args.lock_dir,
            slots_per_gpu=args.slots_per_gpu,
            slots_required=args.slots_required,
            cpu_fallback=args.cpu_fallback,
            min_free_memory_mib=args.min_free_memory_mib,
        )
    )


if __name__ == "__main__":
    main()
