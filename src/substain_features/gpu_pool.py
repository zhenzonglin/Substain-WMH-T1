"""本地多GPU独占调度：每个GPU阶段一次只占用一张卡。"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows只用于源码检查，正式计算要求Linux。
    fcntl = None  # type: ignore[assignment]


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


def run_with_gpu_lock(command: Sequence[str], gpu_ids: Sequence[str], lock_dir: Path) -> int:
    """等待任一GPU锁，持锁运行子进程并设置独立CUDA_VISIBLE_DEVICES。"""

    if fcntl is None:
        raise RuntimeError("GPU文件锁需要Linux/Unix fcntl")
    if not command:
        raise ValueError("GPU包装器缺少待执行命令")
    if not gpu_ids:
        raise RuntimeError("GPU配置已启用，但没有可分配GPU")
    lock_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for gpu_id in gpu_ids:
            lock_path = lock_dir / "gpu-{}.lock".format(gpu_id)
            handle = lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                environment["SUBSTAIN_ASSIGNED_GPU"] = str(gpu_id)
                completed = subprocess.run(list(command), env=environment, check=False)
                return int(completed.returncode)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="为一个GPU任务分配独占设备")
    parser.add_argument("--gpu-ids", required=True, help="逗号分隔的设备编号")
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    raise SystemExit(run_with_gpu_lock(command, [item for item in args.gpu_ids.split(",") if item], args.lock_dir))


if __name__ == "__main__":
    main()
