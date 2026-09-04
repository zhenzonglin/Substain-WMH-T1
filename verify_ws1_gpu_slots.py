#!/usr/bin/env python3
"""Verify two shared GPU0 slots and the exclusive two-slot WMH lock."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("用法: verify_ws1_gpu_slots.py PROJECT_ROOT")
    project_root = Path(sys.argv[1]).resolve()
    source = project_root / "src"
    if not source.is_dir() or source.is_symlink():
        raise SystemExit("项目src不存在或是软链接: {}".format(source))
    sys.path.insert(0, str(source))

    from substain_features.gpu_pool import (  # pylint: disable=import-outside-toplevel
        _release_locks,
        _try_acquire_slots,
        run_with_gpu_lock,
    )

    with tempfile.TemporaryDirectory(prefix="substain-gpu-slots-") as temporary:
        test_root = Path(temporary)
        lock_dir = test_root / "locks"

        first = _try_acquire_slots("0", lock_dir, 2, 1)
        second = _try_acquire_slots("0", lock_dir, 2, 1)
        try:
            if len(first) != 1 or len(second) != 1:
                raise RuntimeError("两个T1未能分别获得一个GPU0槽位")
            if _try_acquire_slots("0", lock_dir, 2, 1):
                raise RuntimeError("第三个T1错误获得了GPU0槽位")
            if _try_acquire_slots("0", lock_dir, 2, 2):
                raise RuntimeError("T1运行时WMH错误获得了两个槽位")
        finally:
            _release_locks(second)
            _release_locks(first)

        exclusive = _try_acquire_slots("0", lock_dir, 2, 2)
        try:
            if len(exclusive) != 2:
                raise RuntimeError("WMH未能独占GPU0的两个槽位")
            if _try_acquire_slots("0", lock_dir, 2, 1):
                raise RuntimeError("WMH运行时T1错误获得了GPU0槽位")
        finally:
            _release_locks(exclusive)

        output = test_root / "assigned.txt"
        command = [
            sys.executable,
            "-c",
            (
                "import os,pathlib; "
                "pathlib.Path({!r}).write_text(os.environ['CUDA_VISIBLE_DEVICES'] + ':' + "
                "os.environ['SUBSTAIN_ASSIGNED_GPU_SLOTS'])"
            ).format(str(output)),
        ]
        if run_with_gpu_lock(command, ["0"], lock_dir, 2, 1) != 0:
            raise RuntimeError("GPU包装器子进程失败")
        if output.read_text(encoding="utf-8") != "0:1":
            raise RuntimeError("GPU包装器没有固定到物理GPU0的一个槽位")

    print("GPU0双T1共享槽位与WMH独占锁: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
