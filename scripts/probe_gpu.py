#!/usr/bin/env python
"""运行一个真实 CUDA 张量算子，避免只凭 torch.cuda.is_available() 误判。"""

import json

import torch


def main() -> None:
    ok = False
    error = ""
    device = ""
    try:
        device = torch.cuda.get_device_name(0)
        value = torch.ones(1, device="cuda")
        ok = bool((value + value).item() == 2.0)
    except Exception as exc:  # GPU/驱动不兼容时必须保留原始错误。
        error = str(exc)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_kernel_ok": ok,
                "device": device,
                "error": error,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
