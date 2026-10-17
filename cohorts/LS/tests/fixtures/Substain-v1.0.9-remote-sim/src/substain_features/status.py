"""阶段状态文件：让失败节点可追溯，同时允许保留另一模态。"""

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

try:
    import resource
except ImportError:  # pragma: no cover - resource在Windows不可用
    resource = None  # type: ignore[assignment]


def write_status(path: Path, stage: str, status: str, participant_id: str, details: Dict[str, object]) -> None:
    payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "participant_id": participant_id,
        "stage": stage,
        "status": status,
        "details": details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    # 最新状态便于Snakemake判定；追加历史保留每一次失败与重跑证据。
    history_path = path.parent / "status_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _peak_rss_mb(who: int) -> Optional[float]:
    """读取Unix进程峰值RSS；不支持的平台返回None。"""

    if resource is None:
        return None
    usage = resource.getrusage(who)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(float(usage.ru_maxrss) / divisor, 3)


def _runtime_details(started_at_utc: str, started_monotonic: float) -> Dict[str, object]:
    finished_at_utc = datetime.now(timezone.utc).isoformat()
    return {
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "duration_seconds": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "peak_rss_mb_self": _peak_rss_mb(resource.RUSAGE_SELF) if resource is not None else None,
        "peak_rss_mb_children": _peak_rss_mb(resource.RUSAGE_CHILDREN) if resource is not None else None,
    }


def guarded_stage(path: Path, stage: str, participant_id: str, function: Callable[[], Dict[str, object]]) -> bool:
    """所有阶段都物化状态文件；失败不会伪造特征。"""

    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    try:
        details = dict(function())
        details["runtime"] = _runtime_details(started_at_utc, started_monotonic)
        write_status(path, stage, "pass", participant_id, details)
        return True
    except Exception as exc:
        runtime = _runtime_details(started_at_utc, started_monotonic)
        write_status(
            path,
            stage,
            "fail",
            participant_id,
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime": runtime,
            },
        )
        return False
