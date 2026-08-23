"""阶段状态文件：让失败节点可追溯，同时允许保留另一模态。"""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional


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


def guarded_stage(path: Path, stage: str, participant_id: str, function: Callable[[], Dict[str, object]]) -> bool:
    """所有阶段都物化状态文件；失败不会伪造特征。"""

    try:
        details = function()
        write_status(path, stage, "pass", participant_id, details)
        return True
    except Exception as exc:
        write_status(
            path,
            stage,
            "fail",
            participant_id,
            {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()},
        )
        return False
