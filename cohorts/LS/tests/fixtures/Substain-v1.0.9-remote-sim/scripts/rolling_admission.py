#!/usr/bin/env python3
"""Keep a bounded rolling window of admitted pipeline participants."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


VALID_PARTICIPANT = re.compile(r"^[A-Za-z0-9._-]+$")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能为负数")
    return parsed


def load_order(path: Path) -> list[str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("队列文件不存在或是软链接: {}".format(path))
    participants = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not participants:
        raise ValueError("滚动队列为空: {}".format(path))
    invalid = [value for value in participants if not VALID_PARTICIPANT.fullmatch(value)]
    if invalid:
        raise ValueError("队列含非法participant_id: {}".format(invalid[0]))
    if len(set(participants)) != len(participants):
        raise ValueError("滚动队列含重复participant_id")
    return participants


def cleanup_finished(derivatives: Path, participant_id: str) -> bool:
    path = derivatives / "sub-{}".format(participant_id) / "status" / "cleanup.json"
    if not path.is_file() or path.is_symlink():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") in {"pass", "fail"}


def write_token(output: Path, participant_id: str, index: int, window: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or output.parent.is_symlink():
        raise ValueError("拒绝向软链接写入放行令牌: {}".format(output))
    payload = {
        "participant_id": participant_id,
        "queue_index": index,
        "rolling_window": window,
        "admitted_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output.with_name(".{}.{}.tmp".format(output.name, os.getpid()))
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def initialize(order_file: Path, token_dir: Path, window: int) -> int:
    participants = load_order(order_file)
    if token_dir.exists() and token_dir.is_symlink():
        raise ValueError("令牌目录不能是软链接: {}".format(token_dir))
    token_dir.mkdir(parents=True, exist_ok=True)
    admitted = min(window, len(participants))
    for index, participant_id in enumerate(participants[:admitted]):
        write_token(token_dir / "{}.json".format(participant_id), participant_id, index, window)
    print("滚动队列初始化: total={}, window={}, admitted={}".format(len(participants), window, admitted))
    return 0


def wait_for_slot(
    order_file: Path,
    derivatives: Path,
    participant_id: str,
    output: Path,
    window: int,
    poll_seconds: float,
) -> int:
    participants = load_order(order_file)
    if participant_id not in participants:
        raise ValueError("participant_id不在滚动队列中: {}".format(participant_id))
    index = participants.index(participant_id)
    required_finished = max(0, index - window + 1)
    last_reported = None
    while True:
        finished = sum(cleanup_finished(derivatives, value) for value in participants[:index])
        if finished >= required_finished:
            write_token(output, participant_id, index, window)
            print(
                "放行participant={} index={} finished_prior={} window={}".format(
                    participant_id, index, finished, window
                )
            )
            return 0
        state = (finished, required_finished)
        if state != last_reported:
            print(
                "等待滚动名额: participant={} finished_prior={}/{} window={}".format(
                    participant_id, finished, required_finished, window
                ),
                flush=True,
            )
            last_reported = state
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("initialize", help="放行滚动窗口中的首批病例")
    init_parser.add_argument("--order-file", type=Path, required=True)
    init_parser.add_argument("--token-dir", type=Path, required=True)
    init_parser.add_argument("--window", type=_positive_int, default=200)

    wait_parser = subparsers.add_parser("wait", help="等待完成数达到阈值后放行一个病例")
    wait_parser.add_argument("--order-file", type=Path, required=True)
    wait_parser.add_argument("--derivatives", type=Path, required=True)
    wait_parser.add_argument("--participant-id", required=True)
    wait_parser.add_argument("--output", type=Path, required=True)
    wait_parser.add_argument("--window", type=_positive_int, default=200)
    wait_parser.add_argument("--poll-seconds", type=_nonnegative_float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "initialize":
            return initialize(args.order_file.resolve(), args.token_dir.resolve(), args.window)
        return wait_for_slot(
            args.order_file.resolve(),
            args.derivatives.resolve(),
            args.participant_id,
            args.output.resolve(),
            args.window,
            args.poll_seconds,
        )
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
