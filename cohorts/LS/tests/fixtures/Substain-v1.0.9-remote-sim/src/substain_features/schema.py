"""参与者清单与配置的数据契约。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml


PARTICIPANT_COLUMNS = [
    "participant_id",
    "age",
    "sex",
    "site_id",
    "t1w",
    "flair",
    "lesion_mask",
]
ALLOWED_SEX = {"female", "male"}


@dataclass(frozen=True)
class Participant:
    """单个受试者的稳定输入契约；路径在加载时已转为绝对路径。"""

    participant_id: str
    age: float
    sex: str
    site_id: str
    t1w: Path
    flair: Path
    lesion_mask: Path

    @property
    def bids_id(self) -> str:
        return "sub-{}".format(self.participant_id)


def load_config(path: Path) -> Dict[str, object]:
    """加载 YAML，并把 project_root 规范化为绝对路径。"""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root_value = Path(str(config["project_root"]))
    root = root_value.resolve() if root_value.is_absolute() else (path.resolve().parent / root_value).resolve()
    config["project_root"] = str(root)
    return config


def _absolute(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def load_participants(path: Path, project_root: Path) -> List[Participant]:
    """读取清单并严格拒绝 0/1 等含糊性别编码。"""

    table = pd.read_csv(path, sep="\t", dtype=str)
    missing = [name for name in PARTICIPANT_COLUMNS if name not in table.columns]
    extra = [name for name in table.columns if name not in PARTICIPANT_COLUMNS]
    if missing or extra:
        raise ValueError("participants.tsv 字段不符合契约；missing={} extra={}".format(missing, extra))
    if table[PARTICIPANT_COLUMNS].isna().any().any():
        raise ValueError("participants.tsv 不允许空字段")

    participants = []
    seen = set()
    for row in table.to_dict(orient="records"):
        sex = str(row["sex"]).strip().lower()
        if sex not in ALLOWED_SEX:
            raise ValueError("sex 只允许 female/male，收到 {!r}".format(row["sex"]))
        participant_id = str(row["participant_id"]).strip()
        if participant_id in seen:
            raise ValueError("participant_id 重复：{}".format(participant_id))
        seen.add(participant_id)
        participants.append(
            Participant(
                participant_id=participant_id,
                age=float(row["age"]),
                sex=sex,
                site_id=str(row["site_id"]),
                t1w=_absolute(project_root, row["t1w"]),
                flair=_absolute(project_root, row["flair"]),
                lesion_mask=_absolute(project_root, row["lesion_mask"]),
            )
        )
    return participants


def select_participants(participants: Iterable[Participant], participant_id: Optional[str]) -> List[Participant]:
    """选择单例或全部；未找到时直接失败，禁止静默缩小样本。"""

    values = list(participants)
    if participant_id in (None, "all"):
        return values
    selected = [item for item in values if item.participant_id == participant_id]
    if not selected:
        raise ValueError("participants.tsv 中未找到 {}".format(participant_id))
    return selected
