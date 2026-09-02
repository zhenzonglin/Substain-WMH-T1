"""T1 常模提供者接口与 GenMIND 临时技术常模。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .t1 import display_name_map, genmind_native145_labels


@dataclass(frozen=True)
class NormativeResult:
    profile: str
    zscores: Dict[str, float]
    reference_n: int
    age_half_window: float
    sex: str
    eligible: bool
    failure_reason: str = ""


class NormativeProvider(ABC):
    """未来中国人群常模必须实现的最小接口。"""

    @abstractmethod
    def transform(self, macro_volumes_ml: Mapping[str, float], denominator_ml: float, age: float, sex: str) -> NormativeResult:
        raise NotImplementedError


def _weighted_mean_sd(values: np.ndarray, races: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """让每个种族层贡献相同总权重，再计算总体均值和总体 SD。"""

    race_values = sorted(set(str(value) for value in races))
    if not race_values:
        raise ValueError("参考样本没有 Race")
    weights = np.zeros(len(races), dtype=float)
    for race in race_values:
        mask = races.astype(str) == race
        weights[mask] = 1.0 / (len(race_values) * int(mask.sum()))
    mean = np.sum(values * weights[:, None], axis=0)
    variance = np.sum(((values - mean) ** 2) * weights[:, None], axis=0)
    return mean, np.sqrt(variance)


class GenMINDGlobalV1Provider(NormativeProvider):
    """GenMIND 18,000 例合成数据的临时技术常模，不宣称中国人群适用。"""

    profile = "genmind_global_v1_provisional"

    def __init__(
        self,
        genmind_csv: Path,
        derived_mapping_path: Path,
        macro_mapping_path: Path,
        min_reference_n: int = 300,
        narrow_window: float = 2.5,
        expanded_window: float = 5.0,
    ) -> None:
        self.genmind_csv = genmind_csv
        self.derived_mapping_path = derived_mapping_path
        self.macro_mapping = pd.read_csv(macro_mapping_path, sep="\t")
        self.min_reference_n = int(min_reference_n)
        self.narrow_window = float(narrow_window)
        self.expanded_window = float(expanded_window)
        self._reference: Optional[pd.DataFrame] = None

    def _build_reference(self) -> pd.DataFrame:
        table = pd.read_csv(self.genmind_csv)
        labels = genmind_native145_labels(self.genmind_csv, self.derived_mapping_path)
        names = display_name_map(self.derived_mapping_path)
        label_columns = {label: names[label] for label in labels}
        macro_ids = self.macro_mapping.sort_values("macro_index")["macro_id"].drop_duplicates().tolist()
        volume_matrix = table[[label_columns[label] for label in labels]].to_numpy(dtype=float)
        denominator_mask = np.asarray([label not in {4, 11, 49, 50, 51, 52} for label in labels])
        denominator = volume_matrix[:, denominator_mask].sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("GenMIND 存在非正非脑室组织总体积")
        reference = table[["PTID", "Sex", "Race", "Age"]].copy()
        for macro_id in macro_ids:
            macro_labels = self.macro_mapping.loc[self.macro_mapping["macro_id"] == macro_id, "native_label"].astype(int).tolist()
            indices = [labels.index(label) for label in macro_labels]
            macro_volume = volume_matrix[:, indices].sum(axis=1)
            if np.any(macro_volume <= 0):
                raise ValueError("GenMIND {} 存在非正宏区体积".format(macro_id))
            reference[str(macro_id)] = np.log(macro_volume / denominator)
        reference["sex_normalized"] = reference["Sex"].map({"F": "female", "M": "male", "Female": "female", "Male": "male"})
        if reference["sex_normalized"].isna().any():
            raise ValueError("GenMIND 包含未知性别编码")
        return reference

    @property
    def reference(self) -> pd.DataFrame:
        if self._reference is None:
            self._reference = self._build_reference()
        return self._reference

    def transform(self, macro_volumes_ml: Mapping[str, float], denominator_ml: float, age: float, sex: str) -> NormativeResult:
        if sex not in {"female", "male"}:
            return NormativeResult(self.profile, {}, 0, 0.0, sex, False, "sex_missing_or_invalid")
        if age < 22 or age > 90:
            return NormativeResult(self.profile, {}, 0, 0.0, sex, False, "age_outside_22_90")
        if denominator_ml <= 0:
            return NormativeResult(self.profile, {}, 0, 0.0, sex, False, "nonpositive_denominator")

        subset = self.reference[self.reference["sex_normalized"] == sex]
        chosen = pd.DataFrame()
        used_window = self.narrow_window
        for window in (self.narrow_window, self.expanded_window):
            candidate = subset[(subset["Age"].astype(float) >= age - window) & (subset["Age"].astype(float) <= age + window)]
            chosen = candidate
            used_window = window
            if len(candidate) >= self.min_reference_n:
                break
        if len(chosen) < self.min_reference_n:
            return NormativeResult(self.profile, {}, len(chosen), used_window, sex, False, "reference_n_below_{}".format(self.min_reference_n))

        macro_ids = self.macro_mapping.sort_values("macro_index")["macro_id"].drop_duplicates().astype(str).tolist()
        missing = [macro_id for macro_id in macro_ids if macro_id not in macro_volumes_ml]
        if missing:
            return NormativeResult(self.profile, {}, len(chosen), used_window, sex, False, "missing_macro_volumes:{}".format(",".join(missing)))
        subject = np.asarray([np.log(float(macro_volumes_ml[name]) / denominator_ml) for name in macro_ids])
        values = chosen[macro_ids].to_numpy(dtype=float)
        mean, sd = _weighted_mean_sd(values, chosen["Race"].astype(str).to_numpy())
        if np.any(sd <= 0):
            return NormativeResult(self.profile, {}, len(chosen), used_window, sex, False, "nonpositive_reference_sd")
        # 疾病方向：参考均值减个体值，因此体积下降时 z 单调增大。
        zscores = (mean - subject) / sd
        return NormativeResult(
            self.profile,
            {"t1_{}_atrophy_z".format(name): float(value) for name, value in zip(macro_ids, zscores)},
            len(chosen),
            used_window,
            sex,
            True,
        )
