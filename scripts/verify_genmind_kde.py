#!/usr/bin/env python3
"""离线加载并抽样全部 GenMIND KDE；不把生成样本用于当前两例常模。"""

from pathlib import Path
import fnmatch
import warnings

import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "resources" / "normative" / "genmind_upstream" / "model"


def main() -> None:
    # sklearn 持久化对象不保证跨版本兼容；上游六模型混合记录1.2.1/1.2.2，
    # 固定1.2.2并只忽略该已知的补丁版本告警，其他反序列化错误仍会失败。
    if sklearn.__version__ != "1.2.2":
        raise RuntimeError("GenMIND KDE 要求 scikit-learn==1.2.2，实际 {}".format(sklearn.__version__))
    kde_files = sorted(MODEL_ROOT.glob("kde_*.npz"))
    if len(kde_files) != 6:
        raise FileNotFoundError("应有6个GenMIND KDE，实际 {}".format(len(kde_files)))
    column_dictionary = np.load(MODEL_ROOT / "col_dict.npz", allow_pickle=True)["dict"].item()
    if len(column_dictionary) < 145:
        raise ValueError("GenMIND列字典不完整")
    for path in kde_files:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator", category=UserWarning)
            payload = np.load(path, allow_pickle=True)["model"].item()
        generated = payload["scaler"].inverse_transform(payload["model"].sample(1, random_state=0))
        roi_columns = fnmatch.filter(list(payload["columns"]), "H_*")
        if generated.shape != (1, 146) or len(roi_columns) != 145:
            raise ValueError("{} 维数不符合官方生成器契约".format(path.name))
    print("genmind_kde_loaded=6 sklearn={} generated_columns=146".format(sklearn.__version__))


if __name__ == "__main__":
    main()
