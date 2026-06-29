"""CV 分割: 地域 group / 衛星バランスの GroupKFold と比較用 random KFold。

本コンペの本質は「未知地域への汎化」（TRAIN 20地域 と EVAL 18地域 は DISJOINT）。
したがって既定は地域（name_location）を group とする分割で、地域が train/val に
跨らないようにする。さらに各 fold で衛星構成が偏らないよう、衛星ごとに地域を
ラウンドロビン配分して fold を決める。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config

_FOLD_COL = "fold"
_LOC_COL = "name_location"
_SAT_COL = "satellite_target"


def make_location_group_kfold(
    df: pd.DataFrame,
    n_splits: int = config.DEFAULT_N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    *,
    fold_col: str = _FOLD_COL,
) -> pd.DataFrame:
    """地域 group・衛星バランスの GroupKFold fold を付与する。

    同一地域の全行は同一 fold に入る（地域リーク防止）。衛星ごとに地域を
    シャッフルしてラウンドロビンで fold に割り当てるため、各 fold の衛星構成が
    そろいやすい。

    Args:
        df: ``name_location`` と ``satellite_target`` 列を持つ DataFrame。
        n_splits: fold 数。
        seed: 地域シャッフルの乱数シード。
        fold_col: 付与する fold 列名。

    Returns:
        ``fold_col`` 列（0..n_splits-1）を加えた df のコピー。
    """
    if n_splits < 2:
        raise ValueError(f"n_splits は 2 以上が必要です（受領: {n_splits}）。")
    out = df.copy()
    rng = np.random.default_rng(seed)

    # 地域 → 衛星（各地域は単一衛星に対応する前提）。
    loc_to_sat = out.groupby(_LOC_COL)[_SAT_COL].first()

    loc_to_fold: dict[str, int] = {}
    # 衛星ごとに地域をシャッフルし、ラウンドロビンで fold を割り当てる。
    for sat in sorted(loc_to_sat.unique()):
        locs = loc_to_sat[loc_to_sat == sat].index.to_numpy()
        rng.shuffle(locs)
        for i, loc in enumerate(locs):
            loc_to_fold[loc] = i % n_splits

    out[fold_col] = out[_LOC_COL].map(loc_to_fold).astype("int64")
    return out


def load_handdesigned_folds(path: "str | None" = None) -> dict[str, int]:
    """conf/folds.yaml の手設計 name_location→fold マップを読む（seed 非依存）。

    Phase1 の正準 CV 分割。`make_location_group_kfold` の seed 依存ラウンドロビンに
    対し、こちらは §30 の手設計マップで完全再現。

    Args:
        path: folds.yaml のパス。None なら REPO_ROOT/conf/folds.yaml。

    Returns:
        name_location → fold(int) の dict。
    """
    import yaml

    from pathlib import Path

    cfg_path = Path(path) if path is not None else config.REPO_ROOT / "conf" / "folds.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("location_to_fold", {})
    return {str(k): int(v) for k, v in raw.items()}


def random_kfold(
    df: pd.DataFrame,
    n_splits: int = config.DEFAULT_N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    *,
    fold_col: str = _FOLD_COL,
) -> pd.DataFrame:
    """行単位のランダム KFold fold を付与する（比較用・楽観的）。

    地域リークを許す行レベル分割。地域 GroupKFold との CV 差を測る基準であり、
    本番の汎化評価には使わない。

    Args:
        df: 任意の DataFrame。
        n_splits: fold 数。
        seed: シャッフルのシード。
        fold_col: 付与する fold 列名。

    Returns:
        ``fold_col`` 列を加えた df のコピー。
    """
    if n_splits < 2:
        raise ValueError(f"n_splits は 2 以上が必要です（受領: {n_splits}）。")
    out = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.empty(len(out), dtype="int64")
    for f, (_, val_idx) in enumerate(kf.split(out)):
        fold[val_idx] = f
    out[fold_col] = fold
    return out
