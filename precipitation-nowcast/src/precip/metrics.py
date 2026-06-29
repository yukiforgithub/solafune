"""評価指標。本コンペの公式指標は画素レベル RMSE（小さいほど良い）。"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE = sqrt(mean((y_true - y_pred)^2)) を返す。

    Args:
        y_true: 真値。任意形状（フラット化して評価する）。
        y_pred: 予測値。``y_true`` とブロードキャスト可能な形状。

    Returns:
        画素レベル RMSE（float）。

    Raises:
        ValueError: 要素数が 0 の場合。
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    diff = yt - yp
    if diff.size == 0:
        raise ValueError("rmse: 入力が空です。")
    return float(np.sqrt(np.mean(diff * diff)))
