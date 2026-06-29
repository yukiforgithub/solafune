"""Phase2 アンサンブル（two_part × tweedie ブレンド）。

2 つの variant（hurdle=two_part と tweedie）の生予測を凸結合する。目的関数が異なり
誤差が脱相関するため、単一最良 variant より RMSE を下げうる。重みは衛星ごとに 1 スカラ
（極低容量）で、後処理較正と違い**地域転移しやすい**。ただし採用可否は必ず地域
GroupKFold の fold-out OOF で検証する（in-sample 改善は当てにしない）。

凸結合 pred = w·p_tweedie + (1-w)·p_two_part, w∈[0,1]。
"""

from __future__ import annotations

import numpy as np

from .metrics import rmse


def optimal_blend_weight(p_primary: np.ndarray, p_other: np.ndarray, y: np.ndarray) -> float:
    """min_w Σ(w·p_primary + (1-w)·p_other − y)² の閉形式解を [0,1] にクリップして返す。

    d=p_primary−p_other, r=y−p_other とおくと w*=Σ(d·r)/Σ(d²)（凸2次）。
    分母が極小（2 予測がほぼ同一）なら 1.0（primary 採用）にフォールバック。
    """
    d = np.asarray(p_primary, np.float64) - np.asarray(p_other, np.float64)
    r = np.asarray(y, np.float64) - np.asarray(p_other, np.float64)
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        return 1.0
    w = float(np.dot(d, r) / denom)
    return min(1.0, max(0.0, w))


def blend(p_primary: np.ndarray, p_other: np.ndarray, w: float) -> np.ndarray:
    """凸結合 w·p_primary + (1-w)·p_other（非負クリップ）。"""
    out = w * np.asarray(p_primary, np.float64) + (1.0 - w) * np.asarray(p_other, np.float64)
    return np.clip(out, 0.0, None)


def _cond_rmse(y: np.ndarray, p: np.ndarray, thr: float) -> float | None:
    m = y >= thr
    return rmse(y[m], p[m]) if np.any(m) else None


def evaluate(
    oof_primary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    oof_other: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_splits: int,
) -> dict:
    """衛星別ブレンドを fold-out 評価し、最終重みと before/after 指標を返す。

    Args:
        oof_primary: {sat: (y, pred_primary, fold)}（tweedie 等の主予測）。
        oof_other:   {sat: (y, pred_other,   fold)}（two_part 等）。y/fold は primary と一致前提。
        n_splits: fold 数。

    Returns:
        {"weights": {sat: w}, "primary": {...}, "other": {...}, "blend": {...}}。
        各指標 dict は overall / per_sat / cond_ge1 / cond_ge5 の RMSE。
    """
    ys, pa, pb, pc = [], [], [], []
    per_sat_primary: dict[str, float] = {}
    per_sat_other: dict[str, float] = {}
    per_sat_blend: dict[str, float] = {}
    weights: dict[str, float] = {}

    for sat in oof_primary:
        if sat not in oof_other:
            continue
        y, a, fold = oof_primary[sat]
        _, b, _ = oof_other[sat]
        y = np.asarray(y, np.float64)
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        fold = np.asarray(fold)

        # fold-out ブレンド（学習 fold で重みを決め、held-out fold に適用）。
        cal = a.copy()
        for f in range(n_splits):
            te = fold == f
            tr = fold != f
            if te.sum() == 0 or tr.sum() == 0:
                continue
            w_f = optimal_blend_weight(a[tr], b[tr], y[tr])
            cal[te] = blend(a[te], b[te], w_f)

        per_sat_primary[sat] = rmse(y, a)
        per_sat_other[sat] = rmse(y, b)
        per_sat_blend[sat] = rmse(y, cal)
        weights[sat] = optimal_blend_weight(a, b, y)  # 配備用（全 OOF）
        ys.append(y)
        pa.append(a)
        pb.append(b)
        pc.append(cal)

    Y = np.concatenate(ys)
    A = np.concatenate(pa)
    B = np.concatenate(pb)
    C = np.concatenate(pc)

    def _pack(per_sat, P):
        return {
            "overall": rmse(Y, P),
            "per_sat": per_sat,
            "cond_ge1": _cond_rmse(Y, P, 1.0),
            "cond_ge5": _cond_rmse(Y, P, 5.0),
        }

    return {
        "weights": weights,
        "primary": _pack(per_sat_primary, A),
        "other": _pack(per_sat_other, B),
        "blend": _pack(per_sat_blend, C),
    }
