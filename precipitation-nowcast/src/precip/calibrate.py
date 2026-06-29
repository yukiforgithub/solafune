"""Phase2 後処理較正（isotonic recalibration）。

GBDT の OOF 予測に対し衛星別の単調較正器（isotonic regression）を fit し、予測を
RMSE 最適な単調変換で補正する。82% が厳密 0・強雨で系統的過小という miscalibration を、
単調性を保ったまま補正する。

設計:
  - isotonic は単調制約下で二乗誤差最小 ⇒ 低予測域は「その範囲の条件付き平均 y」へ、
    強雨域の圧縮（過小予測）を引き伸ばす。**ゼロ丸め閾値は isotonic が条件付き平均へ
    マップするため RMSE では劣る**ので別途は設けない（isotonic が最適に包含する）。
  - 較正器は衛星別（モデルが衛星別で誤差特性が異なるため）。
  - LUT を巨大化させないため予測値を分位ビン化して重み付き isotonic を fit し、
    区分線形 lookup(x,y) として保存。予測時は sklearn 非依存で np.interp で適用する。

リーク防止:
  - 較正の効果は OOF を fold-out（fold f を除いて fit → fold f に適用）で正直に評価する。
  - 配備用の最終 LUT は当該衛星の全 OOF で fit する。
"""

from __future__ import annotations

import numpy as np

from .metrics import rmse


def fit_isotonic_lut(
    y_true: np.ndarray, pred: np.ndarray, n_bins: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """予測→較正値の単調 LUT(x,y) を fit して返す。

    予測値を分位 n_bins ビンに集約し、ビンごとの (平均予測, 平均 y, 件数) に重み付き
    isotonic regression を当てる。返り値は区分線形 lookup の x/y 閾値（昇順, 非負）。

    Args:
        y_true: 真値 (N,)。
        pred:   生予測 (N,)。
        n_bins: 分位ビン数（LUT 解像度の上限）。

    Returns:
        (x_thresholds, y_thresholds)。np.interp(pred, x, y) で適用する。
    """
    from sklearn.isotonic import IsotonicRegression

    pred = np.asarray(pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # 退化（予測がほぼ定数）時は定数マップ。
    pmin, pmax = float(pred.min()), float(pred.max())
    if not np.isfinite(pmin) or pmax - pmin < 1e-12:
        ymean = float(y_true.mean())
        return np.array([pmin, pmin + 1e-6]), np.array([ymean, ymean])

    edges = np.unique(np.quantile(pred, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        ymean = float(y_true.mean())
        return np.array([pmin, pmax + 1e-6]), np.array([ymean, ymean])

    # 内側のエッジで digitize（端ビンに外側を寄せる）。
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, edges.size - 2)
    nb = edges.size - 1
    sum_p = np.bincount(idx, weights=pred, minlength=nb)
    sum_y = np.bincount(idx, weights=y_true, minlength=nb)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    m = cnt > 0
    xp = sum_p[m] / cnt[m]
    yp = sum_y[m] / cnt[m]
    w = cnt[m]

    iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    iso.fit(xp, yp, sample_weight=w)
    xt = np.asarray(iso.X_thresholds_, dtype=np.float64)
    yt = np.asarray(iso.y_thresholds_, dtype=np.float64)
    return xt, yt


def apply_lut(pred: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LUT(x,y) を予測に適用（区分線形補間, 範囲外は端値クランプ, 非負クリップ）。"""
    out = np.interp(np.asarray(pred, dtype=np.float64), x, y)
    return np.clip(out, 0.0, None)


def _cond_rmse(y: np.ndarray, p: np.ndarray, thr: float) -> float | None:
    m = y >= thr
    return rmse(y[m], p[m]) if np.any(m) else None


def run(
    oof_by_sat: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_splits: int,
    n_bins: int = 512,
) -> dict:
    """OOF から衛星別 isotonic 較正を fold-out 評価し、最終 LUT を返す。

    Args:
        oof_by_sat: {satellite: (y, pred, fold)}。すべて同じ長さの 1D 配列。
        n_splits: fold 数（fold-out 評価に使用）。
        n_bins: LUT の分位ビン数。

    Returns:
        {"before": {...}, "after": {...}, "luts": {sat: {"x":[...], "y":[...]}}}。
        before/after は overall / per_sat / cond_ge1 / cond_ge5 の RMSE。
    """
    ys, ps, cs = [], [], []
    per_sat_before: dict[str, float] = {}
    per_sat_after: dict[str, float] = {}
    luts: dict[str, dict[str, list[float]]] = {}

    for sat, (y, p, fold) in oof_by_sat.items():
        y = np.asarray(y, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        fold = np.asarray(fold)

        # fold-out 較正（リーク防止）: fold f を除いて fit → fold f に適用。
        cal = p.copy()
        for f in range(n_splits):
            te = fold == f
            tr = fold != f
            if te.sum() == 0 or tr.sum() == 0:
                continue
            xt, yt = fit_isotonic_lut(y[tr], p[tr], n_bins)
            cal[te] = apply_lut(p[te], xt, yt)

        per_sat_before[sat] = rmse(y, p)
        per_sat_after[sat] = rmse(y, cal)
        ys.append(y)
        ps.append(p)
        cs.append(cal)

        # 配備用の最終 LUT（当該衛星の全 OOF で fit）。
        xt, yt = fit_isotonic_lut(y, p, n_bins)
        luts[sat] = {"x": xt.tolist(), "y": yt.tolist()}

    Y = np.concatenate(ys)
    P = np.concatenate(ps)
    C = np.concatenate(cs)
    return {
        "before": {
            "overall": rmse(Y, P),
            "per_sat": per_sat_before,
            "cond_ge1": _cond_rmse(Y, P, 1.0),
            "cond_ge5": _cond_rmse(Y, P, 5.0),
        },
        "after": {
            "overall": rmse(Y, C),
            "per_sat": per_sat_after,
            "cond_ge1": _cond_rmse(Y, C, 1.0),
            "cond_ge5": _cond_rmse(Y, C, 5.0),
        },
        "luts": luts,
    }
