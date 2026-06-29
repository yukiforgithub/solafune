"""Phase1 物理 Tb→RR モデルの fit と地域 GroupKFold CV（閉形式）。

充足統計（sufficient statistics）キャッシュ
``eda_cache/phase1_window_hist.parquet`` / ``phase1_splitdiff_hist.parquet``
のみを用いて、IR 窓 Tb（DN）単独・split-window 差単独の Tb→RR 写像を
fit し、地域 GroupKFold CV の画素 RMSE を生データ再読込なしで算出する。

閉形式 RMSE
-----------
各 (satellite, name_location, bin) で count=n, sum_y=Σy, sum_y2=Σy² を持つ。
任意の写像 f(bin)→ŷ に対し、その bin に属する画素群の二乗誤差和は

    Σ(y - ŷ)² = Σy² - 2 ŷ Σy + ŷ² n = sum_y2 - 2 f·sum_y + f²·count

なので、val 集合全体で総和して画素数で割れば MSE、その平方根が RMSE。
→ 生画素を持たずとも bin ヒストグラムだけで CV RMSE が厳密に出る。

候補モデル
----------
  - window_lookup   : f(DN)=学習 fold の bin 条件付き平均 E[y|DN]。
                      スパース bin は isotonic 回帰（単調減少）で平滑化。
  - window_powerlaw : RR = a·((255-DN)/255)^b（画素 MSE 重み付き最小二乗）。
  - window_exp      : RR = a·exp(-b·DN) + c（同上）。
  - splitdiff_lookup: f(d)=E[y|splitdiff]、isotonic は使わず単純平均＋ガード。

scope は {per_satellite（衛星別 fit）, unified（全衛星統一 fit）} の2種。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from sklearn.isotonic import IsotonicRegression

# 充足統計から導いた全体平均（fold 外フォールバック・gain 基準）。
# 注: target_stats ベースの honest baseline は 1.4048。キャッシュは
# 0フレーム/IO スキップ行を含まないため僅かに異なる（global-mean ≈1.4061）。
GLOBAL_MEAN_FALLBACK: float = 0.2886


# --- 充足統計ユーティリティ -------------------------------------------------


def _weighted_mean(sum_y: np.ndarray, count: np.ndarray) -> np.ndarray:
    """bin ごとの条件付き平均 E[y|bin]=sum_y/count（count=0 は NaN）。"""
    out = np.full(sum_y.shape, np.nan, dtype=np.float64)
    nz = count > 0
    out[nz] = sum_y[nz] / count[nz]
    return out


def _sse_for_map(
    sum_y: np.ndarray, sum_y2: np.ndarray, count: np.ndarray, f: np.ndarray
) -> float:
    """写像 f を bin 配列で評価したときの二乗誤差和 Σ(y-f)²（閉形式）。"""
    return float(np.sum(sum_y2 - 2.0 * f * sum_y + (f * f) * count))


# --- lookup（条件付き平均 + isotonic 平滑化） -------------------------------


def fit_lookup_window(
    bins: np.ndarray,
    sum_y: np.ndarray,
    count: np.ndarray,
    *,
    n_bins: int = 256,
    isotonic: bool = True,
    fallback: float = GLOBAL_MEAN_FALLBACK,
) -> np.ndarray:
    """窓 DN の条件付き平均 lookup を 0..n_bins-1 の密配列で返す。

    Args:
        bins: 観測された DN 値（int, 0..255）。
        sum_y: 各 bin の Σy。
        count: 各 bin の画素数。
        n_bins: lookup 長（窓 DN は 256）。
        isotonic: True なら DN に対し単調減少を課す isotonic 回帰で平滑化。
            物理（低DN=冷たい雲頂=強雨）と整合し、スパース bin のノイズを抑える。
        fallback: 学習データに 1 画素も無かった DN への既定値。

    Returns:
        形状 (n_bins,) の float64 lookup。index=DN。
    """
    dense_n = np.zeros(n_bins, dtype=np.float64)
    dense_sy = np.zeros(n_bins, dtype=np.float64)
    b = bins.astype(np.int64)
    np.add.at(dense_n, b, count)
    np.add.at(dense_sy, b, sum_y)

    table = np.full(n_bins, fallback, dtype=np.float64)
    nz = dense_n > 0
    table[nz] = dense_sy[nz] / dense_n[nz]

    if isotonic and nz.sum() >= 2:
        x = np.nonzero(nz)[0].astype(np.float64)
        y = table[nz]
        w = dense_n[nz]
        # DN 増加に対し RR 単調減少（increasing=False）。
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        y_iso = iso.fit_transform(x, y, sample_weight=w)
        table[nz] = y_iso
        # 未観測 bin は最近傍の観測値で埋める（区分定数の自然な外挿）。
        obs_idx = np.nonzero(nz)[0]
        all_idx = np.arange(n_bins)
        nearest = obs_idx[np.searchsorted(obs_idx, all_idx).clip(0, len(obs_idx) - 1)]
        # searchsorted は右側最近傍なので左側候補とも比較して近い方を採用。
        left = np.clip(np.searchsorted(obs_idx, all_idx) - 1, 0, len(obs_idx) - 1)
        nearest_left = obs_idx[left]
        choose_left = np.abs(all_idx - nearest_left) < np.abs(all_idx - nearest)
        nearest = np.where(choose_left, nearest_left, nearest)
        filled = table[nearest]
        table[~nz] = filled[~nz]
    return table


def fit_lookup_simple(
    bins: np.ndarray,
    sum_y: np.ndarray,
    count: np.ndarray,
    *,
    index_offset: int,
    n_bins: int,
    fallback: float = GLOBAL_MEAN_FALLBACK,
) -> np.ndarray:
    """汎用の条件付き平均 lookup（isotonic なし、未観測は fallback）。

    splitdiff（d=-255..255）用。index = bin - index_offset。
    """
    dense_n = np.zeros(n_bins, dtype=np.float64)
    dense_sy = np.zeros(n_bins, dtype=np.float64)
    idx = (bins.astype(np.int64) - index_offset)
    np.add.at(dense_n, idx, count)
    np.add.at(dense_sy, idx, sum_y)
    table = np.full(n_bins, fallback, dtype=np.float64)
    nz = dense_n > 0
    table[nz] = dense_sy[nz] / dense_n[nz]
    # 未観測 bin は最近傍観測で埋める。
    if nz.any():
        obs_idx = np.nonzero(nz)[0]
        all_idx = np.arange(n_bins)
        pos = np.searchsorted(obs_idx, all_idx).clip(0, len(obs_idx) - 1)
        nearest = obs_idx[pos]
        left = np.clip(pos - 1, 0, len(obs_idx) - 1)
        nearest_left = obs_idx[left]
        choose_left = np.abs(all_idx - nearest_left) < np.abs(all_idx - nearest)
        nearest = np.where(choose_left, nearest_left, nearest)
        table[~nz] = table[nearest][~nz]
    return table


# --- パラメトリック fit（充足統計上の重み付き最小二乗） ---------------------


def _powerlaw_pred(params: np.ndarray, dn: np.ndarray) -> np.ndarray:
    a, b = params
    return a * np.power(np.clip((255.0 - dn) / 255.0, 0.0, 1.0), b)


def _exp_pred(params: np.ndarray, dn: np.ndarray) -> np.ndarray:
    a, b, c = params
    return a * np.exp(-b * dn) + c


def _weighted_residuals(pred_bin: np.ndarray, mean_bin: np.ndarray, count: np.ndarray):
    """bin 平均との残差に sqrt(count) 重みを掛けたもの。

    画素 MSE Σ_pix(y-ŷ)² = Σ_bin[ Σy² - 2ŷΣy + ŷ²n ] の ŷ 依存部分の最小化は
    Σ_bin n·(ŷ_bin - mean_bin)² の最小化と同値（Σy² は定数, mean=Σy/n）。
    よって残差 = sqrt(n)·(ŷ_bin - mean_bin) を least_squares に与えればよい。
    """
    return np.sqrt(count) * (pred_bin - mean_bin)


def fit_powerlaw(
    dn: np.ndarray, sum_y: np.ndarray, count: np.ndarray
) -> tuple[float, float]:
    """RR=a·((255-DN)/255)^b を画素 MSE 重み付き最小二乗で fit。"""
    mean = _weighted_mean(sum_y, count)
    m = count > 0
    dn_m, mean_m, cnt_m = dn[m].astype(np.float64), mean[m], count[m].astype(np.float64)

    def resid(p):
        return _weighted_residuals(_powerlaw_pred(p, dn_m), mean_m, cnt_m)

    x0 = np.array([5.0, 3.0])
    res = least_squares(
        resid, x0, bounds=([0.0, 0.1], [50.0, 20.0]), method="trf", max_nfev=2000
    )
    return float(res.x[0]), float(res.x[1])


def fit_exp(
    dn: np.ndarray, sum_y: np.ndarray, count: np.ndarray
) -> tuple[float, float, float]:
    """RR=a·exp(-b·DN)+c を画素 MSE 重み付き最小二乗で fit。"""
    mean = _weighted_mean(sum_y, count)
    m = count > 0
    dn_m, mean_m, cnt_m = dn[m].astype(np.float64), mean[m], count[m].astype(np.float64)

    def resid(p):
        return _weighted_residuals(_exp_pred(p, dn_m), mean_m, cnt_m)

    x0 = np.array([5.0, 0.02, 0.0])
    res = least_squares(
        resid, x0, bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 5.0]),
        method="trf", max_nfev=4000,
    )
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


# --- 予測ベクトル化（fit 済みモデル → bin ごとの ŷ） -----------------------


@dataclass
class FoldFit:
    """1 fold（= 学習 fold 群）で fit したモデルの予測子。

    predict(bins) は観測 bin 配列に対し ŷ を返す（負値は 0 クリップ）。
    """

    kind: str  # "window_lookup" / "window_powerlaw" / "window_exp" / "splitdiff_lookup"
    table: np.ndarray | None = None          # lookup 用 dense 配列
    index_offset: int = 0                    # lookup の bin→index オフセット
    params: tuple[float, ...] = field(default_factory=tuple)  # パラメトリック用

    def predict(self, bins: np.ndarray) -> np.ndarray:
        if self.kind in ("window_lookup", "splitdiff_lookup"):
            idx = (bins.astype(np.int64) - self.index_offset)
            idx = np.clip(idx, 0, len(self.table) - 1)
            yhat = self.table[idx]
        elif self.kind == "window_powerlaw":
            yhat = _powerlaw_pred(np.asarray(self.params), bins.astype(np.float64))
        elif self.kind == "window_exp":
            yhat = _exp_pred(np.asarray(self.params), bins.astype(np.float64))
        else:
            raise ValueError(f"未知の kind: {self.kind}")
        return np.clip(yhat, 0.0, None)


# --- window_lookup / per_satellite の refit と閉形式 CV（train.py 用） -------
#
# Phase1 最良モデル（window_lookup / per_satellite）の確定 lookup を全 TRAIN から
# fit し、conf/folds.yaml の手設計 fold を正準として地域 GroupKFold CV RMSE を
# 充足統計キャッシュ（phase1_window_hist.parquet）から閉形式で再現する。
# train.py から呼ばれ、outputs/phase1_model.json と同型の確定テーブルを得る。


def fit_window_lookup_per_satellite(
    win_hist,
    *,
    n_bins: int = 256,
    isotonic: bool = True,
    fallback: float = GLOBAL_MEAN_FALLBACK,
) -> dict[str, np.ndarray]:
    """全 TRAIN の窓 DN ヒストグラムから衛星別 256bin lookup を fit する。

    Args:
        win_hist: 列 satellite, name_location, dn, count, sum_y, sum_y2 の DataFrame。
        n_bins: lookup 長（窓 DN は 256）。
        isotonic: DN 増 → RR 単調減を課す isotonic 平滑化。
        fallback: 未観測 DN の既定値。

    Returns:
        satellite → 長さ n_bins の float64 lookup（index=DN）。
    """
    tables: dict[str, np.ndarray] = {}
    for sat, g in win_hist.groupby("satellite"):
        bins = g["dn"].to_numpy()
        sum_y = g["sum_y"].to_numpy(dtype=np.float64)
        count = g["count"].to_numpy(dtype=np.float64)
        tables[str(sat)] = fit_lookup_window(
            bins, sum_y, count, n_bins=n_bins, isotonic=isotonic, fallback=fallback
        )
    return tables


def cv_rmse_window_lookup_per_satellite(
    win_hist,
    loc_to_fold: dict[str, int],
    *,
    n_splits: int = 5,
    n_bins: int = 256,
    isotonic: bool = True,
    fallback: float = GLOBAL_MEAN_FALLBACK,
) -> dict:
    """手設計 fold で window_lookup/per_satellite の閉形式 CV RMSE を算出する。

    各 fold で「val を除く学習 fold の充足統計」から衛星別 lookup を fit し、
    val 集合の (satellite, dn) ヒストグラムへ適用して二乗誤差和を閉形式で得る。
    生画素は不要（sum_y2 - 2·f·sum_y + f²·count を積算）。

    Args:
        win_hist: 窓 DN ヒストグラム DataFrame（satellite, name_location, dn,
            count, sum_y, sum_y2）。
        loc_to_fold: name_location → fold（conf/folds.yaml の手設計マップ）。
        n_splits: fold 数。
        n_bins / isotonic / fallback: fit パラメータ。

    Returns:
        {"overall": float, "per_fold": [float,...], "per_satellite": {sat: rmse},
         "n_pixels": int}。
    """
    hist = win_hist.copy()
    hist["fold"] = hist["name_location"].map(loc_to_fold)
    # fold 未割当（EVAL 地域や未知地域）の行は CV から除外。
    hist = hist[hist["fold"].notna()].copy()
    hist["fold"] = hist["fold"].astype(int)

    oof_sse = 0.0
    oof_npix = 0.0
    per_fold: list[float | None] = []
    # 衛星別 OOF 集計。
    sat_sse: dict[str, float] = {}
    sat_npix: dict[str, float] = {}

    for f in range(n_splits):
        trn = hist[hist["fold"] != f]
        val = hist[hist["fold"] == f]
        if len(val) == 0:
            per_fold.append(None)
            continue
        # 学習 fold から衛星別 lookup を fit。
        tables = fit_window_lookup_per_satellite(
            trn, n_bins=n_bins, isotonic=isotonic, fallback=fallback
        )
        fold_sse = 0.0
        fold_npix = 0.0
        for sat, gv in val.groupby("satellite"):
            sat = str(sat)
            table = tables.get(sat)
            dn = gv["dn"].to_numpy()
            sy = gv["sum_y"].to_numpy(dtype=np.float64)
            sy2 = gv["sum_y2"].to_numpy(dtype=np.float64)
            cnt = gv["count"].to_numpy(dtype=np.float64)
            if table is None:
                fpred = np.full(len(dn), fallback, dtype=np.float64)
            else:
                idx = np.clip(dn.astype(np.int64), 0, len(table) - 1)
                fpred = np.clip(table[idx], 0.0, None)
            sse = float(np.sum(sy2 - 2.0 * fpred * sy + (fpred * fpred) * cnt))
            npix = float(cnt.sum())
            fold_sse += sse
            fold_npix += npix
            sat_sse[sat] = sat_sse.get(sat, 0.0) + sse
            sat_npix[sat] = sat_npix.get(sat, 0.0) + npix
        per_fold.append(float(np.sqrt(fold_sse / fold_npix)) if fold_npix > 0 else None)
        oof_sse += fold_sse
        oof_npix += fold_npix

    overall = float(np.sqrt(oof_sse / oof_npix)) if oof_npix > 0 else None
    per_satellite = {
        s: float(np.sqrt(sat_sse[s] / sat_npix[s])) for s in sat_sse if sat_npix[s] > 0
    }
    return {
        "overall": overall,
        "per_fold": per_fold,
        "per_satellite": per_satellite,
        "n_pixels": int(oof_npix),
    }


def satellite_means(win_hist) -> dict[str, float]:
    """衛星別の画素重み付き平均 RR（フォールバック候補・参考用）。"""
    out: dict[str, float] = {}
    for sat, g in win_hist.groupby("satellite"):
        sy = float(g["sum_y"].sum())
        cnt = float(g["count"].sum())
        if cnt > 0:
            out[str(sat)] = sy / cnt
    return out
