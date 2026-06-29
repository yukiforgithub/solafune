"""Phase1 充足統計キャッシュの構築エントリスクリプト。

TRAIN 全行を1パス走査し、窓 DN / split-window 差の画素ヒストグラム集計を
2 つの parquet（eda_cache/phase1_window_hist.parquet,
eda_cache/phase1_splitdiff_hist.parquet）へ保存する。以後の Phase1 の全
fit / 地域 GroupKFold CV はこのキャッシュから再読込なしの閉形式で行う。

実行:
    uv run python -m src.build_phase1_suffstats
または
    uv run python src/build_phase1_suffstats.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# パッケージとして実行されない場合（直接実行）でも import を通す。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.precip import config  # noqa: E402
from src.precip.dataio import load_train_df  # noqa: E402
from src.precip.phase1_suffstats import build_suffstats  # noqa: E402

WINDOW_PARQUET = config.EDA_CACHE_DIR / "phase1_window_hist.parquet"
SPLITDIFF_PARQUET = config.EDA_CACHE_DIR / "phase1_splitdiff_hist.parquet"


def _spearman_from_hist(
    df: pd.DataFrame, bin_col: str
) -> float:
    """ヒストグラム集計から bin と画素降水の Spearman 順位相関を算出する。

    各 bin は固定値なので、bin 内の全画素が同順位（中央順位）を取る tie として
    順位を割り当て、Σ(rank_x - mean)(rank_y - mean) を tie 補正付きの標準
    Spearman 公式で計算する。ただし sum_y2 だけでは y の bin 内分布が分からない
    ため、y 側は「bin 平均降水 = sum_y/count を bin 代表値」とみなした近似順位
    相関（bin レベル重み付き）で符号と概略を確認する。

    ここでは符号確認が主目的なので、(bin, mean_y, weight=count) を点とした
    重み付き Spearman（bin 平均 RR を用いる）を返す。
    """
    g = (
        df.groupby(bin_col, as_index=False)[["count", "sum_y"]]
        .sum()
        .sort_values(bin_col)
    )
    g = g[g["count"] > 0]
    if len(g) < 3:
        return float("nan")
    x = g[bin_col].to_numpy(dtype=np.float64)
    mean_y = g["sum_y"].to_numpy() / g["count"].to_numpy()
    w = g["count"].to_numpy(dtype=np.float64)

    rx = _weighted_rank(x, w)
    ry = _weighted_rank(mean_y, w)
    return _weighted_corr(rx, ry, w)


def _weighted_rank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """重み付き平均順位（tie は中央順位）を返す。

    各ユニーク値に「累積重み下端 + (自重み+1)/2」を割り当てる近似順位。
    bin 値はユニークなので順序ソートで素直に割り当てる。
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cum = 0.0
    for idx in order:
        wgt = weights[idx]
        ranks[idx] = cum + (wgt + 1.0) / 2.0
        cum += wgt
    return ranks


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """重み付きピアソン相関（順位に適用すると重み付き Spearman）。"""
    sw = w.sum()
    ma = (a * w).sum() / sw
    mb = (b * w).sum() / sw
    da = a - ma
    db = b - mb
    cov = (w * da * db).sum()
    va = (w * da * da).sum()
    vb = (w * db * db).sum()
    if va <= 0 or vb <= 0:
        return float("nan")
    return float(cov / np.sqrt(va * vb))


def main() -> None:
    config.ensure_output_dirs()
    t0 = time.time()

    print("TRAIN メタ CSV 読み込み中...", flush=True)
    df = load_train_df()
    print(f"  行数={len(df)}  地域={df['name_location'].nunique()}  "
          f"衛星別={df['satellite_target'].value_counts().to_dict()}", flush=True)

    print("充足統計を1パス構築中（窓 DN / split 差）...", flush=True)
    win_df, diff_df, rows_read, rows_skipped = build_suffstats(
        df, max_workers=8, progress_every=2000
    )
    dt = time.time() - t0
    print(f"走査完了: rows_read={rows_read} rows_skipped={rows_skipped} "
          f"({dt:.1f}s)", flush=True)

    win_df.to_parquet(WINDOW_PARQUET, index=False)
    diff_df.to_parquet(SPLITDIFF_PARQUET, index=False)
    print(f"保存: {WINDOW_PARQUET} ({len(win_df)} 行)", flush=True)
    print(f"保存: {SPLITDIFF_PARQUET} ({len(diff_df)} 行)", flush=True)

    # --- 衛星別 Spearman（符号確認）。窓は負相関のはず ---
    print("\n衛星別 Spearman（bin 平均 RR ベース, 符号確認）:", flush=True)
    sats = sorted(set(win_df["satellite"].unique()) | set(diff_df["satellite"].unique()))
    for sat in sats:
        sp_win = _spearman_from_hist(win_df[win_df["satellite"] == sat], "dn")
        sp_diff = _spearman_from_hist(diff_df[diff_df["satellite"] == sat], "diff")
        print(f"  {sat:10s} window={sp_win:+.4f}  splitdiff={sp_diff:+.4f}", flush=True)


if __name__ == "__main__":
    main()
