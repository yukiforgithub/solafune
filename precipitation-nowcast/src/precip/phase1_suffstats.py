"""Phase1 充足統計（sufficient statistics）の1パス構築。

IR 窓 Tb（明るさ温度の代理 DN）単独の Tb→RR 関係を地域 CV でフィットするため、
TRAIN 全行を1度だけ走査して画素レベルのヒストグラム集計を作る。集計は

  - F1 = 窓 DN（0..255 の整数, 衛星別 IR 窓バンド）
  - F2 = split-window 差 d = int(窓DN) - int(splitDN)（範囲 -255..255）

の2特徴について、(satellite, name_location, bin) ごとに
``count`` / ``sum_y`` / ``sum_y2`` を積算する。これにより任意の写像 f(feature)→RR
の画素 RMSE と地域 GroupKFold CV が再読込なしの閉形式で算出できる
（RMSE^2 = Σsum_y2/N - 2 Σ f·sum_y/N + Σ f^2·count/N）。

設計判断:
  - フレーム選択は「最新（リスト最後）」。Phase0/Phase1 で確立済みの規約に従う。
  - 入力はネイティブ解像度 → cv2.INTER_AREA で 41x41 へ縮約してからターゲットと画素対応。
  - 窓 / split バンドは rasterio の 1-based read index（衛星別）。
  - 0 フレーム行・読み込み失敗行はスキップし件数を記録（気候値フォールバックは fit 側の責務）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

from . import config
from .dataio import parse_frame_list

# --- 衛星別 IR 窓 / split バンド（rasterio 1-based read index） ---------------
# 窓 ~10.x µm（冷たい雲頂 = 強雨, E[y|DN] 単調減少）、split 相手 ~12.x µm。
# Phase0/Phase1 で検証済みの index 規約。
WINDOW_BAND_INDEX: dict[str, int] = {
    "himawari": 13,
    "goes": 13,
    "meteosat": 14,
}
SPLIT_BAND_INDEX: dict[str, int] = {
    "himawari": 15,
    "goes": 15,
    "meteosat": 15,
}

# --- ビン定義 ---------------------------------------------------------------
DN_BINS: int = 256          # 窓 DN: 0..255
DIFF_MIN: int = -255        # split-window 差の最小値
DIFF_MAX: int = 255         # split-window 差の最大値
DIFF_BINS: int = DIFF_MAX - DIFF_MIN + 1  # 511

TARGET_H, TARGET_W = config.TARGET_SIZE  # (41, 41)


@dataclass
class RowResult:
    """1 行ぶんの画素特徴とターゲット（集計前の中間表現）。"""

    satellite: str
    name_location: str
    window_dn: np.ndarray  # uint8 (N,)  N=41*41
    diff: np.ndarray       # int16 (N,)  範囲 -255..255
    y: np.ndarray          # float64 (N,)


def _resize_area(band: np.ndarray) -> np.ndarray:
    """ネイティブ解像度のバンドを INTER_AREA で 41x41 へ縮約する。"""
    return cv2.resize(
        band, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA
    )


def _process_row(
    satellite: str,
    name_location: str,
    frame_path: Path,
    target_path: Path,
) -> RowResult | None:
    """1 行を処理して画素特徴とターゲットを返す。失敗時は None。

    窓 / split バンドを read → INTER_AREA で 41x41 → ターゲット読み込み →
    画素ごとに窓 DN・split 差・降水を平坦化して返す。
    """
    w_idx = WINDOW_BAND_INDEX[satellite]
    s_idx = SPLIT_BAND_INDEX[satellite]
    try:
        with rasterio.open(frame_path) as ds:
            win_native = ds.read(w_idx)   # ネイティブ HxW
            spl_native = ds.read(s_idx)
        with rasterio.open(target_path) as tds:
            y = tds.read(1).astype(np.float64, copy=False)
    except (rasterio.errors.RasterioIOError, IndexError, OSError):
        return None

    # 窓 DN は 41x41 縮約後に丸めて 0..255 の整数化。
    win41 = _resize_area(win_native)
    spl41 = _resize_area(spl_native)
    # INTER_AREA は uint8 入力なら uint8 を返すが、安全のため明示クリップ。
    win_dn = np.clip(np.rint(win41), 0, 255).astype(np.int16)
    spl_dn = np.clip(np.rint(spl41), 0, 255).astype(np.int16)

    if win_dn.shape != (TARGET_H, TARGET_W) or y.shape != (TARGET_H, TARGET_W):
        return None

    diff = (win_dn - spl_dn).astype(np.int16)  # -255..255
    return RowResult(
        satellite=satellite,
        name_location=name_location,
        window_dn=win_dn.reshape(-1).astype(np.uint8),
        diff=diff.reshape(-1),
        y=y.reshape(-1),
    )


def _accumulate(
    win_acc: dict[tuple[str, str], np.ndarray],
    diff_acc: dict[tuple[str, str], np.ndarray],
    r: RowResult,
) -> None:
    """RowResult を (satellite, name_location) 別の集計配列へ積算する。

    各集計配列は形状 (n_bins, 3): 列 = [count, sum_y, sum_y2]。
    np.bincount を使い weights で sum_y / sum_y2 を同時に作る。
    """
    key = (r.satellite, r.name_location)
    y = r.y
    y2 = y * y

    # --- 窓 DN ヒストグラム ---
    wa = win_acc.get(key)
    if wa is None:
        wa = np.zeros((DN_BINS, 3), dtype=np.float64)
        win_acc[key] = wa
    wdn = r.window_dn.astype(np.int64)
    wa[:, 0] += np.bincount(wdn, minlength=DN_BINS)
    wa[:, 1] += np.bincount(wdn, weights=y, minlength=DN_BINS)
    wa[:, 2] += np.bincount(wdn, weights=y2, minlength=DN_BINS)

    # --- split 差ヒストグラム（-255..255 → 0..510 にシフト） ---
    da = diff_acc.get(key)
    if da is None:
        da = np.zeros((DIFF_BINS, 3), dtype=np.float64)
        diff_acc[key] = da
    didx = (r.diff.astype(np.int64) - DIFF_MIN)
    da[:, 0] += np.bincount(didx, minlength=DIFF_BINS)
    da[:, 1] += np.bincount(didx, weights=y, minlength=DIFF_BINS)
    da[:, 2] += np.bincount(didx, weights=y2, minlength=DIFF_BINS)


def build_suffstats(
    df: pd.DataFrame,
    *,
    max_workers: int = 8,
    progress_every: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """TRAIN を1パス走査して窓 DN / split 差の充足統計を構築する。

    Args:
        df: TRAIN メタ DataFrame（name_location, satellite_target,
            last_30_minutes_observation_filename, gpm_imerg_filename を含む）。
        max_workers: ThreadPoolExecutor のスレッド数（rasterio I/O は GIL を
            解放するためスレッドで実効並列が効く）。
        progress_every: 進捗ログの間隔（行数）。

    Returns:
        (window_hist_df, splitdiff_hist_df, rows_read, rows_skipped)。
        window_hist_df: 列 satellite, name_location, dn, count, sum_y, sum_y2。
        splitdiff_hist_df: 列 satellite, name_location, diff, count, sum_y, sum_y2。
    """
    # 走査対象タスク（0 フレーム行はここでスキップ）。
    tasks: list[tuple[str, str, Path, Path]] = []
    skipped_zero_frame = 0
    for row in df.itertuples(index=False):
        sat = row.satellite_target
        loc = row.name_location
        frames = parse_frame_list(row.last_30_minutes_observation_filename)
        if not frames:
            skipped_zero_frame += 1
            continue
        latest = frames[-1]  # 最新（リスト最後）フレーム
        subdir = config.TRAIN_DIR / config.SATELLITE_DIRNAMES[sat]
        frame_path = subdir / latest
        target_path = config.TRAIN_TARGET_DIR / row.gpm_imerg_filename
        tasks.append((sat, loc, frame_path, target_path))

    win_acc: dict[tuple[str, str], np.ndarray] = {}
    diff_acc: dict[tuple[str, str], np.ndarray] = {}
    rows_read = 0
    skipped_io = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_row, *t) for t in tasks]
        for done, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if r is None:
                skipped_io += 1
            else:
                _accumulate(win_acc, diff_acc, r)
                rows_read += 1
            if progress_every and done % progress_every == 0:
                print(
                    f"  進捗 {done}/{len(tasks)}  read={rows_read} skip_io={skipped_io}",
                    flush=True,
                )

    win_df = _acc_to_df(win_acc, bin_col="dn", bin_offset=0)
    diff_df = _acc_to_df(diff_acc, bin_col="diff", bin_offset=DIFF_MIN)
    rows_skipped = skipped_zero_frame + skipped_io
    return win_df, diff_df, rows_read, rows_skipped


def _acc_to_df(
    acc: dict[tuple[str, str], np.ndarray], *, bin_col: str, bin_offset: int
) -> pd.DataFrame:
    """集計 dict を long 形式 DataFrame へ。空ビンは落とす（疎に保存）。"""
    parts: list[pd.DataFrame] = []
    for (sat, loc), arr in acc.items():
        nonzero = arr[:, 0] > 0
        if not nonzero.any():
            continue
        idx = np.nonzero(nonzero)[0]
        parts.append(
            pd.DataFrame(
                {
                    "satellite": sat,
                    "name_location": loc,
                    bin_col: (idx + bin_offset).astype(np.int16),
                    "count": arr[idx, 0].astype(np.int64),
                    "sum_y": arr[idx, 1],
                    "sum_y2": arr[idx, 2],
                }
            )
        )
    if not parts:
        return pd.DataFrame(
            columns=["satellite", "name_location", bin_col, "count", "sum_y", "sum_y2"]
        )
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["satellite", "name_location", bin_col]).reset_index(drop=True)
