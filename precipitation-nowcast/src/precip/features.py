"""Phase2 画素特徴抽出（GBDT用）。学習・予測で共通の単一実装。

1 行の直近最大3フレーム（30分・10分間隔）の衛星 tif（uint8 16band）から、
ターゲット格子 41x41 の各画素について特徴ベクトルを作る。前処理(preprocess) /
学習(train) / 予測(predict) すべてが本モジュールを呼ぶことで特徴抽出を厳密に一致させる。

特徴群（合計 97）:
  1. spectral(16) : 最新フレーム 16band の画素値（INTER_AREA で 41x41 へ縮約）
  2. BTD(2)       : split-window 差(win-split), 水蒸気差(wv_upper-win)
  3. 空間統計(32) : 主要4band{win,split,wv,ir38} × 窓{3,7} × {mean,std,min,max}
  4. 構造(3)      : win の 7x7 Z スコア / 局所コントラスト(x-min7) / Sobel 勾配強度
  5. メタ(7)      : n_frames, hour sin/cos, month sin/cos, 可視昼夜プロキシ平均, day_flag
  6. 位置(3)      : i/(H-1), j/(W-1), 中心からの正規化距離
  7. パッチ(20)   : 主要4band × サブグリッド{mean,median,max,min,std}（INTER_AREA前のネイティブ極値保持）
  8. 時間(14)     : 主要4band × {dt(最新-最古=雲頂冷却率), tmin, tmean} + {win,ir38}の冷却率3x3平滑

設計上の規約（Phase0/1 と一致）:
  - 基本特徴(1-7)は「最新フレーム（リスト最後）」基準。0 フレーム行は None（呼び出し側が気候値フォールバック）。
  - 時間特徴(8)は直近最大3フレーム（古→新）を使用。1フレームのみなら dt=0・tmin/tmean=最新値。
  - 入力はネイティブ解像度 → cv2.INTER_AREA で 41x41 へ縮約してから画素対応。
  - 衛星別の主要バンド index は config.SATELLITE_BANDS / phase1 の WINDOW/SPLIT 定数と整合。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from . import config
from .dataio import read_satellite, read_satellite_bands
from .phase1_suffstats import (
    SPLIT_BAND_INDEX,
    TARGET_H,
    TARGET_W,
    WINDOW_BAND_INDEX,
    _resize_area,
)

PathLike = Union[str, Path]

# --- 主要バンドの 0-based numpy index（read_satellite の band 軸 0..15） -------
# window/split は phase1 の 1-based 定数から変換。wv_upper(6.x µm)/ir38(3.9µm) は
# config.SATELLITE_BANDS と整合（himawari B08/B07, goes C08/C07, meteosat wv_63/ir_38）。
_WV_UPPER_IDX: dict[str, int] = {"himawari": 7, "goes": 7, "meteosat": 9}
_IR38_IDX: dict[str, int] = {"himawari": 6, "goes": 6, "meteosat": 8}
VISIBLE_IDX: tuple[int, ...] = (0, 1, 2)  # 昼夜プロキシ用の可視バンド

SPATIAL_KEYS: tuple[str, ...] = ("win", "split", "wv", "ir38")
WINDOWS: tuple[int, ...] = (3, 7)
STATS: tuple[str, ...] = ("mean", "std", "min", "max")

# サブグリッド・パッチ統計（各出力セル 41x41 に対応するネイティブ入力パッチの集約）。
# feature importance 上位が局所 MIN（最も冷たい雲頂＝対流コア）だった知見に基づき、
# INTER_AREA 平均化で消えるパッチ内の極値を保持するための新特徴。NaN は除外して集計。
PATCH_KEYS: tuple[str, ...] = ("win", "split", "wv", "ir38")
PATCH_STATS: tuple[str, ...] = ("mean", "median", "max", "min", "std")

# 時間特徴（直近最大3フレーム=30分, 10分間隔）。空間統計とは直交する新情報源で、
# 対流の発達/衰弱を捉える。LH24 ディスカッション推奨の雲頂冷却率 BT_t0−BT_t2 を DN 差で
# 近似する（低 DN=冷たい雲頂=強雨 ⇒ 負の dt=雲頂が冷却＝対流発達）。
#   - dt   : 最新フレーム − 最古フレーム（利用可能な時間窓全体の変化。1枚なら 0）
#   - tmin : フレーム方向の最小値（窓内で最も冷たくなった雲頂。移流で最新からは消えうる極値）
#   - tmean: フレーム方向の平均値（時間平均状態）
MAX_FRAMES: int = 3
TEMPORAL_KEYS: tuple[str, ...] = ("win", "split", "wv", "ir38")
TEMPORAL_STATS: tuple[str, ...] = ("dt", "tmin", "tmean")
# 冷却率 dt を 3x3 box-mean で平滑化し移流ノイズを抑えた版（主要 IR2 バンドのみ）。
TEMPORAL_SMOOTH_KEYS: tuple[str, ...] = ("win", "ir38")

# 【棄却済】地域不変なシーン内正規化(scene_z/scene_rank)を試したが、同一データの
# アブレーションで純効果 +0.0002(無効)、LBも 0.70705→0.70790 と悪化のため撤去。
# 入力分布が train/eval で既に揃っており（EDA）、入力適応系は天井が低いと確定。詳細は STATUS §5。

# 【棄却済】物理/対流特徴(冷却分解 cool01/cool_accel + 雲相BTD 8.6µm−IR窓)を試したが、
# 同一データ・同一HP のアブレーション(実験2)で純効果 overall +0.0010 / cond5 +0.032(無効〜微悪化)。
# cond≥5(強雨)を狙ったが逆効果。空間統計・入力適応に続き、この軸も飽和と確定。撤去。詳細 STATUS §5。


def band_indices(satellite: str) -> dict[str, int]:
    """衛星の主要バンド 0-based index（win/split/wv/ir38）を返す。"""
    return {
        "win": WINDOW_BAND_INDEX[satellite] - 1,
        "split": SPLIT_BAND_INDEX[satellite] - 1,
        "wv": _WV_UPPER_IDX[satellite],
        "ir38": _IR38_IDX[satellite],
    }


# --- 位置グリッド（モジュール定数） -----------------------------------------
_II, _JJ = np.meshgrid(np.arange(TARGET_H), np.arange(TARGET_W), indexing="ij")
_POS_I = (_II.astype(np.float32) / max(TARGET_H - 1, 1)).ravel()
_POS_J = (_JJ.astype(np.float32) / max(TARGET_W - 1, 1)).ravel()
_CX, _CY = (TARGET_H - 1) / 2.0, (TARGET_W - 1) / 2.0
_POS_R = (
    np.sqrt((_II - _CX) ** 2 + (_JJ - _CY) ** 2).astype(np.float32)
    / float(np.sqrt(_CX**2 + _CY**2))
).ravel()


# --- 近傍統計ヘルパ（41x41 float32 → 41x41 float32） -------------------------


def _box_mean(x: np.ndarray, k: int) -> np.ndarray:
    return cv2.boxFilter(x, ddepth=-1, ksize=(k, k), normalize=True, borderType=cv2.BORDER_REPLICATE)


def _box_std(x: np.ndarray, k: int) -> np.ndarray:
    m = _box_mean(x, k)
    m2 = _box_mean(x * x, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)


def _box_min(x: np.ndarray, k: int) -> np.ndarray:
    return cv2.erode(x, np.ones((k, k), np.uint8))


def _box_max(x: np.ndarray, k: int) -> np.ndarray:
    return cv2.dilate(x, np.ones((k, k), np.uint8))


_STAT_FN = {"mean": _box_mean, "std": _box_std, "min": _box_min, "max": _box_max}


def patch_block_stats(native_band: np.ndarray) -> dict[str, np.ndarray]:
    """各出力セル(41x41)に対応するネイティブ入力パッチの集約統計を返す（NaN 除外）。

    非整数比（himawari 81/41, goes 141/41, meteosat 144/41）に対応するため、
    ネイティブを 41*k（k=ceil(native/41)）へ最近傍補間でアップサンプルしてから
    k×k ブロックに分割し、ブロックごとに NaN を除外して mean/median/max/min/std を計算する。
    最近傍補間はネイティブ画素値をそのまま保持するため、min/max がパッチ内の真の極値
    （平均化では消える最も冷たい/暖かい画素）を近似する。

    Args:
        native_band: ネイティブ解像度の 1 バンド (H, W) float32。

    Returns:
        {"mean","median","max","min","std"} -> (41, 41) float32 の dict。
    """
    h, w = native_band.shape[:2]
    k = int(np.ceil(max(h, w) / TARGET_H))
    hi = TARGET_H * k
    up = cv2.resize(native_band, (hi, hi), interpolation=cv2.INTER_NEAREST)
    a = up.reshape(TARGET_H, k, TARGET_W, k)
    ax = (1, 3)
    # 全画素 NaN ブロックは nan を返す（衛星 uint8 入力では通常発生しない）。
    return {
        "mean": np.nanmean(a, axis=ax).astype(np.float32),
        "median": np.nanmedian(a, axis=ax).astype(np.float32),
        "max": np.nanmax(a, axis=ax).astype(np.float32),
        "min": np.nanmin(a, axis=ax).astype(np.float32),
        "std": np.nanstd(a, axis=ax).astype(np.float32),
    }


def temporal_block_stats(maps: list[np.ndarray]) -> dict[str, np.ndarray]:
    """同一バンドの時系列 41x41 マップ（古→新, 長さ1-3）から時間統計を返す。

    Args:
        maps: 古→新の順に並んだ 1 バンドの (41,41) float32 マップ列（最低1枚）。

    Returns:
        {"dt","tmin","tmean"} -> (41,41) float32 の dict。
        dt=最新−最古（1枚なら0）, tmin/tmean=フレーム方向の min/mean。
    """
    stk = np.stack(maps, axis=0)  # (T,41,41)
    return {
        "dt": (maps[-1] - maps[0]).astype(np.float32),
        "tmin": stk.min(axis=0).astype(np.float32),
        "tmean": stk.mean(axis=0).astype(np.float32),
    }


# --- 特徴名（抽出の列順と厳密に一致させる） ---------------------------------


def feature_names() -> list[str]:
    """特徴列名を抽出時と同一順序で返す（キャッシュ列順・モデル入力順の正準）。"""
    names: list[str] = [f"band_{i:02d}" for i in range(16)]
    names += ["btd_split", "btd_wv"]
    for key in SPATIAL_KEYS:
        for k in WINDOWS:
            for s in STATS:
                names.append(f"{key}_{s}{k}")
    names += ["win_z7", "win_contrast7", "win_grad"]
    names += ["n_frames", "hour_sin", "hour_cos", "month_sin", "month_cos", "vis_mean", "day_flag"]
    names += ["pos_i", "pos_j", "pos_r"]
    # サブグリッド・パッチ統計（NaN 除外）。
    for key in PATCH_KEYS:
        for stat in PATCH_STATS:
            names.append(f"{key}_patch_{stat}")
    # 時間特徴（直近最大3フレーム）。
    for key in TEMPORAL_KEYS:
        for stat in TEMPORAL_STATS:
            names.append(f"{key}_{stat}")
    for key in TEMPORAL_SMOOTH_KEYS:
        names.append(f"{key}_dt_s3")
    return names


N_FEATURES: int = len(feature_names())  # = 97


# --- 抽出本体 ---------------------------------------------------------------


def resize_stack(arr_uint8: np.ndarray) -> np.ndarray:
    """(16, H, W) uint8 → (16, 41, 41) float32（各バンド INTER_AREA 縮約）。"""
    return np.stack(
        [_resize_area(arr_uint8[b].astype(np.float32)) for b in range(arr_uint8.shape[0])],
        axis=0,
    )


def extract_features_from_stack(
    stack: np.ndarray, satellite: str, n_frames: int, hour: int, month: int,
    patch: dict[str, dict[str, np.ndarray]],
    temporal: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """(16,41,41) float32 と パッチ/時間統計から画素特徴 (1681, N_FEATURES) float32 を作る。

    列順は feature_names() と厳密に一致させる。

    Args:
        patch: PATCH_KEYS ごとの {stat: (41,41)} 辞書（patch_block_stats の出力）。
        temporal: TEMPORAL_KEYS ごとの {dt,tmin,tmean: (41,41)} 辞書（temporal_block_stats の出力）。
    """
    idx = band_indices(satellite)
    win = stack[idx["win"]]
    split = stack[idx["split"]]
    wv = stack[idx["wv"]]
    ir38 = stack[idx["ir38"]]
    bands_map = {"win": win, "split": split, "wv": wv, "ir38": ir38}

    cols: list[np.ndarray] = []

    # 1. spectral(16)
    for i in range(16):
        cols.append(stack[i].ravel())

    # 2. BTD(2)
    cols.append((win - split).ravel())
    cols.append((wv - win).ravel())

    # 3. 空間統計(32)
    for key in SPATIAL_KEYS:
        b = bands_map[key]
        for k in WINDOWS:
            for s in STATS:
                cols.append(_STAT_FN[s](b, k).ravel())

    # 4. 構造(3) — win バンド
    m7 = _box_mean(win, 7)
    s7 = _box_std(win, 7)
    mn7 = _box_min(win, 7)
    z7 = (win - m7) / (s7 + 1e-6)
    contrast7 = win - mn7
    gx = cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    cols.append(z7.ravel())
    cols.append(contrast7.ravel())
    cols.append(grad.ravel())

    # 5. メタ(7) — 行スカラを全画素へブロードキャスト
    n = TARGET_H * TARGET_W
    vis_mean = float(np.mean([stack[v] for v in VISIBLE_IDX]))
    cols.append(np.full(n, float(n_frames), np.float32))
    cols.append(np.full(n, np.float32(np.sin(2 * np.pi * (hour % 24) / 24.0)), np.float32))
    cols.append(np.full(n, np.float32(np.cos(2 * np.pi * (hour % 24) / 24.0)), np.float32))
    cols.append(np.full(n, np.float32(np.sin(2 * np.pi * ((month - 1) % 12) / 12.0)), np.float32))
    cols.append(np.full(n, np.float32(np.cos(2 * np.pi * ((month - 1) % 12) / 12.0)), np.float32))
    cols.append(np.full(n, np.float32(vis_mean), np.float32))
    cols.append(np.full(n, np.float32(1.0 if vis_mean > 20.0 else 0.0), np.float32))

    # 6. 位置(3)
    cols.append(_POS_I)
    cols.append(_POS_J)
    cols.append(_POS_R)

    # 7. サブグリッド・パッチ統計（NaN 除外）。feature_names() と同一順序。
    for key in PATCH_KEYS:
        ps = patch[key]
        for stat in PATCH_STATS:
            cols.append(ps[stat].ravel())

    # 8. 時間特徴（直近最大3フレーム）。feature_names() と同一順序。
    for key in TEMPORAL_KEYS:
        ts = temporal[key]
        for stat in TEMPORAL_STATS:
            cols.append(ts[stat].ravel())
    for key in TEMPORAL_SMOOTH_KEYS:
        cols.append(_box_mean(temporal[key]["dt"], 3).ravel())

    X = np.column_stack(cols).astype(np.float32)
    assert X.shape[1] == N_FEATURES, f"特徴数不一致: {X.shape[1]} != {N_FEATURES}"
    return X


def extract_features_for_row(
    satellite: str, frame_list: list[str], hour: int, month: int, *, train: bool
) -> np.ndarray | None:
    """1 行の画素特徴 (1681, N_FEATURES) を返す。0フレーム/最新フレーム失敗は None。

    基本特徴(spectral/空間/パッチ等)は最新フレーム基準。時間特徴は直近最大3フレーム
    （古→新）の主要バンドから算出する。最新フレームが読めなければ None（呼び出し側で
    気候値フォールバック）。過去フレームの読込失敗は単に時間窓が短くなるだけで無視する。
    """
    if not frame_list:
        return None
    subdir = (config.TRAIN_DIR if train else config.EVAL_DIR) / config.SATELLITE_DIRNAMES[satellite]
    idx = band_indices(satellite)

    use = frame_list[-MAX_FRAMES:]  # 古→新, 最大3フレーム
    key_band_1based = [idx[k] + 1 for k in TEMPORAL_KEYS]  # rasterio 1-based
    arr_latest: np.ndarray | None = None
    # 主要バンドの時系列 41x41 マップ（古→新）。最新フレーム読込成功時に最低1枚入る。
    key_series: dict[str, list[np.ndarray]] = {k: [] for k in TEMPORAL_KEYS}

    for pos, fname in enumerate(use):
        is_latest = pos == len(use) - 1
        try:
            if is_latest:
                arr = read_satellite(subdir / fname)  # (16, H, W) uint8
                if arr.shape[0] != config.N_INPUT_BANDS:
                    continue
                arr_latest = arr
                for k in TEMPORAL_KEYS:
                    key_series[k].append(_resize_area(arr[idx[k]].astype(np.float32)))
            else:
                # 過去フレームは主要バンドのみ読む（デコード節約）。band 軸=TEMPORAL_KEYS 順。
                arr = read_satellite_bands(subdir / fname, key_band_1based)
                for ki, k in enumerate(TEMPORAL_KEYS):
                    key_series[k].append(_resize_area(arr[ki].astype(np.float32)))
        except Exception:
            continue

    if arr_latest is None:
        return None
    stack = resize_stack(arr_latest)
    # サブグリッド・パッチ統計は INTER_AREA 前のネイティブ画素から計算（極値を保持）。
    patch = {key: patch_block_stats(arr_latest[idx[key]].astype(np.float32)) for key in PATCH_KEYS}
    temporal = {k: temporal_block_stats(key_series[k]) for k in TEMPORAL_KEYS}
    return extract_features_from_stack(
        stack, satellite, len(frame_list), hour, month, patch, temporal
    )
