"""Phase1 物理 IR モデルの推論ユーティリティ（特徴抽出 + lookup 適用）。

Phase1 §10 で確定した最良モデル ``window_lookup / per_satellite`` を、Phase0 の
3 モジュール構成（前処理 / 学習 / 予測）から共通利用するための推論器。

責務:
  - 入力衛星 tif の **最新フレーム** から IR 窓（必要なら split）バンドを読み、
    ``cv2.INTER_AREA`` で 41x41 へ縮約し 0..255 の整数 DN 41x41 を得る（特徴抽出）。
  - 衛星別 256bin lookup table を引いて RR を 41x41 へ写像する（マッピング適用）。
  - 予測の負値を 0 にクリップ（降水は非負）。
  - フレーム 0 / 読み込み失敗時は気候値（global_mean_fallback、または衛星平均）へ
    フォールバックする。

充足統計の構築（src/precip/phase1_suffstats.py）と**完全に同じ**特徴抽出規約を使う
（最新フレーム・INTER_AREA・衛星別 1-based バンド index・rint+clip 整数化）。学習と
推論で前処理を厳密一致させるため、バンド index 定数も suffstats から再利用する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import rasterio

from . import config
from .phase1_suffstats import (
    SPLIT_BAND_INDEX,
    TARGET_H,
    TARGET_W,
    WINDOW_BAND_INDEX,
    _resize_area,
)

PathLike = Union[str, Path]

# Phase1 で確定したモデル成果物（衛星別 256bin lookup table）。
PHASE1_MODEL_JSON: Path = config.OUTPUTS_DIR / "phase1_model.json"

# 推論メソッド名（Phase0 の定数 BASELINE_METHODS とは別系統）。
METHOD_PHYSICAL_IR: str = "physical_ir"


# --- モデル読み込み ---------------------------------------------------------


@dataclass
class PhysicalIRModel:
    """Phase1 物理 IR 推論器（衛星別 DN→RR lookup）。

    Attributes:
        tables: satellite → 長さ 256 の float32 lookup（index=DN）。
        window_band_index: satellite → rasterio 1-based 窓バンド index。
        split_band_index: satellite → rasterio 1-based split バンド index（参照用）。
        global_mean_fallback: 0 フレーム / 読込失敗 / 未知衛星のフォールバック値。
        satellite_mean: satellite → 衛星別平均（フォールバックの優先候補）。
    """

    tables: dict[str, np.ndarray]
    window_band_index: dict[str, int]
    split_band_index: dict[str, int]
    global_mean_fallback: float
    satellite_mean: dict[str, float]

    @classmethod
    def from_json(cls, path: PathLike = PHASE1_MODEL_JSON) -> "PhysicalIRModel":
        """outputs/phase1_model.json を読み、推論器を構築する。"""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
        tables: dict[str, np.ndarray] = {}
        sat_mean: dict[str, float] = {}
        for sat, entry in spec.get("per_satellite", {}).items():
            tbl = np.asarray(entry["table"], dtype=np.float32)
            # 念のため負値 0 クリップ（lookup は推論時もクリップする規約）。
            tbl = np.clip(tbl, 0.0, None)
            tables[sat] = tbl
            # 衛星別フォールバックは保存があれば使う（無ければ後段で global にフォールバック）。
            if "satellite_mean" in entry:
                sat_mean[sat] = float(entry["satellite_mean"])
        return cls(
            tables=tables,
            window_band_index=dict(
                spec.get("window_band_index", WINDOW_BAND_INDEX)
            ),
            split_band_index=dict(spec.get("split_band_index", SPLIT_BAND_INDEX)),
            global_mean_fallback=float(
                spec.get("global_mean_fallback", config_global_mean())
            ),
            satellite_mean=sat_mean,
        )

    def fallback_value(self, satellite: str) -> float:
        """衛星別平均があればそれを、無ければ気候値（global_mean_fallback）を返す。"""
        if satellite in self.satellite_mean:
            return self.satellite_mean[satellite]
        return self.global_mean_fallback

    def apply_table(self, satellite: str, window_dn: np.ndarray) -> np.ndarray:
        """窓 DN 41x41（int）に衛星別 lookup を引いて RR 41x41 を返す。

        未知衛星は気候値の定数面を返す。予測は負値 0 クリップ。
        """
        table = self.tables.get(satellite)
        if table is None:
            val = self.fallback_value(satellite)
            return np.full((TARGET_H, TARGET_W), max(val, 0.0), dtype=np.float32)
        idx = np.clip(window_dn.astype(np.int64), 0, len(table) - 1)
        rr = table[idx].astype(np.float32)
        return np.clip(rr, 0.0, None)

    def constant_tile(self, satellite: str) -> np.ndarray:
        """0 フレーム行用の気候値フォールバック面（41x41 float32, 非負）。"""
        val = max(self.fallback_value(satellite), 0.0)
        return np.full((TARGET_H, TARGET_W), val, dtype=np.float32)


def config_global_mean() -> float:
    """phase1_fit の GLOBAL_MEAN_FALLBACK（0.2886）を参照する薄いラッパ。"""
    from .phase1_fit import GLOBAL_MEAN_FALLBACK

    return float(GLOBAL_MEAN_FALLBACK)


# --- 特徴抽出（最新フレーム → 窓/split DN 41x41） ---------------------------


def extract_window_dn(
    frame_path: PathLike, satellite: str, window_band_index: dict[str, int]
) -> np.ndarray | None:
    """最新フレーム tif から IR 窓バンドを読み 41x41 の整数 DN を返す。

    suffstats と同一規約: read(1-based index) → INTER_AREA で 41x41 →
    rint + clip(0,255) で整数化。読み込み失敗・形状不一致は None。

    Args:
        frame_path: 衛星 tif（最新フレーム）の絶対パス。
        satellite: 衛星名。
        window_band_index: satellite → 1-based 窓バンド index。

    Returns:
        形状 (41, 41) の int16 DN 配列、または失敗時 None。
    """
    w_idx = window_band_index[satellite]
    try:
        with rasterio.open(frame_path) as ds:
            win_native = ds.read(w_idx)
    except (rasterio.errors.RasterioIOError, IndexError, OSError):
        return None
    win41 = _resize_area(win_native)
    win_dn = np.clip(np.rint(win41), 0, 255).astype(np.int16)
    if win_dn.shape != (TARGET_H, TARGET_W):
        return None
    return win_dn


def extract_window_split_dn(
    frame_path: PathLike,
    satellite: str,
    window_band_index: dict[str, int],
    split_band_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """窓 DN と split 差（窓DN-splitDN）の 41x41 を返す（多変量化の足場・参照用）。

    Phase1 最良モデルは窓 DN 単独だが、Phase2 への足場として split も同時抽出できる
    ようにしておく（predict.py は窓 DN のみ使用）。
    """
    w_idx = window_band_index[satellite]
    s_idx = split_band_index[satellite]
    try:
        with rasterio.open(frame_path) as ds:
            win_native = ds.read(w_idx)
            spl_native = ds.read(s_idx)
    except (rasterio.errors.RasterioIOError, IndexError, OSError):
        return None
    win_dn = np.clip(np.rint(_resize_area(win_native)), 0, 255).astype(np.int16)
    spl_dn = np.clip(np.rint(_resize_area(spl_native)), 0, 255).astype(np.int16)
    if win_dn.shape != (TARGET_H, TARGET_W):
        return None
    diff = (win_dn - spl_dn).astype(np.int16)
    return win_dn, diff


def latest_frame_path(frame_names: list[str], satellite: str, *, train: bool) -> Path | None:
    """フレーム basename リストから「最新（リスト最後）」の絶対パスを返す。

    空リスト（0 フレーム行）は None。Phase0/Phase1 のフレーム選択規約に従う。
    """
    if not frame_names:
        return None
    base = config.TRAIN_DIR if train else config.EVAL_DIR
    subdir = base / config.SATELLITE_DIRNAMES[satellite]
    return subdir / frame_names[-1]


# --- 1 行の予測（特徴抽出 + lookup + フォールバック） -----------------------


def predict_row(
    model: PhysicalIRModel,
    satellite: str,
    frame_names: list[str],
    *,
    train: bool,
) -> np.ndarray:
    """EVAL/TRAIN 1 行の物理 IR 予測 41x41 float32（非負）を返す。

    手順:
      1. 最新フレームパスを解決。0 フレームなら気候値フォールバック面。
      2. 窓 DN を抽出。読み込み失敗なら気候値フォールバック面。
      3. 衛星別 lookup を引いて RR 41x41。負値 0 クリップ。
    """
    path = latest_frame_path(frame_names, satellite, train=train)
    if path is None:
        return model.constant_tile(satellite)
    win_dn = extract_window_dn(path, satellite, model.window_band_index)
    if win_dn is None:
        return model.constant_tile(satellite)
    return model.apply_table(satellite, win_dn)
