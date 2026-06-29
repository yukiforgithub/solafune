"""データ I/O: GeoTIFF 読み書き・フレームリスト解析・メタ CSV 読み込み。

ターゲット GPM tif と入力衛星 tif、提出用予測 tif の読み書きを一元化する。
提出 tif は sample_submission と同一プロファイル（GTiff / float32 / 1band /
41x41 / CRS なし / nodata なし / 恒等変換）で出力する。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from . import config

PathLike = Union[str, Path]


# --- 読み込み ---------------------------------------------------------------


def read_target(path: PathLike) -> np.ndarray:
    """ターゲット GPM-IMERG tif を読み込み float32 (41, 41) で返す。

    Args:
        path: GPM-IMERG GeoTIFF のパス。

    Returns:
        形状 (41, 41) の float32 配列（mm/hr）。
    """
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32, copy=False)
    return arr


def read_satellite(path: PathLike) -> np.ndarray:
    """入力衛星 tif を読み込み uint8 (16, H, W) で返す。

    Args:
        path: 16band の衛星 GeoTIFF のパス。

    Returns:
        形状 (16, H, W) の uint8 配列（band 軸が先頭）。
    """
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.uint8, copy=False)  # (bands, H, W)
    return arr


def read_satellite_bands(path: PathLike, bands: list[int]) -> np.ndarray:
    """衛星 tif から指定バンドだけを読み uint8 (len(bands), H, W) で返す。

    全 16band を読まずに主要バンドのみデコードしてコスト削減する用途
    （時間特徴抽出で過去フレームの主要 IR/WV バンドだけ要るとき等）。

    Args:
        path: 衛星 GeoTIFF のパス。
        bands: rasterio の 1-based バンド番号リスト（戻り値の band 軸はこの順）。

    Returns:
        形状 (len(bands), H, W) の uint8 配列。
    """
    with rasterio.open(path) as ds:
        arr = ds.read(bands).astype(np.uint8, copy=False)
    return arr


# --- フレームリスト解析 -----------------------------------------------------


def parse_frame_list(cell: object) -> list[str]:
    """CSV の last_30_minutes_observation_filename セルを文字列リストへ。

    CSV 上は ``"['a.tif', 'b.tif', 'c.tif']"`` のような Python リテラル文字列。
    空・NaN・空リストはすべて空リストとして扱う（入力フレーム 0 の行が存在する）。

    Args:
        cell: CSV セルの値（文字列 / NaN / 既にリスト）。

    Returns:
        ファイル名（basename）の list[str]。先頭が古いフレーム。
    """
    if isinstance(cell, (list, tuple)):
        return [str(x) for x in cell]
    if cell is None:
        return []
    if isinstance(cell, float) and np.isnan(cell):
        return []
    s = str(cell).strip()
    if s == "" or s == "[]":
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    return [str(parsed)]


def satellite_frame_paths(satellite: str, frame_names: list[str], *, train: bool) -> list[Path]:
    """フレーム basename を実ファイルの絶対パスへ解決する。

    Args:
        satellite: "himawari" / "goes" / "meteosat"。
        frame_names: parse_frame_list で得た basename リスト。
        train: True なら TRAIN、False なら EVAL の衛星ディレクトリを参照。

    Returns:
        絶対パスの list[Path]（存在チェックはしない）。
    """
    base = config.TRAIN_DIR if train else config.EVAL_DIR
    subdir = base / config.SATELLITE_DIRNAMES[satellite]
    return [subdir / name for name in frame_names]


# --- 書き出し ---------------------------------------------------------------


def write_prediction_tif(path: PathLike, arr: np.ndarray) -> None:
    """予測配列を提出形式の 1band float32 GeoTIFF として書き出す。

    sample_submission と同一プロファイル: GTiff / float32 / count=1 /
    41x41 / CRS なし / nodata なし / 恒等アフィン変換。

    Args:
        path: 出力先パス（親ディレクトリが無ければ作成する）。
        arr: 形状 (41, 41) の配列（float32 にキャストして書き出す）。

    Raises:
        ValueError: 形状が TARGET_SIZE と異なる場合。
    """
    out = np.asarray(arr, dtype=np.float32)
    if out.shape != config.TARGET_SIZE:
        raise ValueError(
            f"write_prediction_tif: 形状 {out.shape} は TARGET_SIZE {config.TARGET_SIZE} と一致しません。"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = config.TARGET_SIZE
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": w,
        "height": h,
        "count": 1,
        "crs": None,
        "transform": Affine.identity(),
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(out, 1)


# --- メタ CSV 読み込み ------------------------------------------------------


def _load_df(csv_path: Path) -> pd.DataFrame:
    """メタ CSV を読み込み、frame_list 列を付与した DataFrame を返す。"""
    df = pd.read_csv(csv_path)
    df["frame_list"] = df["last_30_minutes_observation_filename"].map(parse_frame_list)
    df["n_frames"] = df["frame_list"].map(len)
    return df


def load_train_df() -> pd.DataFrame:
    """TRAIN メタ CSV を読み込む（frame_list / n_frames 列を付与）。"""
    return _load_df(config.TRAIN_CSV)


def load_eval_df() -> pd.DataFrame:
    """EVAL メタ CSV を読み込む（frame_list / n_frames 列を付与）。"""
    return _load_df(config.EVAL_CSV)
