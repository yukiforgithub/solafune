"""Phase2 GBDT モデル（衛星別・LightGBM）。

2 つの variant を提供する:
  - two_part（hurdle, 本命）:
      分類器 P(y>=thr) × 有雨回帰 E[log1p(y) | y>=thr] を学習し、
      期待値 pred = P(rain) · expm1(reg) を予測（RMSE 最適な条件付き期待値の近似）。
      学習画素は base rate を保つ一様サンプリング（確率を素直に保つため不均衡補正なし）。
  - tweedie:
      objective='tweedie' の単一回帰で生 mm/hr を直接予測（ゼロ過剰連続値に適合）。

LightGBM の Booster を保存/読込（sklearn ラッパは fit のみ使用し、推論は Booster.predict）。
予測は常に負値 0 クリップ（降水は非負）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import lightgbm as lgb
import numpy as np

PathLike = Union[str, Path]

# 有雨回帰を学習するための最小有雨画素数（極端に少ない場合は回帰を作らない）。
_MIN_RAIN_PIXELS: int = 100


# --- two_part（hurdle） ------------------------------------------------------


def fit_two_part(
    X: np.ndarray,
    y: np.ndarray,
    rain_threshold: float,
    params_clf: dict,
    params_reg: dict,
    seed: int,
) -> tuple[lgb.Booster, "lgb.Booster | None"]:
    """分類器（P(y>=thr)）と有雨 log1p 回帰を学習し、Booster 組を返す。

    有雨画素が極端に少ない場合は回帰を None とし、予測は 0 になる。
    """
    y = np.asarray(y, dtype=np.float64)
    rain = y >= rain_threshold

    clf = lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=-1, **params_clf)
    clf.fit(X, rain.astype(np.int8))
    clf_booster = clf.booster_

    reg_booster: lgb.Booster | None = None
    if int(rain.sum()) >= _MIN_RAIN_PIXELS:
        reg = lgb.LGBMRegressor(objective="regression", random_state=seed, n_jobs=-1, **params_reg)
        reg.fit(X[rain], np.log1p(y[rain]))
        reg_booster = reg.booster_
    return clf_booster, reg_booster


def predict_two_part(
    clf_booster: lgb.Booster, reg_booster: "lgb.Booster | None", X: np.ndarray
) -> np.ndarray:
    """pred = P(rain) · expm1(reg)。負値 0 クリップ。回帰が無ければ 0。"""
    p_rain = np.asarray(clf_booster.predict(X), dtype=np.float64)  # binary → P(positive)
    if reg_booster is None:
        return np.zeros(X.shape[0], dtype=np.float64)
    intensity = np.expm1(np.asarray(reg_booster.predict(X), dtype=np.float64))
    intensity = np.clip(intensity, 0.0, None)
    return np.clip(p_rain * intensity, 0.0, None)


# --- tweedie -----------------------------------------------------------------


def fit_tweedie(
    X: np.ndarray, y: np.ndarray, variance_power: float, params: dict, seed: int
) -> lgb.Booster:
    """objective='tweedie' の単一回帰で生 mm/hr を学習し Booster を返す。"""
    m = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=variance_power,
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    m.fit(X, np.asarray(y, dtype=np.float64))
    return m.booster_


def predict_tweedie(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """tweedie 予測（負値 0 クリップ）。"""
    return np.clip(np.asarray(booster.predict(X), dtype=np.float64), 0.0, None)


# --- 保存 / 読込 -------------------------------------------------------------


def save_booster(booster: lgb.Booster, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load_booster(path: PathLike) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
