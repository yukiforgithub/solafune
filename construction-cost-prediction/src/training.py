from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from catboost import CatBoostRegressor


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error"""
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))


def run_cv(
    model_fn,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame | None,
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    K-Fold クロスバリデーションを実行する。

    Parameters
    ----------
    model_fn : callable
        (X_tr, y_tr, X_va, y_va) -> fitted model with .predict()
    X_eval : 提出用データ。None の場合は eval 予測をスキップする。

    Returns
    -------
    oof_preds, eval_preds (X_eval=None の場合は None), oof_score
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds  = np.zeros(len(X_train))
    eval_preds = np.zeros(len(X_eval)) if X_eval is not None else None
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx]
        y_va = y_train.iloc[va_idx]

        model = model_fn(X_tr, y_tr, X_va, y_va)

        oof_preds[va_idx] = model.predict(X_va)
        if eval_preds is not None:
            eval_preds += model.predict(X_eval) / n_splits

        score = rmsle(y_va, oof_preds[va_idx])
        fold_scores.append(score)
        print(f'  Fold {fold + 1} RMSLE: {score:.4f}')

    oof_score = rmsle(y_train, oof_preds)
    print(f'  OOF RMSLE: {oof_score:.4f} | Mean Fold: {np.mean(fold_scores):.4f}')
    return oof_preds, eval_preds, oof_score


def make_lgb_fn(params: dict, cat_cols: list[str]):
    """
    LightGBM 用 model_fn を返す。

    Parameters
    ----------
    params : lgb.train に渡すパラメータ辞書
    cat_cols : カテゴリカル列名のリスト
    """
    class _LGBWrapper:
        """predict 時にも category 変換を行う lgb.Booster ラッパー。"""
        def __init__(self, booster, cat_features):
            self.booster = booster
            self.cat_features = cat_features

        def predict(self, X):
            X = X.copy()
            for c in self.cat_features:
                if c in X.columns:
                    X[c] = X[c].astype('category')
            return self.booster.predict(X)

    def _train(X_tr, y_tr, X_va, y_va):
        cat_features = [c for c in cat_cols if c in X_tr.columns]
        X_tr_lgb = X_tr.copy()
        X_va_lgb = X_va.copy()
        for c in cat_features:
            X_tr_lgb[c] = X_tr_lgb[c].astype('category')
            X_va_lgb[c] = X_va_lgb[c].astype('category')
        dtrain = lgb.Dataset(X_tr_lgb, label=y_tr, categorical_feature=cat_features)
        dvalid = lgb.Dataset(X_va_lgb, label=y_va, reference=dtrain)
        booster = lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        return _LGBWrapper(booster, cat_features)

    return _train


def make_cb_fn(params: dict, cat_cols: list[str]):
    """
    CatBoost 用 model_fn を返す。

    Parameters
    ----------
    params : CatBoostRegressor に渡すパラメータ辞書
    cat_cols : カテゴリカル列名のリスト
    """
    def _train(X_tr, y_tr, X_va, y_va):
        cat_features = [c for c in cat_cols if c in X_tr.columns]
        model = CatBoostRegressor(**params, cat_features=cat_features)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        return model

    return _train
