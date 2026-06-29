from __future__ import annotations

import numpy as np
import pandas as pd


def save_oof(
    oof_preds: dict[str, np.ndarray],
    train_df: pd.DataFrame,
    result_dir: str,
    filename: str = 'oof_predictions.csv',
) -> pd.DataFrame:
    """
    OOF（Out-of-Fold）予測を元のテーブルにマージして CSV に保存する。

    Parameters
    ----------
    oof_preds : {'lgb': array, 'cb': array, 'ensemble': array} 形式の辞書
    train_df  : 学習用 DataFrame（元テーブルをそのままマージ）
    result_dir : 保存先ディレクトリ
    filename  : 保存するファイル名

    Returns
    -------
    保存した DataFrame
    """
    df = train_df.reset_index(drop=True).copy()
    for name, preds in oof_preds.items():
        df[f'oof_{name}'] = np.clip(preds, 0, None)
    path = f'{result_dir}/{filename}'
    df.to_csv(path, index=False)
    print(f'Saved -> {path}')
    return df


def save_submission(
    preds: np.ndarray,
    filename: str,
    eval_df: pd.DataFrame,
    result_dir: str,
) -> pd.DataFrame:
    """
    予測値を提出フォーマットで CSV に保存する。

    Parameters
    ----------
    preds : 予測値の配列
    filename : 保存するファイル名（例: 'submission_lgb.csv'）
    eval_df : 評価用 DataFrame（data_id 列を使用）
    result_dir : 保存先ディレクトリ

    Returns
    -------
    保存した DataFrame
    """
    sub = pd.DataFrame({
        'data_id': eval_df['data_id'],
        'construction_cost_per_m2_usd': np.clip(preds, 0, None),
    })
    path = f'{result_dir}/{filename}'
    sub.to_csv(path, index=False)
    print(f'Saved -> {path}')
    return sub
