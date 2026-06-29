"""正則化中心の lgbm(tweedie) HP探索。既存parquet(97特徴)を再利用・前処理不要。

仮説: CV(1.16)≫LB(0.707) の大乖離は訓練地域への過適合。正則化を強めて 5-fold OOF
(=地域汎化の代理) が下がれば過適合仮説が当たりで LB にも transfer する見込み。
評価は overall と cond>=5（強雨の弱点）。相対比較なので 60万行に間引いて高速化。
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from precip import config, features, gbdt, metrics, cv as cvmod

N_SUB = 600_000
seed = 42
names = features.feature_names()  # 97（scene撤去済）
loc2f = cvmod.load_handdesigned_folds()
nsp = max(loc2f.values()) + 1

# 共通ベース。各configはこれを上書き。
BASE = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            force_col_wise=True, max_bin=127)

CONFIGS = {
    "baseline":        (dict(BASE), 1.5),
    "reg_mild":        (dict(BASE, num_leaves=47, min_child_samples=300, reg_lambda=2.0), 1.5),
    "reg_strong":      (dict(BASE, num_leaves=31, min_child_samples=500, reg_lambda=5.0,
                             reg_alpha=1.0, colsample_bytree=0.6, subsample=0.7), 1.5),
    "reg_strong_long": (dict(BASE, num_leaves=31, min_child_samples=500, reg_lambda=5.0,
                             reg_alpha=1.0, colsample_bytree=0.6, subsample=0.7, n_estimators=1000), 1.5),
    "power_1.3":       (dict(BASE), 1.3),
    "power_1.7":       (dict(BASE), 1.7),
}

# データを一度だけ読んで間引きキャッシュ（メモリ節約）。
cache = {}
for sat in ("himawari", "goes", "meteosat"):
    df = pd.read_parquet(config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet")
    if len(df) > N_SUB:
        df = df.sample(n=N_SUB, random_state=seed)
    cache[sat] = (df[names].to_numpy(np.float32), df["y"].to_numpy(np.float64), df["fold"].to_numpy(int))
    print(f"loaded {sat}: {cache[sat][0].shape}", flush=True)
    del df


def evaluate(params, power):
    ys, ps = [], []
    for sat in ("himawari", "goes", "meteosat"):
        X, y, fold = cache[sat]
        o = np.full(len(y), np.nan)
        for f in range(nsp):
            tr = fold != f
            va = fold == f
            if va.sum() == 0 or tr.sum() == 0:
                continue
            b = gbdt.fit_tweedie(X[tr], y[tr], power, params, seed)
            o[va] = gbdt.predict_tweedie(b, X[va])
        m = ~np.isnan(o)
        ys.append(y[m])
        ps.append(o[m])
    Y = np.concatenate(ys)
    P = np.concatenate(ps)
    c5 = metrics.rmse(Y[Y >= 5], P[Y >= 5]) if (Y >= 5).any() else float("nan")
    return metrics.rmse(Y, P), c5


print(f"\n{'config':18s} {'overall':>9s} {'cond>=5':>9s}  (N_SUB={N_SUB})", flush=True)
results = {}
base_ov = None
for name, (params, power) in CONFIGS.items():
    ov, c5 = evaluate(params, power)
    results[name] = (ov, c5)
    if name == "baseline":
        base_ov = ov
    delta = f"  Δ={ov-base_ov:+.4f}" if base_ov is not None else ""
    print(f"{name:18s} {ov:9.4f} {c5:9.3f}{delta}", flush=True)

best = min(results, key=lambda k: results[k][0])
print(f"\nBEST: {best}  overall={results[best][0]:.4f} (baseline比 {results[best][0]-base_ov:+.4f})", flush=True)
