"""HP探索 第2ラウンド: reg_strong × tweedie_power の組合せ最適化（前処理不要）。

R1の発見: 正則化(reg_strong, cond5にも効く) と tweedie_power(1.7が最良, 単調) が
ほぼ直交。組合せて積み上がるか、powerの最適点(1.7~1.9)を詰める。
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from precip import config, features, gbdt, metrics, cv as cvmod

N_SUB = 400_000
seed = 42
# 既存parquetに実在する特徴のみ使用（①物理特徴は未生成なので除外＝97 temporal特徴）。
import pyarrow.parquet as pq
_pcols = set(pq.ParquetFile(str(config.OUTPUTS_DIR / "phase2_features_himawari.parquet")).schema.names)
names = [n for n in features.feature_names() if n in _pcols]
print(f"using {len(names)} features (parquet実在分)", flush=True)
loc2f = cvmod.load_handdesigned_folds()
nsp = max(loc2f.values()) + 1

BASE = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, force_col_wise=True, max_bin=127)
REG_STRONG = dict(BASE, num_leaves=31, min_child_samples=500, reg_lambda=5.0,
                  reg_alpha=1.0, colsample_bytree=0.6, subsample=0.7)
REG_XSTRONG = dict(BASE, num_leaves=21, min_child_samples=800, reg_lambda=10.0,
                   reg_alpha=2.0, colsample_bytree=0.5, subsample=0.7)
REG_STRONG_LONG = dict(REG_STRONG, n_estimators=1000)

CONFIGS = {
    "baseline":       (dict(BASE), 1.5),
    "regS_p1.7":      (dict(REG_STRONG), 1.7),
    "regS_p1.8":      (dict(REG_STRONG), 1.8),
    "regS_p1.9":      (dict(REG_STRONG), 1.9),
    "regXstrong_p1.8": (dict(REG_XSTRONG), 1.8),
}

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
    c5 = metrics.rmse(Y[Y >= 5], P[Y >= 5])
    return metrics.rmse(Y, P), c5


print(f"\n{'config':18s} {'overall':>9s} {'cond>=5':>9s}", flush=True)
results = {}
base_ov = None
for name, (params, power) in CONFIGS.items():
    ov, c5 = evaluate(params, power)
    results[name] = (ov, c5)
    if name == "baseline":
        base_ov = ov
    print(f"{name:18s} {ov:9.4f} {c5:9.3f}  Δ={ov-base_ov:+.4f}", flush=True)

best = min(results, key=lambda k: results[k][0])
print(f"\nBEST: {best}  overall={results[best][0]:.4f} cond5={results[best][1]:.3f} (baseline比 {results[best][0]-base_ov:+.4f})", flush=True)
