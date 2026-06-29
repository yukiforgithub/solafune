"""軽量アブレーション: 同一データで scene特徴 有/無 の純効果を測る（低メモリ・unbuffered）。"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from precip import config, features, gbdt, metrics, cv as cvmod

cfg = config.load_yaml_config(config.DEFAULT_CONFIG_YAML)
g = cfg["gbdt"]
vpow = float(g["tweedie_variance_power"])
params = dict(g["lgb_regressor"])
seed = int(g["seed"])
N_SUB = 300_000  # 各衛星 間引き（相対比較には十分・低メモリ・高速）

names = features.feature_names()
scene = [n for n in names if "scene" in n]
names97 = [n for n in names if n not in scene]
loc2f = cvmod.load_handdesigned_folds()
nsp = max(loc2f.values()) + 1
col_idx = {n: i for i, n in enumerate(names)}
idx97 = [col_idx[n] for n in names97]

cache = {}
for sat in ("himawari", "goes", "meteosat"):
    df = pd.read_parquet(config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet")
    if len(df) > N_SUB:
        df = df.sample(n=N_SUB, random_state=seed)
    cache[sat] = (df[names].to_numpy(np.float32), df["y"].to_numpy(np.float64), df["fold"].to_numpy(int))
    print(f"loaded {sat}: {cache[sat][0].shape}", flush=True)
    del df


def oof(use_all):
    ys, ps = [], []
    for sat in ("himawari", "goes", "meteosat"):
        Xfull, y, fold = cache[sat]
        X = Xfull if use_all else Xfull[:, idx97]
        o = np.full(len(y), np.nan)
        for f in range(nsp):
            tr = fold != f
            va = fold == f
            if va.sum() == 0 or tr.sum() == 0:
                continue
            b = gbdt.fit_tweedie(X[tr], y[tr], vpow, params, seed)
            o[va] = gbdt.predict_tweedie(b, X[va])
        m = ~np.isnan(o)
        ys.append(y[m])
        ps.append(o[m])
    Y = np.concatenate(ys)
    P = np.concatenate(ps)
    return metrics.rmse(Y, P), metrics.rmse(Y[Y >= 5], P[Y >= 5])


print("running 97 (scene除外)...", flush=True)
r97 = oof(False)
print(f"  97:  overall={r97[0]:.4f} cond5={r97[1]:.3f}", flush=True)
print("running 105 (scene込み)...", flush=True)
r105 = oof(True)
print(f"  105: overall={r105[0]:.4f} cond5={r105[1]:.3f}", flush=True)
print(f"\nSCENE純効果(同一データ): overall {r105[0]-r97[0]:+.4f}  cond5 {r105[1]-r97[1]:+.3f}", flush=True)
