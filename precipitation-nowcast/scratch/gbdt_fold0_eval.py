"""GBDT tweedie を fold0 検証で評価し CNN holdout と公平比較する診断。"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from precip import config, features, gbdt, metrics

cfg = config.load_yaml_config(config.DEFAULT_CONFIG_YAML)
g = cfg.get("gbdt", {})
names = features.feature_names()
vpower = float(g.get("tweedie_variance_power", 1.5))
params = dict(g.get("lgb_regressor", {}) or {})
seed = int(g.get("seed", 42))

ys, ps, per = [], [], {}
for sat in ("himawari","goes","meteosat"):
    p = config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet"
    if not p.exists():
        print("skip", sat); continue
    df = pd.read_parquet(p)
    X = df[names].to_numpy(np.float32); y = df["y"].to_numpy(np.float64); fold = df["fold"].to_numpy(int)
    tr = fold!=0; va = fold==0
    if va.sum()==0 or tr.sum()==0:
        print(sat,"no fold0"); continue
    b = gbdt.fit_tweedie(X[tr], y[tr], vpower, params, seed)
    pred = gbdt.predict_tweedie(b, X[va])
    per[sat] = metrics.rmse(y[va], pred)
    ys.append(y[va]); ps.append(pred)
    print(f"{sat}: fold0 val n={va.sum():,} tweedie RMSE={per[sat]:.4f}")
    del X,y,fold,df
Y=np.concatenate(ys); P=np.concatenate(ps)
cond5 = metrics.rmse(Y[Y>=5],P[Y>=5]) if (Y>=5).any() else None
print(f"\nGBDT fold0 overall_RMSE={metrics.rmse(Y,P):.4f} per_sat={ {k:round(v,4) for k,v in per.items()} } cond>=5={cond5 and round(cond5,3)} bias={float((P-Y).mean()):+.4f}")
