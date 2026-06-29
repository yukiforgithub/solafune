import sys, json
sys.path.insert(0,"src")
import numpy as np, pandas as pd
from precip import config, features, gbdt, metrics, cv as cvmod
cfg=config.load_yaml_config(config.DEFAULT_CONFIG_YAML); g=cfg["gbdt"]
vpow=float(g["tweedie_variance_power"]); params=dict(g["lgb_regressor"]); seed=int(g["seed"])
names=features.feature_names()
scene=[n for n in names if "scene" in n]
names97=[n for n in names if n not in scene]
loc2f=cvmod.load_handdesigned_folds(); nsp=max(loc2f.values())+1
def oof(cols):
    ys,ps=[],[]
    for sat in ("himawari","goes","meteosat"):
        df=pd.read_parquet(config.OUTPUTS_DIR/f"phase2_features_{sat}.parquet")
        X=df[cols].to_numpy(np.float32); y=df["y"].to_numpy(np.float64); fold=df["fold"].to_numpy(int)
        o=np.full(len(y),np.nan)
        for f in range(nsp):
            tr=fold!=f; va=fold==f
            if va.sum()==0 or tr.sum()==0: continue
            b=gbdt.fit_tweedie(X[tr],y[tr],vpow,params,seed); o[va]=gbdt.predict_tweedie(b,X[va])
        m=~np.isnan(o); ys.append(y[m]); ps.append(o[m])
        del X,y,fold,df
    Y=np.concatenate(ys); P=np.concatenate(ps)
    c5=metrics.rmse(Y[Y>=5],P[Y>=5])
    return metrics.rmse(Y,P), c5
r97=oof(names97); r105=oof(names)
print(f"\nABLATION (同一2.0Mデータ):")
print(f"  97特徴(scene除外): overall={r97[0]:.4f} cond>=5={r97[1]:.3f}")
print(f"  105特徴(scene込み): overall={r105[0]:.4f} cond>=5={r105[1]:.3f}")
print(f"  scene効果(純): overall {r105[0]-r97[0]:+.4f}  cond>=5 {r105[1]-r97[1]:+.3f}")
