"""実験2: 物理特徴の純効果を切り分け（実験1の2.5M/102 parquetを再利用, 確定HPで）。

同一データ・同一HP(reg_strong+power1.7)で 97(物理なし) vs 102(物理あり) を 5-fold OOF 比較。
差が物理特徴(cool分解+btd_phase)の純効果。参考に temporal(旧HP,2.5M) CV=1.1629 も併記。
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from precip import config, features, gbdt, metrics, cv as cvmod

N_SUB = 800_000
seed = 42
cfg = config.load_yaml_config(config.DEFAULT_CONFIG_YAML)
g = cfg["gbdt"]
power = float(g["tweedie_variance_power"])
params = dict(g["lgb_regressor"])
loc2f = cvmod.load_handdesigned_folds()
nsp = max(loc2f.values()) + 1

PHYS = {"win_cool01", "win_cool_accel", "ir38_cool01", "ir38_cool_accel", "btd_phase"}
names102 = features.feature_names()  # 102
# 実在チェック（実験1のparquetに物理列があるはず）。
pcols = set(pq.ParquetFile(str(config.OUTPUTS_DIR / "phase2_features_himawari.parquet")).schema.names)
missing = [n for n in names102 if n not in pcols]
if missing:
    print(f"[exp2] 警告: parquetに無い特徴 {missing} → 実在分のみ使用", flush=True)
names102 = [n for n in names102 if n in pcols]
names97 = [n for n in names102 if n not in PHYS]
print(f"[exp2] 102特徴={len(names102)} / 97特徴(物理除外)={len(names97)} | HP power={power} num_leaves={params.get('num_leaves')}", flush=True)

cache = {}
for sat in ("himawari", "goes", "meteosat"):
    df = pd.read_parquet(config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet")
    if len(df) > N_SUB:
        df = df.sample(n=N_SUB, random_state=seed)
    cache[sat] = (df, )  # DataFrame保持（列選択は評価時）
    print(f"[exp2] loaded {sat}: {len(df):,} 行", flush=True)


def oof(cols):
    ys, ps = [], []
    for sat in ("himawari", "goes", "meteosat"):
        df = cache[sat][0]
        X = df[cols].to_numpy(np.float32)
        y = df["y"].to_numpy(np.float64)
        fold = df["fold"].to_numpy(int)
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
        del X
    Y = np.concatenate(ys)
    P = np.concatenate(ps)
    return metrics.rmse(Y, P), metrics.rmse(Y[Y >= 5], P[Y >= 5])


print("[exp2] running 97 (物理なし, 新HP)...", flush=True)
r97 = oof(names97)
print(f"[exp2]   97 : overall={r97[0]:.4f} cond5={r97[1]:.3f}", flush=True)
print("[exp2] running 102 (物理あり, 新HP)...", flush=True)
r102 = oof(names102)
print(f"[exp2]   102: overall={r102[0]:.4f} cond5={r102[1]:.3f}", flush=True)
print(f"\n[exp2] 物理特徴の純効果(同一データ・同一HP): overall {r102[0]-r97[0]:+.4f}  cond5 {r102[1]-r97[1]:+.3f}", flush=True)
print(f"[exp2] 参考: temporal(旧HP,full2.5M) CV=1.1629。今回はN_SUB={N_SUB}・新HPなので絶対値は別物、相対差で判断。", flush=True)
