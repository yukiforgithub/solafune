"""Kaggle GPU 用 U-Net 学習 + 提出生成（自己完結, コードセルに貼るだけ）。

入力データ: Kaggle Private Dataset `precip-nowcast-tensors`（前処理済みテンソル）
  /kaggle/input/precip-nowcast-tensors/
    train_{sat}_X.npy (N,51,41,41)f16 / train_{sat}_y.npy (N,41,41)f16 / train_{sat}_meta.npz(fold)
    eval_{sat}_X.npy  (M,51,41,41)f16 / eval_{sat}_names.json / eval_manifest.parquet
    evaluation_target.csv / meta.json

方針（CPUで得た知見を注入）:
  - 衛星別 U-Net（バンドの物理的意味が衛星で異なるため）
  - 入力 51ch = 3フレーム×16band(/255) + presence3面, 41×41（pad48でU-Net, 出力crop41）
  - log1p(y) で学習 → expm1 → 0クリップ（LH24実証）
  - 地域 GroupKFold（未知地域汎化の検証＝LBの代理）。CV_MODE='single'でfold0検証→良ければ'full'
  - データ拡張: h/v flip + 90°回転（IR/WV主役なので向き不変が効く）
  - 混合精度(AMP)で高速化

使い方:
  1) Settings → Accelerator → GPU, Internet ON
  2) このファイル全体をコードセルに貼って実行
  3) CV_MODE='single' で fold0 val RMSE を GBDT(≈1.15)と比較 → 良ければ 'full' に変えて再実行
  4) /kaggle/working/submission.zip をダウンロード → Solafune提出
"""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================ 設定 ============================
DATA = Path("/kaggle/input/precip-nowcast-tensors")
WORK = Path("/kaggle/working")
SATS = ("himawari", "goes", "meteosat")
CV_MODE = "single"      # 'single'(fold0検証で高速) | 'full'(5fold OOF + 全データ最終fit)
SUBMIT_NAME = "phase3_cnn_unet"  # 提出zip名（実験を識別。例: phase3_cnn_unet_e40, _smp 等に変えて区別）
HOLDOUT_FOLD = 0
EPOCHS = 40
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CH = 32            # U-Net 基底チャネル(32→64→128)
SEED = 42
FALLBACK = 0.2886       # 0フレーム等の気候値
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch:", torch.__version__)

torch.manual_seed(SEED)
np.random.seed(SEED)

meta = json.loads((DATA / "meta.json").read_text())
C = int(meta["channels"])           # 51
H = W = int(meta["target_size"][0]) # 41
N_FOLDS = int(meta["n_folds"])
print("channels:", C, "target:", H, "folds:", N_FOLDS)


# ============================ U-Net ============================
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """41×41 を pad48 で扱う小型 U-Net（2階層 down/up, skip付き）。出力 1ch=log1p(降水)。"""

    def __init__(self, cin, base=32):
        super().__init__()
        self.inc = DoubleConv(cin, base)            # 48
        self.pool = nn.MaxPool2d(2)
        self.d1 = DoubleConv(base, base * 2)        # 24
        self.d2 = DoubleConv(base * 2, base * 4)    # 12
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 12->24
        self.c2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)      # 24->48
        self.c1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # 41 -> 48（reflect pad で端の人工境界を抑える）
        x = F.pad(x, (3, 4, 3, 4), mode="reflect")  # 41+7=48
        x0 = self.inc(x)            # 48
        x1 = self.d1(self.pool(x0))  # 24
        x2 = self.d2(self.pool(x1))  # 12
        u2 = self.up2(x2)            # 24
        u2 = self.c2(torch.cat([u2, x1], 1))
        u1 = self.up1(u2)            # 48
        u1 = self.c1(torch.cat([u1, x0], 1))
        out = self.outc(u1)         # (B,1,48,48)
        return out[:, :, 3:3 + H, 3:3 + W]  # crop 41×41


# ============================ Dataset ============================
class TensorDS(Dataset):
    def __init__(self, X, y, idx, augment=False):
        self.X = X
        self.y = y
        self.idx = idx
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        x = np.asarray(self.X[i], dtype=np.float32)         # (C,41,41)
        yt = np.log1p(np.asarray(self.y[i], dtype=np.float32))[None]  # (1,41,41)
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy(); yt = yt[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy(); yt = yt[:, ::-1, :].copy()
            r = np.random.randint(4)
            if r:
                x = np.rot90(x, r, axes=(1, 2)).copy(); yt = np.rot90(yt, r, axes=(1, 2)).copy()
        return torch.from_numpy(x), torch.from_numpy(yt)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cond_rmse(y, p, thr):
    m = y >= thr
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else float("nan")


# ============================ 学習・予測 ============================
def predict_idx(model, X, idx, batch=256):
    """idx の生 mm/hr 予測 (len(idx),41,41) を返す（expm1, 0クリップ）。"""
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sl = idx[s:s + batch]
            xb = torch.from_numpy(np.asarray(X[sl], dtype=np.float32)).to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                pl = model(xb).float().cpu().numpy()[:, 0]
            outs.append(np.clip(np.expm1(pl), 0, None).astype(np.float32))
    return np.concatenate(outs) if outs else np.zeros((0, H, W), np.float32)


def train_one(X, y, tr_idx, va_idx, epochs=EPOCHS):
    model = UNet(C, BASE_CH).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDS(X, y, tr_idx, augment=True), batch_size=BATCH, shuffle=True,
                    num_workers=2, pin_memory=(DEVICE == "cuda"), drop_last=True)
    for ep in range(epochs):
        model.train()
        run = 0.0
        nb = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * xb.size(0); nb += xb.size(0)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            if len(va_idx):
                p = predict_idx(model, X, va_idx)
                yv = np.asarray(y[va_idx], np.float32)
                print(f"    ep{ep+1}/{epochs} train_logmse={run/max(nb,1):.4f} val_RMSE={rmse(yv,p):.4f}", flush=True)
            else:
                print(f"    ep{ep+1}/{epochs} train_logmse={run/max(nb,1):.4f}", flush=True)
    return model


def run():
    models = {}
    all_y, all_p = [], []
    per_sat = {}
    for sat in SATS:
        Xp = DATA / f"train_{sat}_X.npy"
        if not Xp.exists():
            print("skip", sat); continue
        X = np.load(Xp, mmap_mode="r")
        y = np.load(DATA / f"train_{sat}_y.npy", mmap_mode="r")
        fold = np.load(DATA / f"train_{sat}_meta.npz")["fold"]
        valid = np.where(fold >= 0)[0]
        print(f"[{sat}] {len(valid):,} 有効サンプル, CV_MODE={CV_MODE}")
        if CV_MODE == "full":
            ys, ps = [], []
            for f in range(N_FOLDS):
                tr = valid[fold[valid] != f]; va = valid[fold[valid] == f]
                if len(va) == 0 or len(tr) == 0:
                    continue
                print(f"  fold{f}: train{len(tr):,}/val{len(va):,}")
                m = train_one(X, y, tr, va)
                ys.append(np.asarray(y[va], np.float32)); ps.append(predict_idx(m, X, va))
            yv = np.concatenate(ys); pv = np.concatenate(ps)
            per_sat[sat] = rmse(yv, pv); all_y.append(yv.reshape(len(yv), -1)); all_p.append(pv.reshape(len(pv), -1))
            print(f"  [{sat}] 最終fit on all {len(valid):,}")
            models[sat] = train_one(X, y, valid, np.array([], int))
        else:
            tr = valid[fold[valid] != HOLDOUT_FOLD]; va = valid[fold[valid] == HOLDOUT_FOLD]
            print(f"  holdout fold{HOLDOUT_FOLD}: train{len(tr):,}/val{len(va):,}")
            m = train_one(X, y, tr, va)
            yv = np.asarray(y[va], np.float32); pv = predict_idx(m, X, va)
            per_sat[sat] = rmse(yv, pv); all_y.append(yv.reshape(len(yv), -1)); all_p.append(pv.reshape(len(pv), -1))
            models[sat] = m  # 80%学習モデル（提出可）
    Y = np.concatenate(all_y).ravel(); P = np.concatenate(all_p).ravel()
    print("\n==== 評価 ====")
    print(f"overall_RMSE={rmse(Y,P):.4f} per_sat={ {k:round(v,4) for k,v in per_sat.items()} }")
    print(f"cond>=1={cond_rmse(Y,P,1):.3f} cond>=5={cond_rmse(Y,P,5):.3f}")
    print("（比較: GBDT temporal CV≈1.16 / 新HP≈1.15。これを下回れば有望）")
    return models


def make_submission(models):
    import shutil
    # 生tifは一時領域(/kaggle/temp)に書き、/kaggle/working には .zip だけ残す（29kファイルでUIが固まるのを防ぐ）
    test_dir = Path("/kaggle/temp") / SUBMIT_NAME / "test_files"
    if test_dir.parent.exists():
        shutil.rmtree(test_dir.parent, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    import rasterio
    from rasterio.transform import Affine
    profile = dict(driver="GTiff", dtype="float32", nodata=None, width=W, height=H, count=1,
                   crs=None, transform=Affine.identity())

    def write_tif(name, arr):
        with rasterio.open(test_dir / name, "w", **profile) as ds:
            ds.write(np.asarray(arr, np.float32), 1)

    manifest = pd.read_parquet(DATA / "eval_manifest.parquet")
    done = 0
    for sat in SATS:
        Xp = DATA / f"eval_{sat}_X.npy"
        if not Xp.exists() or sat not in models:
            continue
        names = json.loads((DATA / f"eval_{sat}_names.json").read_text())
        X = np.load(Xp, mmap_mode="r")
        pred = predict_idx(models[sat], X, np.arange(len(names)))
        for i, nm in enumerate(names):
            write_tif(nm, pred[i]); done += 1
    # fallback（無効行）= 気候値
    fb = manifest[~manifest["valid"]]
    tile = np.full((H, W), FALLBACK, np.float32)
    for nm in fb["gpm_imerg_filename"]:
        write_tif(nm, tile); done += 1
    print(f"書き出し tif: {done}")
    # zip（中身は規定通り evaluation_target.csv + test_files/。外側の名前は SUBMIT_NAME で識別）
    zpath = WORK / f"{SUBMIT_NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(DATA / "evaluation_target.csv", "evaluation_target.csv")
        for tif in sorted(test_dir.iterdir()):
            zf.write(tif, f"test_files/{tif.name}")
    shutil.rmtree(test_dir.parent, ignore_errors=True)   # 一時tifを掃除（workingは.zipのみ）
    print(f"提出zip: {zpath}  (test_filesは/kaggle/tempに書いて削除済)")


# ============================ 実行 ============================
models = run()
if CV_MODE == "full":
    make_submission(models)   # full のときだけ全データ最終モデルで提出生成
else:
    print("\nCV_MODE='single': まず val_RMSE を確認。有望なら CV_MODE='full' で再実行して提出生成。")
    # single でも 80%モデルで暫定提出を作りたい場合は次行を有効化:
    # make_submission(models)
