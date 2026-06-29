"""Kaggle GPU 用 U-Net v2（安定版）: v1のlog1p + 衛星別正規化 + 6-way TTA + seed平均。

dual head（生mm/hr）は小型41×41モデルでは崩壊した（near-zeroに潰れる）ため不採用。
v1の log1p（裾を圧縮＝安定, 既にLB0.692）を土台に、0.6947解法の「安く転移する」3技術だけ追加:
  1. 衛星別 per-channel 正規化（train memmapから mean/std を計算, z-score。再前処理不要）
  2. 6-way TTA（flip/rot 平均）
  3. seed平均（SEEDS指定）
+ ckpt保存（SUBMIT_ONLYで提出だけ再生成）。

使い方: DATAパスを実際の値に。CV_MODE='single'でfold0検証→v1(fold0 1.1717)と比較→'full'で提出。
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================ 設定 ============================
DATA = Path("/kaggle/input/datasets/yukiforkaggle/precip-nowcast-tensors")  # ←実際のパスに
WORK = Path("/kaggle/working")
SATS = ("himawari", "goes", "meteosat")
CV_MODE = "single"            # 'single'(fold0) | 'full'(5fold OOF + 全データfit + 提出)
SUBMIT_NAME = "phase3_cnn_unet_v2"
SUBMIT_ONLY = False           # True: 学習せず保存ckptから提出zipだけ作る
HOLDOUT_FOLD = 0
EPOCHS = 40
BATCH = 128
LR = 1e-3                     # v1と同じ（AdamW/warmupはやめv1のAdam+cosineに揃える）
WEIGHT_DECAY = 1e-4
BASE_CH = 32
SEEDS = [42]                  # 例 [42,123] で2-seed平均（学習2倍）
USE_TTA = True
FALLBACK = 0.2886
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch:", torch.__version__)
torch.backends.cudnn.benchmark = True

meta = json.loads((DATA / "meta.json").read_text())
C = int(meta["channels"]); H = W = int(meta["target_size"][0]); N_FOLDS = int(meta["n_folds"])
print("channels:", C, "target:", H, "folds:", N_FOLDS)


# ============================ U-Net（単一head, log1p予測） ============================
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    """41→pad48 小型U-Net。出力1ch = log1p(降水)（線形）。"""
    def __init__(self, cin, base=32):
        super().__init__()
        self.inc = DoubleConv(cin, base); self.pool = nn.MaxPool2d(2)
        self.d1 = DoubleConv(base, base * 2); self.d2 = DoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2); self.c2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2); self.c1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x = F.pad(x, (3, 4, 3, 4), mode="reflect")
        x0 = self.inc(x); x1 = self.d1(self.pool(x0)); x2 = self.d2(self.pool(x1))
        u2 = self.c2(torch.cat([self.up2(x2), x1], 1)); u1 = self.c1(torch.cat([self.up1(u2), x0], 1))
        return self.outc(u1)[:, :, 3:3 + H, 3:3 + W]   # (B,1,41,41) = log1p予測


# ============================ 正規化 ============================
# ★地域GroupKFold(未知地域val)では「全地域統計のz-score」が地域シフトを増幅し崩壊させる
# （あの解法のper-sat正規化は"時間split=in-distribution"前提の話）。データは既に/255済なので
# 恒等（z-scoreしない）＝v1の安定した /255 入力に戻す。TTA/seedの上積みだけ採用。
def compute_norm(X, idx, n_sample=2000):
    return np.zeros(C, np.float32), np.ones(C, np.float32)   # 恒等: x=(X-0)/1=X（/255済のまま）


class DS(Dataset):
    def __init__(self, X, y, idx, mean, std, augment=False):
        self.X, self.y, self.idx, self.augment = X, y, idx, augment
        self.mean = mean[:, None, None]; self.std = std[:, None, None]
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = int(self.idx[k])
        x = (np.asarray(self.X[i], np.float32) - self.mean) / self.std
        yt = np.log1p(np.asarray(self.y[i], np.float32))[None]
        if self.augment:
            if np.random.rand() < 0.5: x = x[:, :, ::-1].copy(); yt = yt[:, :, ::-1].copy()
            if np.random.rand() < 0.5: x = x[:, ::-1, :].copy(); yt = yt[:, ::-1, :].copy()
            r = np.random.randint(4)
            if r: x = np.rot90(x, r, (1, 2)).copy(); yt = np.rot90(yt, r, (1, 2)).copy()
        return torch.from_numpy(x), torch.from_numpy(yt)


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def cond_rmse(y, p, t):
    m = y >= t; return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else float("nan")


# ============================ TTA ============================
_TTA = [(0, False, False), (0, True, False), (0, False, True), (1, False, False), (2, False, False), (3, False, False)]
def _fwd(x, k, hf, vf):
    if hf: x = torch.flip(x, [-1])
    if vf: x = torch.flip(x, [-2])
    if k: x = torch.rot90(x, k, [-2, -1])
    return x
def _inv(p, k, hf, vf):
    if k: p = torch.rot90(p, 4 - k, [-2, -1])
    if vf: p = torch.flip(p, [-2])
    if hf: p = torch.flip(p, [-1])
    return p


def predict_idx(models, X, idx, mean, std, batch=256, tta=USE_TTA):
    """models(list)のseed平均 + TTA で生mm/hr予測 (len(idx),41,41)。log1p出力→expm1→0クリップ。"""
    for model in models:
        model.eval()   # ★必須: BatchNormを推論モードに（忘れると学習モードのバッチ統計で予測が壊れる）
    m_ = mean[:, None, None]; s_ = std[:, None, None]
    outs = []
    with torch.no_grad():
        for st in range(0, len(idx), batch):
            sl = idx[st:st + batch]
            xb = torch.from_numpy((np.asarray(X[sl], np.float32) - m_) / s_).to(DEVICE)
            acc = torch.zeros((xb.shape[0], 1, H, W), device=DEVICE)
            tfs = _TTA if tta else [(0, False, False)]
            for model in models:
                for (k, hf, vf) in tfs:
                    with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
                        out = model(_fwd(xb, k, hf, vf)).float()
                    acc += _inv(torch.expm1(out).clamp(0), k, hf, vf)
            acc /= (len(models) * len(tfs))
            outs.append(acc[:, 0].cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, H, W), np.float32)


# ============================ 学習 ============================
def train_one(X, y, tr_idx, va_idx, mean, std, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = UNet(C, BASE_CH).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # v1と同じ
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    lossf = nn.MSELoss()
    dl = DataLoader(DS(X, y, tr_idx, mean, std, augment=True), batch_size=BATCH, shuffle=True,
                    num_workers=2, pin_memory=(DEVICE == "cuda"), drop_last=True)
    for ep in range(epochs):
        model.train(); run = 0.0; nb = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item() * xb.size(0); nb += xb.size(0)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            if len(va_idx):
                p = predict_idx([model], X, va_idx, mean, std)
                print(f"      s{seed} ep{ep+1}/{epochs} logmse={run/max(nb,1):.4f} val_RMSE={rmse(np.asarray(y[va_idx],np.float32),p):.4f}", flush=True)
            else:
                print(f"      s{seed} ep{ep+1}/{epochs} logmse={run/max(nb,1):.4f}", flush=True)
    return model


def train_seeds(X, y, tr, va, mean, std):
    return [train_one(X, y, tr, va, mean, std, sd) for sd in SEEDS]


def run():
    models_by_sat, norm_by_sat = {}, {}
    all_y, all_p, per_sat = [], [], {}
    for sat in SATS:
        Xp = DATA / f"train_{sat}_X.npy"
        if not Xp.exists(): print("skip", sat); continue
        X = np.load(Xp, mmap_mode="r"); y = np.load(DATA / f"train_{sat}_y.npy", mmap_mode="r")
        fold = np.load(DATA / f"train_{sat}_meta.npz")["fold"]; valid = np.where(fold >= 0)[0]
        mean, std = compute_norm(X, valid); norm_by_sat[sat] = (mean, std)
        print(f"[{sat}] {len(valid):,} 有効, CV_MODE={CV_MODE}, seeds={SEEDS}, TTA={USE_TTA}")
        if CV_MODE == "full":
            ys, ps = [], []
            for f in range(N_FOLDS):
                tr = valid[fold[valid] != f]; va = valid[fold[valid] == f]
                if len(va) == 0 or len(tr) == 0: continue
                print(f"  fold{f}: train{len(tr):,}/val{len(va):,}")
                ms = train_seeds(X, y, tr, va, mean, std)
                ys.append(np.asarray(y[va], np.float32)); ps.append(predict_idx(ms, X, va, mean, std))
            yv = np.concatenate(ys); pv = np.concatenate(ps)
            per_sat[sat] = rmse(yv, pv); all_y.append(yv.reshape(len(yv), -1)); all_p.append(pv.reshape(len(pv), -1))
            print(f"  [{sat}] 最終fit on all {len(valid):,}")
            models_by_sat[sat] = train_seeds(X, y, valid, np.array([], int), mean, std)
        else:
            tr = valid[fold[valid] != HOLDOUT_FOLD]; va = valid[fold[valid] == HOLDOUT_FOLD]
            print(f"  holdout fold{HOLDOUT_FOLD}: train{len(tr):,}/val{len(va):,}")
            ms = train_seeds(X, y, tr, va, mean, std)
            yv = np.asarray(y[va], np.float32); pv = predict_idx(ms, X, va, mean, std)
            per_sat[sat] = rmse(yv, pv); all_y.append(yv.reshape(len(yv), -1)); all_p.append(pv.reshape(len(pv), -1))
            models_by_sat[sat] = ms
    Y = np.concatenate(all_y).ravel(); P = np.concatenate(all_p).ravel()
    print("\n==== 評価 ====")
    print(f"overall_RMSE={rmse(Y,P):.4f} per_sat={ {k:round(v,4) for k,v in per_sat.items()} }")
    print(f"cond>=1={cond_rmse(Y,P,1):.3f} cond>=5={cond_rmse(Y,P,5):.3f}")
    print("（比較: v1 CNN fold0 overall1.1717 cond5 7.915 / GBDT新HP fold0 1.1755）")
    ck = WORK / f"{SUBMIT_NAME}_ckpt"; ck.mkdir(parents=True, exist_ok=True)
    for sat, ms in models_by_sat.items():
        for i, m in enumerate(ms): torch.save(m.state_dict(), ck / f"{sat}_seed{i}.pt")
        mean, std = norm_by_sat[sat]; np.savez(ck / f"{sat}_norm.npz", mean=mean, std=std)
    print(f"ckpt保存: {ck}")
    return models_by_sat, norm_by_sat


def load_ckpts():
    ck = WORK / f"{SUBMIT_NAME}_ckpt"; models_by_sat, norm_by_sat = {}, {}
    for sat in SATS:
        ms = []
        for i in range(len(SEEDS)):
            p = ck / f"{sat}_seed{i}.pt"
            if not p.exists(): break
            m = UNet(C, BASE_CH).to(DEVICE); m.load_state_dict(torch.load(p, map_location=DEVICE)); m.eval(); ms.append(m)
        if not ms: continue
        d = np.load(ck / f"{sat}_norm.npz"); models_by_sat[sat] = ms; norm_by_sat[sat] = (d["mean"], d["std"])
    print(f"ckpt読込: {sorted(models_by_sat)}")
    return models_by_sat, norm_by_sat


def make_submission(models_by_sat, norm_by_sat):
    import shutil
    import rasterio
    from rasterio.transform import Affine
    tmp = Path("/kaggle/temp") / SUBMIT_NAME / "test_files"
    if tmp.parent.exists(): shutil.rmtree(tmp.parent, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", dtype="float32", nodata=None, width=W, height=H, count=1, crs=None, transform=Affine.identity())
    def wt(name, arr):
        with rasterio.open(tmp / name, "w", **prof) as ds: ds.write(np.asarray(arr, np.float32), 1)
    manifest = pd.read_parquet(DATA / "eval_manifest.parquet"); done = 0
    for sat in SATS:
        Xp = DATA / f"eval_{sat}_X.npy"
        if not Xp.exists() or sat not in models_by_sat: continue
        names = json.loads((DATA / f"eval_{sat}_names.json").read_text())
        X = np.load(Xp, mmap_mode="r"); mean, std = norm_by_sat[sat]
        pred = predict_idx(models_by_sat[sat], X, np.arange(len(names)), mean, std)
        for i, nm in enumerate(names): wt(nm, pred[i]); done += 1
    tile = np.full((H, W), FALLBACK, np.float32)
    for nm in manifest[~manifest["valid"]]["gpm_imerg_filename"]: wt(nm, tile); done += 1
    print(f"書き出し tif: {done}")
    zpath = WORK / f"{SUBMIT_NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(DATA / "evaluation_target.csv", "evaluation_target.csv")
        for tif in sorted(tmp.iterdir()): zf.write(tif, f"test_files/{tif.name}")
    shutil.rmtree(tmp.parent, ignore_errors=True)
    print(f"提出zip: {zpath}（workingは.zipのみ）")


# ============================ 実行 ============================
if SUBMIT_ONLY:
    models_by_sat, norm_by_sat = load_ckpts()
    make_submission(models_by_sat, norm_by_sat)
else:
    models_by_sat, norm_by_sat = run()
    if CV_MODE == "full":
        make_submission(models_by_sat, norm_by_sat)
    else:
        print("\nsingle: val_RMSEを確認。良ければ CV_MODE='full' で提出生成。")
