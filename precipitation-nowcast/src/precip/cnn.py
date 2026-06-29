"""Phase3 CNN（衛星別・小型FCN, CPU学習）。

GBDT が ~0.707 で頭打ちのため、空間文脈をモデル構造で取り込む小型 FCN を導入する。
較正/ブレンドと違いモデル構造の改善は地域非依存なので未知地域へ transfer する見込み。

設計（RAM 7.7GB / CPU 8コア制約）:
  - 入力: 直近最大3フレーム × 16band を 41x41 へ INTER_AREA 縮約し /255 した
          C=51 ch テンソル（3frame×16band=48 + frame有効 presence 3面）。最新フレームは
          常に t0 スロット（右詰め）。欠測フレームは 0 埋め＋presence=0。
  - モデル: プーリング無しの全層畳み込み（41x41 を維持, dilation で受容野拡大）。
            odd-size の面倒がなく CPU で軽量。出力 1ch = log1p(降水)。
  - 損失: log1p(y) で MSE 学習 → 推論は expm1 → 0クリップ（LH24 実証）。評価は生 RMSE。
  - データ: preprocess が衛星別 float16 memmap (N,C,41,41) を作り、学習は mmap で
            RAM を食わずに読む。
  - CV: 地域 GroupKFold。cv_mode=holdout（fold0検証で高速に競争力を見る）/ full（5fold OOF）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config, features
from .dataio import read_satellite, read_target
from .metrics import rmse

# 入力チャネル: 3 フレーム × 16 band + 3 presence 面。
N_FRAMES_CNN: int = 3
CNN_CHANNELS: int = N_FRAMES_CNN * config.N_INPUT_BANDS + N_FRAMES_CNN  # = 51
TARGET_H, TARGET_W = config.TARGET_SIZE


# --- 入力テンソル構築（学習/予測で共通） -------------------------------------


def build_cnn_input(satellite: str, frame_list: list[str], *, train: bool) -> np.ndarray | None:
    """1 行の CNN 入力 (C,41,41) float32（[0,1]）を返す。0フレーム/最新失敗は None。

    直近最大3フレームを右詰め（最新=t0スロット）で配置。各フレーム16band を 41x41 へ
    INTER_AREA 縮約し /255。欠測スロットは 0、presence 面でフレーム有効を示す。
    """
    if not frame_list:
        return None
    subdir = (config.TRAIN_DIR if train else config.EVAL_DIR) / config.SATELLITE_DIRNAMES[satellite]
    use = frame_list[-N_FRAMES_CNN:]  # 古→新, 最大3
    n_use = len(use)

    X = np.zeros((CNN_CHANNELS, TARGET_H, TARGET_W), dtype=np.float32)
    latest_ok = False
    nb = config.N_INPUT_BANDS
    for i, fname in enumerate(use):
        slot = N_FRAMES_CNN - n_use + i  # 0..2（右詰め: 最後の要素=t0=slot2）
        is_latest = i == n_use - 1
        try:
            arr = read_satellite(subdir / fname)  # (16,H,W) uint8
        except Exception:
            if is_latest:
                return None
            continue
        if arr.shape[0] != nb:
            if is_latest:
                return None
            continue
        stack = features.resize_stack(arr).astype(np.float32) / 255.0  # (16,41,41)
        X[slot * nb:(slot + 1) * nb] = stack
        X[N_FRAMES_CNN * nb + slot] = 1.0  # presence 面
        if is_latest:
            latest_ok = True
    if not latest_ok:
        return None
    return X


# --- モデル（小型 FCN） ------------------------------------------------------


def build_model():
    """プーリング無しの小型 FCN を返す（41x41 を維持, dilation で受容野拡大）。"""
    import torch.nn as nn

    def block(cin, cout, dilation):
        pad = dilation
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(CNN_CHANNELS, 48, 1),
        block(48, 64, 1),
        block(64, 64, 2),   # 受容野拡大
        block(64, 64, 4),
        block(64, 32, 1),
        nn.Conv2d(32, 1, 1),  # 出力 = log1p(降水)
    )


# --- memmap データセット -----------------------------------------------------


def _paths(out_dir: Path, sat: str) -> tuple[Path, Path, Path]:
    base = out_dir / "phase3_cnn"
    return base / f"{sat}_X.npy", base / f"{sat}_y.npy", base / f"{sat}_meta.npz"


class _MemmapDataset:
    """memmap (N,C,41,41) f16 / (N,41,41) f16 から指定 index の (x, log1p(y)) を返す。"""

    def __init__(self, X, y, idx):
        self.X = X
        self.y = y
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        import torch

        i = int(self.idx[k])
        x = torch.from_numpy(np.asarray(self.X[i], dtype=np.float32))
        yt = torch.from_numpy(np.log1p(np.asarray(self.y[i], dtype=np.float32)))[None]  # (1,41,41)
        return x, yt


# --- 学習・評価ループ --------------------------------------------------------


def _train_loop(tr_idx, va_idx, X, y, ccfg, seed):
    """1 モデルを学習し (model, val_oof_pred[N行に対応する dict ではなく va 順の生mm/hr]) を返す。"""
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = int(ccfg.get("epochs", 25))
    batch = int(ccfg.get("batch_size", 64))
    lr = float(ccfg.get("lr", 1e-3))
    wd = float(ccfg.get("weight_decay", 1e-4))
    nthreads = int(ccfg.get("num_threads", 8))
    log_every = int(ccfg.get("log_every", 5))
    torch.set_num_threads(nthreads)

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.MSELoss()

    tr_ds = _MemmapDataset(X, y, tr_idx)
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)

    for ep in range(epochs):
        model.train()
        running = 0.0
        nb = 0
        for xb, yb in tr_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            vr = _eval_rmse(model, va_idx, X, y, batch) if len(va_idx) else float("nan")
            print(f"      epoch {ep+1}/{epochs} train_logmse={running/max(nb,1):.4f} val_rawRMSE={vr:.4f}", flush=True)

    # val の生 mm/hr 予測を返す（OOF 集計用）。
    va_pred = _predict_idx(model, va_idx, X, batch) if len(va_idx) else np.zeros(0, np.float32)
    return model, va_pred


def _predict_idx(model, idx, X, batch) -> np.ndarray:
    """指定 index の生 mm/hr 予測を (len(idx)*1681,) で返す（expm1, 0クリップ）。"""
    import torch

    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sl = idx[s:s + batch]
            xb = torch.from_numpy(np.asarray(X[sl], dtype=np.float32))
            pred_log = model(xb).numpy()[:, 0]  # (b,41,41)
            outs.append(np.clip(np.expm1(pred_log), 0.0, None).astype(np.float32))
    return np.concatenate(outs).ravel() if outs else np.zeros(0, np.float32)


def _eval_rmse(model, idx, X, y, batch) -> float:
    p = _predict_idx(model, idx, X, batch)
    yt = np.asarray(y[idx], dtype=np.float32).ravel()
    return rmse(yt, p)


def _cond_rmse(y, p, thr):
    m = y >= thr
    return float(rmse(y[m], p[m])) if np.any(m) else None


def _save_model(model, path: Path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_model(path: Path):
    import torch

    model = build_model()
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model


# --- 前処理: 衛星別 float16 memmap 構築 --------------------------------------


def build_memmaps(df_full, cfg, out_dir: Path, *, limit: int | None = None) -> dict:
    """衛星別に CNN 入力 (N,C,41,41) f16 / ターゲット (N,41,41) f16 / fold を memmap 保存する。

    RAM を食わないよう open_memmap で逐次ディスク書き込み。最新フレーム失敗 / ターゲット
    欠落の行は fold=-1（学習から除外）。
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy.lib.format as npf

    from . import cv as cvmod

    ccfg = cfg.get("cnn", {}) or {}
    seed = int(ccfg.get("seed", 42))
    n_workers = int(ccfg.get("n_workers", 8))
    loc_to_fold = cvmod.load_handdesigned_folds()
    work = df_full.assign(fold_=df_full["name_location"].map(loc_to_fold))
    work = work[work["fold_"].notna()].copy()
    work["fold_"] = work["fold_"].astype(int)

    base = out_dir / "phase3_cnn"
    base.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for sat in ("himawari", "goes", "meteosat"):
        sub = work[work["satellite_target"] == sat]
        if limit is not None:
            sub = sub.sample(n=min(limit, len(sub)), random_state=seed)
        N = int(len(sub))
        if N == 0:
            print(f"[preprocess:cnn] {sat}: 対象行なし、スキップ。")
            continue
        Xp, yp, mp = _paths(out_dir, sat)
        X = npf.open_memmap(str(Xp), mode="w+", dtype=np.float16,
                            shape=(N, CNN_CHANNELS, TARGET_H, TARGET_W))
        Y = npf.open_memmap(str(yp), mode="w+", dtype=np.float16, shape=(N, TARGET_H, TARGET_W))
        fold = np.full(N, -1, np.int16)
        rows = [(i, r.frame_list, int(r.fold_), r.gpm_imerg_filename)
                for i, r in enumerate(sub.itertuples(index=False))]

        def _w(t):
            i, fl, fd, gpm = t
            xi = build_cnn_input(sat, list(fl), train=True)
            if xi is None:
                return
            try:
                yv = read_target(config.TRAIN_TARGET_DIR / gpm).astype(np.float32)
            except Exception:
                return
            X[i] = xi.astype(np.float16)
            Y[i] = yv.astype(np.float16)
            fold[i] = fd

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for _ in ex.map(_w, rows, chunksize=16):
                pass
        X.flush()
        Y.flush()
        nvalid = int((fold >= 0).sum())
        np.savez(str(mp), fold=fold, n=N, n_valid=nvalid, channels=CNN_CHANNELS)
        summary[sat] = nvalid
        print(f"[preprocess:cnn] {sat}: {nvalid:,}/{N:,} 有効 -> {Xp} (C={CNN_CHANNELS})")
        del X, Y
    print(f"[preprocess:cnn] 完了 summary={summary}")
    return summary


# --- 学習オーケストレーション ------------------------------------------------


def train_cnn(cfg, out_dir: Path) -> dict:
    """衛星別 FCN を学習し成果物を保存する。cv_mode=holdout（高速）/ full（5fold OOF）。"""
    from . import cv as cvmod

    ccfg = cfg.get("cnn", {}) or {}
    seed = int(ccfg.get("seed", 42))
    cv_mode = str(ccfg.get("cv_mode", "holdout"))
    holdout_fold = int(ccfg.get("holdout_fold", 0))
    loc_to_fold = cvmod.load_handdesigned_folds()
    n_splits = (max(loc_to_fold.values()) + 1) if loc_to_fold else 5

    models_dir = out_dir / "phase3_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    present: dict[str, tuple] = {}
    for sat in ("himawari", "goes", "meteosat"):
        Xp, yp, mp = _paths(out_dir, sat)
        if not Xp.exists():
            print(f"[train:cnn] 警告: memmap なし、スキップ: {Xp}")
            continue
        present[sat] = (Xp, yp, mp)
    if not present:
        raise FileNotFoundError(
            "CNN memmap がありません。先に `uv run python src/preprocess_train.py --method cnn`。"
        )

    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    per_sat: dict[str, float] = {}
    for sat, (Xp, yp, mp) in present.items():
        X = np.load(str(Xp), mmap_mode="r")
        Y = np.load(str(yp), mmap_mode="r")
        fold = np.load(str(mp))["fold"]
        valid = np.where(fold >= 0)[0]
        print(f"[train:cnn] {sat}: {len(valid):,} 有効サンプル, cv_mode={cv_mode}")

        if cv_mode == "full":
            ys, ps = [], []
            for f in range(n_splits):
                tr = valid[fold[valid] != f]
                va = valid[fold[valid] == f]
                if len(va) == 0 or len(tr) == 0:
                    continue
                print(f"   [{sat}] fold {f}: train {len(tr):,} / val {len(va):,}")
                _model, vp = _train_loop(tr, va, X, Y, ccfg, seed + f)
                ys.append(np.asarray(Y[va], np.float32).ravel())
                ps.append(vp)
            yv = np.concatenate(ys)
            pv = np.concatenate(ps)
            per_sat[sat] = rmse(yv, pv)
            all_y.append(yv)
            all_p.append(pv)
            print(f"   [{sat}] 最終 fit on all {len(valid):,}")
            model, _ = _train_loop(valid, np.zeros(0, int), X, Y, ccfg, seed)
            _save_model(model, models_dir / f"{sat}.pt")
        else:  # holdout
            tr = valid[fold[valid] != holdout_fold]
            va = valid[fold[valid] == holdout_fold]
            print(f"   [{sat}] holdout fold {holdout_fold}: train {len(tr):,} / val {len(va):,}")
            model, vp = _train_loop(tr, va, X, Y, ccfg, seed)
            yv = np.asarray(Y[va], np.float32).ravel()
            per_sat[sat] = rmse(yv, vp)
            all_y.append(yv)
            all_p.append(vp)
            _save_model(model, models_dir / f"{sat}.pt")  # 80%学習モデル（提出可）
        del X, Y

    Yc = np.concatenate(all_y)
    Pc = np.concatenate(all_p)
    overall = rmse(Yc, Pc)
    cond1 = _cond_rmse(Yc, Pc, 1.0)
    cond5 = _cond_rmse(Yc, Pc, 5.0)
    bias = float(np.mean(Pc - Yc))
    print(f"[train:cnn] {cv_mode} 評価 overall_RMSE={overall:.4f} "
          f"per_sat={ {k: round(v, 4) for k, v in per_sat.items()} } "
          f"cond>=1={cond1 and round(cond1, 3)} cond>=5={cond5 and round(cond5, 3)} bias={bias:+.4f}")

    cv_path = out_dir / "phase3_cv.json"
    with cv_path.open("w", encoding="utf-8") as f:
        json.dump({
            "method": "cnn", "cv_mode": cv_mode, "holdout_fold": holdout_fold,
            "overall": overall, "per_satellite": per_sat,
            "cond_rmse_ge1": cond1, "cond_rmse_ge5": cond5, "bias": bias,
            "gbdt_temporal_ref": {"cv": 1.1629, "lb": 0.70705},
            "honest_groupkfold_globalmean": 1.4048,
        }, f, ensure_ascii=False, indent=2)
    sel_path = out_dir / "phase3_selected.json"
    with sel_path.open("w", encoding="utf-8") as f:
        json.dump({
            "method": "cnn", "models_dir": str(models_dir),
            "channels": CNN_CHANNELS, "cv_mode": cv_mode,
            "cv_rmse_overall": overall,
        }, f, ensure_ascii=False, indent=2)
    print(f"[train:cnn] 保存: {cv_path} / {sel_path} / {models_dir}/")
    return {"overall": overall, "per_satellite": per_sat, "cv_mode": cv_mode}
