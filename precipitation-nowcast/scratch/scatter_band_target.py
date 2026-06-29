"""バンドDN × 降水量の散布（hexbin密度 + 条件付き平均）を衛星別に可視化する。

ターゲットは82%が厳密0のゼロ過剰分布なので、生散布は y=0 の帯になる。そこで
hexbin（全点の密度＝実質散布）に E[y|DN] / median を重ねて関係を見せる。
y 軸は log1p(RR)（ゼロ過剰と強雨の裾を同時に見るため）。
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from precip import config, dataio, features  # noqa: E402

N_ROWS = 500          # 衛星あたりサンプル行数
SEED = 42
OUT_DIR = config.REPO_ROOT / "eda_cache" / "scatter_band_target"

# 衛星別に見せるバンド: (0-based index, ラベル)
BAND_PANELS = {
    "himawari": [(0, "B01 可視0.47µm"), (6, "B07 3.9µm"), (7, "B08 WV6.2µm"),
                 (10, "B11 雲相8.6µm"), (12, "B13 IR窓10.4µm"), (14, "B15 12.4µm(split)")],
    "goes": [(0, "C01 可視0.47µm"), (6, "C07 3.9µm"), (7, "C08 WV6.2µm"),
             (10, "C11 8.4µm"), (12, "C13 IR窓10.3µm"), (14, "C15 12.3µm(split)")],
    "meteosat": [(0, "vis_04 0.4µm"), (8, "ir_38 3.8µm"), (9, "wv_63 6.3µm"),
                 (11, "ir_87 8.7µm"), (13, "ir_105 IR窓"), (14, "ir_123 12.3µm(split)")],
}


def collect(sat: str, df) -> tuple[np.ndarray, np.ndarray]:
    """sat の最新フレームから (16,41,41) と target を集め (npix,16), (npix,) を返す。"""
    sub = df[df["satellite_target"] == sat].sample(n=min(N_ROWS, (df["satellite_target"] == sat).sum()),
                                                    random_state=SEED)
    subdir = config.TRAIN_DIR / config.SATELLITE_DIRNAMES[sat]

    def _one(r):
        fl = list(r.frame_list)
        if not fl:
            return None
        try:
            arr = dataio.read_satellite(subdir / fl[-1])
            if arr.shape[0] != 16:
                return None
            y = dataio.read_target(config.TRAIN_TARGET_DIR / r.gpm_imerg_filename)
        except Exception:
            return None
        stack = features.resize_stack(arr)  # (16,41,41) float32
        return stack.reshape(16, -1).T, y.ravel()

    Xs, ys = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(_one, [r for r in sub.itertuples(index=False)]):
            if res is None:
                continue
            Xs.append(res[0])
            ys.append(res[1])
    return np.concatenate(Xs), np.concatenate(ys)


def spearman(x, y) -> float:
    # 大標本の順位相関（タイは平均順位）。
    from scipy.stats import spearmanr  # scipy は scikit-image 依存で入っている
    # 計算コスト削減のため最大20万点にサブサンプル。
    if len(x) > 200_000:
        idx = np.random.default_rng(0).choice(len(x), 200_000, replace=False)
        x, y = x[idx], y[idx]
    return float(spearmanr(x, y).statistic)


def plot_sat(sat: str, X: np.ndarray, y: np.ndarray):
    panels = BAND_PANELS[sat]
    yl = np.log1p(y)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{sat}: バンドDN × log1p(降水量)  hexbin密度+条件付き平均  (n={len(y):,}画素, "
                 f"ゼロ率{(y==0).mean()*100:.0f}%)", fontsize=13)
    for ax, (bi, label) in zip(axes.ravel(), panels):
        dn = X[:, bi]
        hb = ax.hexbin(dn, yl, gridsize=60, bins="log", cmap="viridis", mincnt=1)
        # 条件付き統計（DN を16幅ビン）。
        edges = np.arange(0, 257, 16)
        idx = np.clip(np.digitize(dn, edges[1:-1]), 0, len(edges) - 2)
        centers, means, meds = [], [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 50:
                continue
            centers.append((edges[b] + edges[b + 1]) / 2)
            means.append(np.log1p(y[m].mean()))      # E[y|DN] を log1p で
            meds.append(np.median(yl[m]))
        ax.plot(centers, means, "r.-", lw=2, ms=8, label="log1p(E[y|DN])")
        ax.plot(centers, meds, "w.--", lw=1.5, ms=5, label="median log1p(y)")
        rho = spearman(dn, y)
        ax.set_title(f"{label}   ρ={rho:+.2f}", fontsize=10)
        ax.set_xlabel("DN (0-255)")
        ax.set_ylabel("log1p(RR)")
        ax.set_xlim(0, 255)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"scatter_{sat}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[scatter] {sat}: 保存 {out}  (zero率 {(y==0).mean()*100:.1f}%, "
          f"max RR {y.max():.1f}, mean {y.mean():.3f})")


def main():
    df = dataio.load_train_df()
    for sat in ("himawari", "goes", "meteosat"):
        X, y = collect(sat, df)
        plot_sat(sat, X, y)
    print(f"[scatter] 完了 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
