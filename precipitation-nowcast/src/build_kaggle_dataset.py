"""Kaggle GPU 用の前処理: train/eval を コンパクトな CNN 入力テンソル(memmap f16) に変換。

27.7万個の生 GeoTIFF(32GB) を Kaggle へ上げるのは非現実的（ファイル数が膨大で遅い/失敗）。
そこで Phase3 の 51ch 入力（3フレーム×16band + presence3面, 41×41）に前処理して
少数の大きい .npy（計~12GB）にまとめ、Kaggle へは「前処理済みテンソルのみ」を上げる。

出力（outputs/kaggle_data/, フラット構成＝kaggle CLIが確実に全ファイルを上げる）:
  train_{sat}_X.npy     (N,51,41,41) f16   入力
  train_{sat}_y.npy     (N,41,41)    f16   ターゲット(mm/hr)
  train_{sat}_meta.npz  fold(N,) int16, n
  eval_{sat}_X.npy      (M,51,41,41) f16   入力（有効行を先頭に詰め）
  eval_{sat}_names.json [gpm_imerg_filename, ...] （X の行順, 有効行のみ）
  eval_manifest.parquet 全eval行: gpm_imerg_filename, satellite, valid（fallback判定用）
  meta.json             チャネル数・fold対応・気候値フォールバック値

低RAM: open_memmap で逐次ディスク書き込み。スレッドI/O並列。

実行: uv run python -m src.build_kaggle_dataset
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import numpy.lib.format as npf  # noqa: E402
import pandas as pd  # noqa: E402

from precip import config, cv as cvmod, dataio  # noqa: E402
from precip.cnn import CNN_CHANNELS, TARGET_H, TARGET_W, build_cnn_input  # noqa: E402

OUT = config.OUTPUTS_DIR / "kaggle_data"
N_WORKERS = 8
SATS = ("himawari", "goes", "meteosat")


def _build_train_sat(sub: pd.DataFrame, sat: str) -> int:
    """train の 1 衛星を memmap 化（入力+ターゲット+fold）。有効行のみ。戻り=有効行数。"""
    rows = [(r.frame_list, int(r.fold_), r.gpm_imerg_filename) for r in sub.itertuples(index=False)]
    N = len(rows)
    Xp, yp, mp = OUT / f"train_{sat}_X.npy", OUT / f"train_{sat}_y.npy", OUT / f"train_{sat}_meta.npz"
    X = npf.open_memmap(str(Xp), mode="w+", dtype=np.float16, shape=(N, CNN_CHANNELS, TARGET_H, TARGET_W))
    Y = npf.open_memmap(str(yp), mode="w+", dtype=np.float16, shape=(N, TARGET_H, TARGET_W))
    fold = np.full(N, -1, np.int16)

    def _w(i_row):
        i, (fl, fd, gpm) = i_row
        xi = build_cnn_input(sat, list(fl), train=True)
        if xi is None:
            return
        try:
            yv = dataio.read_target(config.TRAIN_TARGET_DIR / gpm).astype(np.float32)
        except Exception:
            return
        X[i] = xi.astype(np.float16)
        Y[i] = yv.astype(np.float16)
        fold[i] = fd

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for _ in ex.map(_w, list(enumerate(rows)), chunksize=16):
            pass
    X.flush(); Y.flush()
    nvalid = int((fold >= 0).sum())
    np.savez(str(mp), fold=fold, n=N, n_valid=nvalid)
    del X, Y
    print(f"[kaggle:train] {sat}: {nvalid:,}/{N:,} 有効 -> {Xp.name}", flush=True)
    return nvalid


def _build_eval_sat(sub: pd.DataFrame, sat: str) -> tuple[int, list[str]]:
    """eval の 1 衛星を memmap 化（入力のみ, 有効行のみ）。戻り=(有効行数, 無効行のファイル名)。"""
    rows = [(r.frame_list, r.gpm_imerg_filename) for r in sub.itertuples(index=False)]
    N = len(rows)
    # まず有効/無効を判定しつつ入力を作る（メモリに溜めず memmap へ）。N上限で確保し詰める。
    Xp = OUT / f"eval_{sat}_X.npy"
    X = npf.open_memmap(str(Xp), mode="w+", dtype=np.float16, shape=(N, CNN_CHANNELS, TARGET_H, TARGET_W))
    results: list[tuple[int, str, np.ndarray | None]] = [None] * N  # type: ignore

    def _w(i_row):
        i, (fl, gpm) = i_row
        xi = build_cnn_input(sat, list(fl), train=False)
        results[i] = (i, gpm, xi)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for _ in ex.map(_w, list(enumerate(rows)), chunksize=16):
            pass

    names: list[str] = []
    invalid: list[str] = []
    pos = 0
    for i, gpm, xi in results:
        if xi is None:
            invalid.append(gpm)
            continue
        X[pos] = xi.astype(np.float16)
        names.append(gpm)
        pos += 1
    X.flush()
    # 有効行 pos 件だけが意味を持つ（残りは未使用）。names と一致。
    (OUT / f"eval_{sat}_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    del X
    print(f"[kaggle:eval] {sat}: {pos:,}/{N:,} 有効, 無効{len(invalid)} -> {Xp.name}", flush=True)
    return pos, invalid


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- train ---
    loc2f = cvmod.load_handdesigned_folds()
    tdf = dataio.load_train_df()
    tdf = tdf.assign(fold_=tdf["name_location"].map(loc2f))
    tdf = tdf[tdf["fold_"].notna()].copy()
    tdf["fold_"] = tdf["fold_"].astype(int)
    train_summary = {}
    for sat in SATS:
        sub = tdf[tdf["satellite_target"] == sat]
        if len(sub):
            train_summary[sat] = _build_train_sat(sub, sat)

    # --- eval ---
    edf = dataio.load_eval_df()
    manifest_rows = []
    eval_summary = {}
    for sat in SATS:
        sub = edf[edf["satellite_target"] == sat]
        if not len(sub):
            continue
        nvalid, invalid = _build_eval_sat(sub, sat)
        eval_summary[sat] = nvalid
        # マニフェスト: 有効行(names.json)＋無効行(fallback) を満たすよう全行記録。
        names = json.loads((OUT / f"eval_{sat}_names.json").read_text(encoding="utf-8"))
        for nm in names:
            manifest_rows.append({"gpm_imerg_filename": nm, "satellite": sat, "valid": True})
        for nm in invalid:
            manifest_rows.append({"gpm_imerg_filename": nm, "satellite": sat, "valid": False})
    pd.DataFrame(manifest_rows).to_parquet(OUT / "eval_manifest.parquet", index=False)

    meta = {
        "channels": CNN_CHANNELS,
        "target_size": [TARGET_H, TARGET_W],
        "satellites": list(SATS),
        "n_folds": (max(loc2f.values()) + 1) if loc2f else 5,
        "global_mean_fallback": 0.2886,
        "channel_layout": "3 frames(old->new) x 16 bands + 3 presence; /255 normalized; INTER_AREA 41x41",
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "n_eval_total": int(len(edf)),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kaggle] 完了 -> {OUT}")
    print(f"[kaggle] train={train_summary} eval={eval_summary}")
    print(f"[kaggle] アップロード対象: {OUT}（このフォルダを kaggle datasets create で上げる）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
