"""2つの提出zipの予測tifをファイル名ごとに重み付き平均し、新しい提出zipを作る。

CNN(phase3_cnn_unet 0.692) × GBDT(phase2_gbdt_hp 0.707) のアンサンブル用。
別モデル族で誤差が脱相関なので、平均で両者を上回る可能性がある。

使い方:
  uv run python scratch/ensemble_submissions.py \
    --cnn submissions/phase3_cnn_unet.zip \
    --gbdt submissions/phase2_gbdt_hp.zip \
    --w-cnn 0.5 --out submissions/ens_cnn_gbdt_050.zip
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


def _extract(zpath: Path, dst: Path) -> Path:
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", required=True, type=Path)
    ap.add_argument("--gbdt", required=True, type=Path)
    ap.add_argument("--w-cnn", type=float, default=0.5, help="CNN側の重み(0-1)。GBDTは1-w。")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    w = float(args.w_cnn)

    tmp = Path(tempfile.mkdtemp())
    a = _extract(args.cnn, tmp / "cnn")
    b = _extract(args.gbdt, tmp / "gbdt")
    a_tf = a / "test_files"
    b_tf = b / "test_files"
    out_tf = tmp / "out" / "test_files"
    out_tf.mkdir(parents=True, exist_ok=True)

    a_names = {p.name for p in a_tf.iterdir()}
    b_names = {p.name for p in b_tf.iterdir()}
    common = a_names & b_names
    only_a = a_names - b_names
    only_b = b_names - a_names
    print(f"[ens] CNN={len(a_names)} GBDT={len(b_names)} 共通={len(common)} CNNのみ={len(only_a)} GBDTのみ={len(only_b)}")
    print(f"[ens] 重み: CNN={w:.2f} / GBDT={1-w:.2f}")

    h = wd = 41
    prof = dict(driver="GTiff", dtype="float32", nodata=None, width=wd, height=h, count=1,
                crs=None, transform=Affine.identity())

    def read(p):
        with rasterio.open(p) as ds:
            return ds.read(1).astype(np.float32)

    def write(p, arr):
        with rasterio.open(p, "w", **prof) as ds:
            ds.write(np.clip(arr, 0.0, None).astype(np.float32), 1)

    done = 0
    for nm in common:
        ya = read(a_tf / nm)
        yb = read(b_tf / nm)
        write(out_tf / nm, w * ya + (1.0 - w) * yb)
        done += 1
    # 片方にしか無い行はそのまま採用（通常は共通のはず）
    for nm in only_a:
        write(out_tf / nm, read(a_tf / nm))
    for nm in only_b:
        write(out_tf / nm, read(b_tf / nm))
    print(f"[ens] 平均tif書き出し: {done + len(only_a) + len(only_b)}")

    # evaluation_target.csv はどちらかから
    csv_src = (a / "evaluation_target.csv")
    if not csv_src.exists():
        csv_src = (b / "evaluation_target.csv")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_src, "evaluation_target.csv")
        for tif in sorted(out_tf.iterdir()):
            zf.write(tif, f"test_files/{tif.name}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ens] 提出zip: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
