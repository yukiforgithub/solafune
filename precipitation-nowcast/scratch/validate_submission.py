"""提出 zip を SAMPLE_SUB と形式照合する（使い捨て検証スクリプト）。

チェック:
  1. zip エントリ構成: evaluation_target.csv + test_files/*.tif のみ。
  2. test_files のファイル名集合が SAMPLE_SUB の test_files と完全一致。
  3. 各 tif が (1, 41, 41) float32・NaN 無し・負値無し。
  4. 同梱 evaluation_target.csv が提供 EVAL CSV とバイト一致。
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precip import config  # noqa: E402

ZIP = config.SUBMISSIONS_DIR / "phase1_physical_ir.zip"
SAMPLE_TEST = config.SAMPLE_SUB_DIR / "test_files"


def main() -> int:
    issues: list[str] = []

    # --- zip エントリ構成 ---
    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
    has_csv = "evaluation_target.csv" in names
    tif_entries = [n for n in names if n.startswith("test_files/") and n.endswith(".tif")]
    other = [n for n in names if n != "evaluation_target.csv" and not (n.startswith("test_files/") and n.endswith(".tif")) and not n.endswith("/")]
    if not has_csv:
        issues.append("evaluation_target.csv が zip に無い")
    if other:
        issues.append(f"想定外エントリ: {other[:5]}")

    zip_tif_names = {Path(n).name for n in tif_entries}

    # --- SAMPLE_SUB のファイル名集合 ---
    sample_names = {p.name for p in SAMPLE_TEST.iterdir() if p.suffix == ".tif"}

    only_zip = zip_tif_names - sample_names
    only_sample = sample_names - zip_tif_names
    if only_zip:
        issues.append(f"提出のみに存在 {len(only_zip)} 件: {sorted(only_zip)[:3]}")
    if only_sample:
        issues.append(f"SAMPLE のみに存在 {len(only_sample)} 件: {sorted(only_sample)[:3]}")
    name_set_match = (not only_zip) and (not only_sample)

    # --- 各 tif のプロファイル検証（展開先 work_dir を直接読む） ---
    work_test = config.SUBMISSIONS_DIR / "phase1_physical_ir" / "test_files"
    n_checked = 0
    n_bad = 0
    gmin, gmax = np.inf, -np.inf
    for p in work_test.iterdir():
        if p.suffix != ".tif":
            continue
        with rasterio.open(p) as ds:
            if ds.count != 1 or ds.width != 41 or ds.height != 41 or ds.dtypes[0] != "float32":
                n_bad += 1
                if n_bad <= 3:
                    issues.append(f"プロファイル不一致 {p.name}: count={ds.count} {ds.width}x{ds.height} {ds.dtypes[0]}")
                continue
            a = ds.read(1)
        if not np.isfinite(a).all():
            n_bad += 1
            issues.append(f"NaN/inf を含む: {p.name}")
        if a.min() < 0:
            n_bad += 1
            issues.append(f"負値を含む: {p.name} min={a.min()}")
        gmin = min(gmin, float(a.min()))
        gmax = max(gmax, float(a.max()))
        n_checked += 1

    # --- CSV バイト一致 ---
    with zipfile.ZipFile(ZIP) as zf:
        zip_csv = zf.read("evaluation_target.csv")
    eval_csv = config.EVAL_CSV.read_bytes()
    csv_match = zip_csv == eval_csv
    if not csv_match:
        issues.append("同梱 evaluation_target.csv が EVAL CSV とバイト不一致")

    print(f"zip エントリ: tif={len(tif_entries)} csv={has_csv}")
    print(f"ファイル名集合一致: {name_set_match} (zip={len(zip_tif_names)} sample={len(sample_names)})")
    print(f"tif 検証: checked={n_checked} bad={n_bad} 値域=[{gmin:.4f}, {gmax:.4f}]")
    print(f"CSV バイト一致: {csv_match}")
    valid = name_set_match and (n_bad == 0) and csv_match and has_csv and not other
    print(f"FORMAT_VALID={valid}")
    if issues:
        print("ISSUES:")
        for it in issues[:10]:
            print("  -", it)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
