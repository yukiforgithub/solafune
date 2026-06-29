#!/usr/bin/env bash
# 実験3: HP単独効果の提出版。97特徴(temporalのみ, 物理OFF) + 確定HP(reg_strong+power1.7) を 2.5M で。
# PRECIP_NO_PHYSICS=1 で features を 97特徴に固定（物理特徴を除外）。
# 提出名 phase2_gbdt_hp。実験1(物理+HP)とLBで比べると「物理の上乗せ」が分離できる。
#   - temporal(旧HP, LB0.70705) → phase2_gbdt_hp(新HP) = HP単独のLB効果
#   - phase2_gbdt_hp(新HP,97) → phase2_gbdt_physics(新HP,102) = 物理のLB効果
#
# 前処理のメモリ二重対策(del解放+concatピーク半減)済なので 2.5M でOOMしない見込み。
set -euo pipefail
export PYTHONUNBUFFERED=1
export PRECIP_NO_PHYSICS=1   # ★物理特徴OFF=97特徴に固定

PHASE=phase2
DATETIME_NOW=$(date +"%Y-%m-%d-%H%M%S")
LOG_DIR="./logs/$PHASE/${DATETIME_NOW}-hp"
mkdir -p "$LOG_DIR"
echo "ログ出力先: $LOG_DIR (PRECIP_NO_PHYSICS=$PRECIP_NO_PHYSICS)"

rm -f outputs/phase2_features_*.parquet

uv run python -u src/preprocess_train.py --method gbdt 2>&1 | tee "$LOG_DIR/preprocess_train.log"

uv run python -u - <<'PY' 2>&1 | tee "$LOG_DIR/check_features.log"
import sys
import pyarrow.parquet as pq
sys.path.insert(0, "src")
from precip import config, features
ok = True
expect = len(features.feature_names())  # = 97 (PRECIP_NO_PHYSICS=1)
print(f"expect features = {expect}")
for sat in ("himawari", "goes", "meteosat"):
    p = config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet"
    if not p.exists():
        print(f"NG: {sat} なし: {p}"); ok = False; continue
    md = pq.ParquetFile(str(p)).metadata
    print(f"OK: {sat}: {md.num_rows:,} 行 / {md.num_columns} 列")
    if md.num_rows < 500_000 or md.num_columns != expect + 2:
        print(f"NG: {sat} 行/列が想定外（{md.num_rows:,}行/{md.num_columns}列, 想定{expect+2}列）"); ok = False
sys.exit(0 if ok else 1)
PY

uv run python -u src/train.py --method gbdt 2>&1 | tee "$LOG_DIR/train.log"
uv run python -u src/predict.py --method gbdt --name phase2_gbdt_hp 2>&1 | tee "$LOG_DIR/predict.log"

echo "完了。提出: submissions/phase2_gbdt_hp.zip / CV: outputs/phase2_cv.json"
echo "比較: temporal(旧HP) LB0.70705。HP単独版のCV/LBで HPの純効果を確認。"
