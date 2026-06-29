#!/usr/bin/env bash
# Phase2t（97特徴・時間特徴あり）に後処理較正(isotonic)を載せて再学習+予測する。
# 特徴スキーマは不変（97特徴のまま）なので前処理は不要＝既存 parquet を再利用する。
# train が CV の OOF から衛星別 isotonic LUT を fit し outputs/phase2_calibrators.json に保存、
# predict がそれを適用する。提出名は phase2_gbdt_temporal_cal。
#
# 前提: outputs/phase2_features_{himawari,goes,meteosat}.parquet（97特徴）が存在すること
#       （直前の run-phase2_temporal.sh の生成物）。無ければ先に run-phase2_temporal.sh を実行。
#
# 堅牢化: set -euo pipefail / PYTHONUNBUFFERED / tee。
set -euo pipefail

export PYTHONUNBUFFERED=1

PHASE=phase2
DATETIME_NOW=$(date +"%Y-%m-%d-%H%M%S")
LOG_DIR="./logs/$PHASE/${DATETIME_NOW}-calibrate"
mkdir -p "$LOG_DIR"
echo "ログ出力先: $LOG_DIR"

# 特徴 parquet（97特徴）が揃っているか確認。スキーマ不変なので前処理は再実行しない。
uv run python -u - <<'PY' 2>&1 | tee "$LOG_DIR/check_features.log"
import sys
import pyarrow.parquet as pq
sys.path.insert(0, "src")
from precip import config, features
ok = True
expect_cols = len(features.feature_names())  # = 97
for sat in ("himawari", "goes", "meteosat"):
    p = config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet"
    if not p.exists():
        print(f"NG: {sat} の特徴がありません: {p}  先に run-phase2_temporal.sh を実行。")
        ok = False
        continue
    md = pq.ParquetFile(str(p)).metadata
    print(f"OK: {sat}: {md.num_rows:,} 行 / {md.num_columns} 列")
    if md.num_columns != expect_cols + 2:
        print(f"NG: {sat} の列数 {md.num_columns} が想定 {expect_cols + 2}(特徴{expect_cols}+y+fold) と不一致。")
        ok = False
sys.exit(0 if ok else 1)
PY

# 1. 学習（CV→OOF→isotonic LUT 保存 + 最終fit）。conf gbdt.calibration: isotonic が有効。
uv run python -u src/train.py --method gbdt 2>&1 | tee "$LOG_DIR/train.log"

# 2. 予測（LUT を適用して提出 zip 生成）。提出名は較正版。
uv run python -u src/predict.py --method gbdt --name phase2_gbdt_temporal_cal 2>&1 | tee "$LOG_DIR/predict.log"

echo "完了。提出: submissions/phase2_gbdt_temporal_cal.zip"
echo "train.log の『較正(isotonic) OOF RMSE: X -> Y (gain ...)』を確認。gain が＋なら提出。"
echo "較正なし版(temporal)の LB=0.70705 と比較する。"
