#!/usr/bin/env bash
# Phase2t（97特徴・時間特徴あり）に two_part×tweedie ブレンドを載せて再学習+予測する。
# 特徴スキーマは不変（97特徴）なので前処理は不要＝既存 parquet を再利用する。
# train が CV の OOF から衛星別ブレンド重みを fold-out 評価し、単一最良を上回るときのみ
# variant="blend" を採用（劣れば単一にフォールバック＝有害ブレンドを掴まない自己防衛）。
# predict は phase2_selected.json の variant を見て blend なら両モデルを結合する。
# 提出名は phase2_gbdt_temporal_blend。
#
# 前提: outputs/phase2_features_{himawari,goes,meteosat}.parquet（97特徴）が存在すること
#       （run-phase2_temporal.sh の生成物）。無ければ先に run-phase2_temporal.sh を実行。
# 設定: conf/config.yaml gbdt.ensemble: blend / gbdt.calibration: none。
#
# 堅牢化: set -euo pipefail / PYTHONUNBUFFERED / tee。
set -euo pipefail

export PYTHONUNBUFFERED=1

PHASE=phase2
DATETIME_NOW=$(date +"%Y-%m-%d-%H%M%S")
LOG_DIR="./logs/$PHASE/${DATETIME_NOW}-ensemble"
mkdir -p "$LOG_DIR"
echo "ログ出力先: $LOG_DIR"

# 特徴 parquet（97特徴）が揃っているか確認（スキーマ不変なので前処理は再実行しない）。
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

# 1. 学習（CV→OOF→ブレンド fold-out 判定→採用なら両モデル fit/保存）。
uv run python -u src/train.py --method gbdt 2>&1 | tee "$LOG_DIR/train.log"

# 2. 予測（variant=blend なら per-sat 重みで結合して提出 zip 生成）。
uv run python -u src/predict.py --method gbdt --name phase2_gbdt_temporal_blend 2>&1 | tee "$LOG_DIR/predict.log"

echo "完了。提出: submissions/phase2_gbdt_temporal_blend.zip"
echo "train.log の『ブレンド(...) fold-out OOF: 単一 X -> ブレンド Y (gain ...) => 採用/不採用』を確認。"
echo "『不採用』なら variant=tweedie のまま＝temporal版(LB0.70705)と同一予測なので提出不要。『採用』なら提出して比較。"
