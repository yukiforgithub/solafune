#!/usr/bin/env bash
# Phase2t（時間特徴97）+ 地域不変なシーン内正規化特徴(8) = 105特徴 の一括実行。
# 各主要バンド{win,split,wv,ir38}の41x41を「自分自身の統計」で正規化した
#   scene_z(シーン内zスコア) / scene_rank(シーン内順位[0,1], 単調変換不変)
# を追加。train/eval 地域 DISJOINT への汎化策（地域/センサ固有のDNオフセット・ゲイン差を除去）。
# 絶対DN(Tb→RRの根幹)は既存特徴で保持し、シーン相対を「追加」して GBDT に両方持たせる。
# 提出名は phase2_gbdt_scenenorm。スキーマ変更(97→105)につき前処理から再生成。
#
# ★判定: これは入力の関数=地域非依存な変換なので、改善すれば fold-out OOF にそのまま現れる
#   （較正/ブレンドのような地域非転移の罠は無い）。train.log の overall を temporal の
#   1.1629 と比較。改善すれば提出、横ばい/悪化なら棄却（特徴飽和）。
#
# 堅牢化: set -euo pipefail / PYTHONUNBUFFERED / tee。
set -euo pipefail

export PYTHONUNBUFFERED=1

PHASE=phase2
DATETIME_NOW=$(date +"%Y-%m-%d-%H%M%S")
LOG_DIR="./logs/$PHASE/${DATETIME_NOW}-scenenorm"
mkdir -p "$LOG_DIR"
echo "ログ出力先: $LOG_DIR"

# スキーマ変更（97→105）に伴い旧キャッシュを削除して再生成。
rm -f outputs/phase2_features_*.parquet

# 1. 前処理（全衛星の画素特徴テーブル, 105特徴）。
uv run python -u src/preprocess_train.py --method gbdt 2>&1 | tee "$LOG_DIR/preprocess_train.log"

# 1.5 前処理が全衛星ぶん揃い、列数が想定(105+y+fold)か検査。
uv run python -u - <<'PY' 2>&1 | tee "$LOG_DIR/check_features.log"
import sys
import pyarrow.parquet as pq
sys.path.insert(0, "src")
from precip import config, features
ok = True
expect_cols = len(features.feature_names())  # = 105
for sat in ("himawari", "goes", "meteosat"):
    p = config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet"
    if not p.exists():
        print(f"NG: {sat} の特徴がありません: {p}")
        ok = False
        continue
    md = pq.ParquetFile(str(p)).metadata
    print(f"OK: {sat}: {md.num_rows:,} 行 / {md.num_columns} 列")
    if md.num_rows < 500_000:
        print(f"NG: {sat} の行数 {md.num_rows:,} が少なすぎます（前処理途中失敗の疑い）。")
        ok = False
    if md.num_columns != expect_cols + 2:
        print(f"NG: {sat} の列数 {md.num_columns} が想定 {expect_cols + 2}(特徴{expect_cols}+y+fold) と不一致。")
        ok = False
sys.exit(0 if ok else 1)
PY

# 2. 学習（地域CV + 最終fit + feature importance）。
uv run python -u src/train.py --method gbdt 2>&1 | tee "$LOG_DIR/train.log"

# 3. 予測（eval 提出 zip 生成）。
uv run python -u src/predict.py --method gbdt --name phase2_gbdt_scenenorm 2>&1 | tee "$LOG_DIR/predict.log"

echo "完了。提出: submissions/phase2_gbdt_scenenorm.zip / CV: outputs/phase2_cv.json"
echo "比較: temporal版 CV=1.1629 / LB=0.70705。scene正規化版の overall と cond>=5 を確認。"
echo "feature importance(outputs/phase2_feature_importance.csv)で *_scene_z/*_scene_rank の gain も確認。"
