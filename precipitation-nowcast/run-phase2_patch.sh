#!/usr/bin/env bash
# Phase2 (GBDT) + サブグリッド・パッチ統計特徴 の一括実行: 前処理 → 検査 → 学習 → 予測。
# 特徴スキーマが変わった（63→83）ため、前処理から再生成する（rm + preprocess を有効化）。
# 提出名は phase2_gbdt_patch（patch なし版の submissions/phase2_gbdt.zip を上書きしない）。
#
# 堅牢化:
#   - set -euo pipefail : いずれかの段が失敗したら即停止（壊れた状態で次段へ進まない）
#   - PYTHONUNBUFFERED / -u : 出力をバッファせず即ログ/端末へ
#   - tee               : ログファイルと端末の両方へ出力
#   - 古い特徴キャッシュ削除 + 前処理完了チェック : 部分/smoke データの混入を防ぐ
set -euo pipefail

export PYTHONUNBUFFERED=1

PHASE=phase2
DATETIME_NOW=$(date +"%Y-%m-%d-%H%M%S")
LOG_DIR="./logs/$PHASE/${DATETIME_NOW}-patch"
mkdir -p "$LOG_DIR"
echo "ログ出力先: $LOG_DIR"

# スキーマ変更（patch 統計追加で 63→83 特徴）に伴い、旧キャッシュを必ず削除して再生成。
rm -f outputs/phase2_features_*.parquet

# 1. 前処理（全衛星の画素特徴テーブル, 83特徴）。失敗（OOM等）したら set -e で停止。
uv run python -u src/preprocess_train.py --method gbdt 2>&1 | tee "$LOG_DIR/preprocess_train.log"

# 1.5 前処理が全衛星ぶん揃ったか検査。揃わなければここで停止（壊れた学習を防ぐ）。
# ※ヒアドキュメント <<'PY' は python に結合させる（| tee の前に置く）。tee 側に付くと
#   python が端末 stdin を読みに行きハングするため位置が重要。
uv run python -u - <<'PY' 2>&1 | tee "$LOG_DIR/check_features.log"
import sys
import pyarrow.parquet as pq
sys.path.insert(0, "src")
from precip import config
ok = True
for sat in ("himawari", "goes", "meteosat"):
    p = config.OUTPUTS_DIR / f"phase2_features_{sat}.parquet"
    if not p.exists():
        print(f"NG: {sat} の特徴がありません: {p}")
        ok = False
        continue
    n = pq.ParquetFile(str(p)).metadata.num_rows
    print(f"OK: {sat}: {n:,} 行")
    if n < 500_000:
        print(f"NG: {sat} の行数 {n:,} が少なすぎます（前処理が途中失敗の疑い）。")
        ok = False
sys.exit(0 if ok else 1)
PY

# 2. 学習（地域CV + 最終fit + feature importance）。
uv run python -u src/train.py --method gbdt 2>&1 | tee "$LOG_DIR/train.log"

# 3. 予測（eval 提出 zip 生成）。提出名は patch 版。
uv run python -u src/predict.py --method gbdt --name phase2_gbdt_patch 2>&1 | tee "$LOG_DIR/predict.log"

echo "完了。提出: submissions/phase2_gbdt_patch.zip / CV: outputs/phase2_cv.json"
