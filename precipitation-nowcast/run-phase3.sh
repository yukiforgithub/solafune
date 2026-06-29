#!/usr/bin/env bash
# Phase3 CNN（衛星別 小型FCN, CPU学習）一括: 前処理(memmap) → 検査 → 学習 → 予測。
# GBDT(0.70705)を CNN が超えられるかをまず holdout(fold0検証) で高速に確認する2段構え。
#   1段目(本スクリプト): conf cnn.cv_mode=holdout → 80%学習・fold0検証で competitive か判定。
#     train.log の『holdout 評価 overall_RMSE=…』を GBDT temporal(CV1.1629/LB0.70705)と比較。
#     提出 phase3_cnn_holdout.zip を投げて LB を見てもよい（モデルは80%学習）。
#   2段目(有望なら): conf cnn.cv_mode=full に変えて再実行 → 5fold OOF(正直なLB代理)+全データ最終fit。
#
# ★CPU学習は重い: 8コアで holdout 全データ ≈ 数時間, full ≈ その6倍。長時間ジョブ前提。
#   メモリ: 入力 memmap は衛星別 ~2.2GB/disk(f16, mmap)。学習RAMはバッチ分のみ。OOM時は
#   conf cnn.batch_size を下げる。
#
# 堅牢化: set -euo pipefail / PYTHONUNBUFFERED / tee。スキーマ変更につき memmap を再生成。
set -euo pipefail

export PYTHONUNBUFFERED=1

PHASE=phase3
DATETIME_NOW=$(date +"%Y-%m-%d-%H%M%S")
LOG_DIR="./logs/$PHASE/${DATETIME_NOW}"
mkdir -p "$LOG_DIR"
echo "ログ出力先: $LOG_DIR"

# 旧 memmap を削除して再生成（部分データ混入防止）。
rm -f outputs/phase3_cnn/*.npy outputs/phase3_cnn/*.npz

# 1. 前処理（衛星別 入力テンソル memmap, C=51）。
uv run python -u src/preprocess_train.py --method cnn 2>&1 | tee "$LOG_DIR/preprocess_train.log"

# 1.5 memmap が全衛星ぶん揃ったか検査。
uv run python -u - <<'PY' 2>&1 | tee "$LOG_DIR/check_memmap.log"
import sys
import numpy as np
sys.path.insert(0, "src")
from precip import config, cnn
ok = True
for sat in ("himawari", "goes", "meteosat"):
    Xp, yp, mp = cnn._paths(config.OUTPUTS_DIR, sat)
    if not (Xp.exists() and mp.exists()):
        print(f"NG: {sat} の memmap がありません: {Xp}")
        ok = False
        continue
    m = np.load(str(mp))
    nv = int(m["n_valid"]); ch = int(m["channels"])
    print(f"OK: {sat}: valid {nv:,} / C={ch}")
    if nv < 5000:
        print(f"NG: {sat} の有効サンプル {nv:,} が少なすぎます（前処理途中失敗の疑い）。")
        ok = False
    if ch != cnn.CNN_CHANNELS:
        print(f"NG: {sat} の C={ch} が想定 {cnn.CNN_CHANNELS} と不一致。")
        ok = False
sys.exit(0 if ok else 1)
PY

# 2. 学習（conf cnn.cv_mode に従う。既定 holdout）。
uv run python -u src/train.py --method cnn 2>&1 | tee "$LOG_DIR/train.log"

# 3. 予測（提出 zip 生成）。cv_mode に応じ提出名を変える。
NAME=$(uv run python -c "import json;print('phase3_cnn_'+json.load(open('outputs/phase3_selected.json'))['cv_mode'])")
uv run python -u src/predict.py --method cnn --name "$NAME" 2>&1 | tee "$LOG_DIR/predict.log"

echo "完了。提出: submissions/${NAME}.zip / CV: outputs/phase3_cv.json"
echo "比較基準: GBDT temporal = CV(fold-out)1.1629 / LB0.70705。CNN の overall_RMSE がこれを下回れば有望。"
