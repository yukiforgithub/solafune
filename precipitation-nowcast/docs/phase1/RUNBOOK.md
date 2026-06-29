# Phase 1 実行手順（RUNBOOK）

> IR窓 Tb(DN)→RR 物理ベースライン（window_lookup / 衛星別）を最初から再現する手順。
> すべて `uv run` で実行（リポジトリルートで）。前提: `data/` に train/eval/sample が展開済み、`uv sync` 済み。

## 全体の流れ

```
build_phase1_suffstats   →  train  →  predict  →  validate
(充足統計キャッシュ構築)     (fit/CV)   (eval提出)  (形式照合)
```

`conf/config.yaml` は既に `method: physical_ir` が既定なので、各コマンドの `--method physical_ir` は省略可（明示推奨）。

---

## 手順

### 0.（任意）環境確認
```bash
uv run python src/train.py --help
uv run python src/predict.py --help
```

### 1. 充足統計キャッシュの構築 ★重い（全TRAIN 1パス走査）
```bash
uv run python -m src.build_phase1_suffstats
```
- 動作: TRAIN 40,686 行の最新フレームから 窓DN(衛星別 idx) と split相手(idx15) を読み、41×41(INTER_AREA)へ縮約。per-(衛星×地域×bin) の count/Σy/Σy² を集計。
- 出力: `eda_cache/phase1_window_hist.parquet`, `eda_cache/phase1_splitdiff_hist.parquet`
- 末尾に衛星別 Spearman（符号確認: 窓は負相関のはず）を表示。
- 目安: 8並列で **約5〜10分**。
- ※既にキャッシュが在れば再構築不要（作り直したい時のみ）。

### 2. 学習（fit + 地域CV） 軽い
```bash
uv run python src/train.py --method physical_ir
```
- 動作: 窓DNヒストと `conf/folds.yaml`（手設計5fold）から、window_lookup/衛星別 の**閉形式CV RMSE**を算出し、全TRAINで確定lookupをfit。
- 出力: `outputs/phase1_model.json`（衛星別256bin lookup）, `outputs/phase1_train_cv.json`（CV RMSE）
- 目安: parquet読込のみで **1分未満**。
- 期待値: `CV RMSE overall ≈ 1.1644`（全体平均原点 1.4061 から gain ≈ 0.242）。衛星別 meteosat≈0.77 / goes≈1.29 / himawari≈1.45。

### 3. 予測（eval提出生成） 中程度
```bash
uv run python src/predict.py --method physical_ir --name phase1_physical_ir
```
- 動作: EVAL 29,090 行の最新フレームに衛星別lookupを適用（負値→0クリップ、0フレーム29件は気候値0.2886フォールバック）。8並列。
- 出力: `submissions/phase1_physical_ir/`（test_files/ + evaluation_target.csv）と `submissions/phase1_physical_ir.zip`
- 目安: eval入力29k枚読込で **約3〜6分**。
- 動作確認用に少数だけ試すなら: `--limit 200`（提出には使わない）。

### 4. 提出形式の検証
```bash
uv run python scratch/validate_submission.py
```
- チェック: zip構成 / tif名集合がsample/evalと一致(29,090) / 各tif (1,41,41) float32・非負・有限 / 同梱CSVがEVALとバイト一致。
- 期待: `FORMAT_VALID=True`

---

## 出力物まとめ

| ファイル | 内容 |
|---|---|
| `eda_cache/phase1_window_hist.parquet` | 窓DN per-(衛星×地域×bin) 充足統計 |
| `eda_cache/phase1_splitdiff_hist.parquet` | split-window差 同上 |
| `outputs/phase1_model.json` | 衛星別256bin lookup（確定モデル） |
| `outputs/phase1_train_cv.json` | CV RMSE（overall/fold別/衛星別/gain） |
| `submissions/phase1_physical_ir.zip` | 提出zip（LB投稿用） |

---

## 補足・注意

- **前処理スタブ（任意）**: `src/preprocess_train.py` / `preprocess_test.py --method physical_ir` はデータ妥当性検証＋最小集計テーブルを出すスタブ。`physical_ir` の train/predict はキャッシュ/モデルを直接読むため**必須ではない**（3モジュール構成の体裁として実行可）。
- **CV値は未検証**: 1.1644 は閉形式（キャッシュ由来）。raw からの独立再検算（敵対的検証）は中断で未完。最終採用前に別実装で再計算して突き合わせると安全。
- **モデル比較スイープは未コミット**: lookup/べき乗則/指数/splitdiff × {衛星別,統一} の比較表は `docs/phase1/sections/10_fit_and_cv.md` に結果が残るが、それを生成する一括ドライバCLIは未保存。fit関数は `src/precip/phase1_fit.py`（`fit_lookup_window` / `fit_powerlaw` / `fit_exp` / `fit_lookup_simple`）に在る。比較を再実行したい場合は専用ドライバ `src/experiments/phase1_compare.py` を追加すると良い（未作成）。
- **大容量出力**: `submissions/*.zip`・`outputs/`・`eda_cache/` は `.gitignore` 対象（コミット不要）。
- **LB投稿**: `submissions/phase1_physical_ir.zip` をそのまま Solafune に投稿 → Public LB と CV(1.1644) の相関を確認。
