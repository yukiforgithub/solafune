# Phase 2 実行手順（RUNBOOK） — GBDT 2部 / Tweedie

> 衛星別 LightGBM（2部構成 hurdle + Tweedie 比較）を、Tier-1 特徴（16band + BTD + 多スケール空間統計 + 構造 + メタ + 位置, 計63特徴）で学習。
> すべて `uv run` でリポジトリルートから実行。前提: `data/` 展開済み、`uv sync` 済み。**実行はユーザー側**。

## 全体の流れ
```
preprocess_train --method gbdt   →  train --method gbdt  →  predict --method gbdt  →  validate
(衛星別 画素特徴 parquet)            (CV + 最終fit + importance)  (eval提出zip)         (形式照合)
```
preprocess_test はスタブ（特徴は predict が実行時抽出）。`conf/config.yaml` の `gbdt:` で挙動を制御。

---

## 0. 事前確認（依存）
```bash
uv run python -c "import lightgbm; print('lightgbm', lightgbm.__version__)"
uv run python src/train.py --help     # --method に gbdt があるか
```

## ★まずスモーク（小規模・数分）で配線確認 — 推奨
本番前に必ず小規模で通す。`--limit` は**各衛星からランダム抽出**（fold が分散するよう）。
```bash
# 1) 小さな特徴キャッシュ（各衛星 300行）
uv run python src/preprocess_train.py --method gbdt --limit 300
# 2) CV + 最終fit（tiny データなので速い）
uv run python src/train.py --method gbdt
# 3) 50行だけ予測（tif 生成確認）
uv run python src/predict.py --method gbdt --name phase2_smoke --limit 50
```
期待: train で `variant=... overall_RMSE=...`（NaNでない数値）が出る／predict で tif 50枚生成。
※スモークは行数が少なく fold/地域が偏るため、RMSE 値自体は本番と無関係（配線確認が目的）。

---

## 本番

### 1. 前処理: 衛星別 画素特徴テーブル ★重い（全TRAIN 1パス）
```bash
uv run python src/preprocess_train.py --method gbdt
```
- 出力: `outputs/phase2_features_{himawari,goes,meteosat}.parquet` + `outputs/phase2_feature_names.json`
- 内容: 各行最新フレーム→63特徴、base rate維持で `pixel_sample_frac`(0.25) 一様抽出、衛星別 `max_pixels`(600万) 上限。
- 目安: 入力tif 約4万枚読込で **〜10〜25分**（n_workers依存）。**メモリ 〜6〜8GB** 推奨。

### 2. 学習: CV + 最終モデル
```bash
uv run python src/train.py --method gbdt
```
- 動作: conf/folds.yaml の地域GroupKFoldで **two_part と tweedie の両方**をCV（`variant: both`）→ overall/衛星別/**強雨条件付き(y≥1,y≥5)**/bias を算出 → CV最小の variant を全データで最終fit。
- 出力: `outputs/phase2_cv.json`（CV結果）, `outputs/phase2_selected.json`（採用variant）, `outputs/phase2_models/`（衛星別 Booster）, `outputs/phase2_feature_importance.csv`
- 目安: `both` は重い（5fold×3衛星×2方式の多数fit）。**〜20〜60分**。短縮したい場合は下記「速く回す」参照。
- 評価基準: **Phase1 物理IR CV=1.1644 / 全体平均 honest=1.4048 を下回るか**。注意: CVは絶対値でなくLB転移の順位付け用（Phase1で CV1.164→LB0.763 の乖離を確認済）。

### 3. 予測: eval提出生成
```bash
uv run python src/predict.py --method gbdt --name phase2_gbdt
```
- 出力: `submissions/phase2_gbdt.zip`（test_files/ 29,090枚 + evaluation_target.csv）
- 0フレーム/読込失敗/未学習衛星は気候値 `global_mean_fallback`(0.2886) フォールバック。
- 目安: eval入力 約29k枚読込で **〜5〜10分**。

### 4. 提出形式の検証
```bash
# Phase1 の検証スクリプトを流用（zip名だけ変える）。phase2_gbdt.zip を見るよう編集 or 下記ワンライナー
uv run python -c "
import zipfile,rasterio,numpy as np,sys
sys.path.insert(0,'src'); from precip import config
z=config.SUBMISSIONS_DIR/'phase2_gbdt.zip'
names=zipfile.ZipFile(z).namelist()
tif=[n for n in names if n.endswith('.tif')]
import pandas as pd; ev=pd.read_csv(config.EVAL_CSV)
zs={n.split('/')[-1] for n in tif}; es=set(ev['gpm_imerg_filename'])
print('tif数',len(tif),'CSV同梱',('evaluation_target.csv' in names))
print('ファイル名集合一致',zs==es,'過不足',len(zs-es),len(es-zs))
"
```
期待: tif数=29090、ファイル名集合一致=True。

---

## 出力物まとめ
| ファイル | 内容 |
|---|---|
| `outputs/phase2_features_{sat}.parquet` | 衛星別 サンプリング済み画素特徴（fold,y,63特徴） |
| `outputs/phase2_cv.json` | two_part/tweedie の CV（overall/衛星別/強雨条件付き/bias）と採用variant |
| `outputs/phase2_selected.json` | 採用variant・特徴名・モデルパス |
| `outputs/phase2_models/` | 衛星別 LightGBM Booster（two_part: _clf/_reg, tweedie: _tweedie） |
| `outputs/phase2_feature_importance.csv` | 衛星×submodel×特徴×gain（Phase3/特徴選択へ還元） |
| `submissions/phase2_gbdt.zip` | 提出zip |

## 速く回す（反復用の設定変更, `conf/config.yaml` の `gbdt:`）
- `variant: two_part`（tweedie比較を省き半分の時間）
- `lgb_classifier.n_estimators` / `lgb_regressor.n_estimators` を 150〜200 に下げる
- `pixel_sample_frac: 0.15` / `max_pixels: 3000000`（学習画素削減）
- `n_workers` をコア数に合わせる

## 結果のフィードバックでほしい情報
- `outputs/phase2_cv.json` の overall / per_satellite / cond_rmse_ge1,ge5 / bias（two_part vs tweedie）
- 採用variant、Phase1(1.1644)からの改善幅
- （提出したら）Public LB スコア → CV との相関を一緒に評価
- `outputs/phase2_feature_importance.csv` 上位特徴（空間特徴が効いているか＝GBDTで空間文脈を取れているかの判断材料）

## 設計メモ（実装の前提）
- **衛星別モデル**（3つ）。各地域は単一衛星対応・eval地域もDISJOINTなので per-satellite が素直。
- **2部構成の予測 = P(rain)·E[y|rain]**（期待値, RMSE最適の近似）。学習画素は base rate維持の一様サンプリング（確率を歪めない）。有雨回帰は y≥threshold のみ・log1p。
- **CV評価は一様サンプル画素上の unbiased 推定**（全画素RMSEの不偏推定）。強雨条件付きは裾の挙動監視用。
- 特徴・予測は `src/precip/features.py` に一元化（学習/予測で厳密一致）。
