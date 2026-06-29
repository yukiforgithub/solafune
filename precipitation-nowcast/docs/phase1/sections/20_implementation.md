# Phase1 §20 — 物理 IR モデルの3モジュール統合実装と eval 提出

> §10 で確定した最良モデル **`window_lookup / per_satellite`（CV RMSE 1.16437）** を、
> Phase0 の3モジュール構成（前処理 / 学習 / 予測）に統合し、EVAL 29,090 行の提出
> `submissions/phase1_physical_ir.zip` を E2E で生成・形式検証した記録。
> Phase0 の定数ベースライン（zero / global_mean / location_mean）は壊さず、`method` で分岐する。

---

## TL;DR

- **CV RMSE = 1.16437**（地域 GroupKFold 5fold, 手設計 fold, 充足統計からの閉形式再現）。
  §10 の確定値・fold 別（fold2=0.6976）・衛星別（meteosat 0.767 / goes 1.288 / himawari 1.451）と**完全一致**。
  全体平均原点 1.40606 から **gain +0.24169**。
- **提出 `submissions/phase1_physical_ir.zip`（約 72 MB, 29,090 tif）を生成し、SAMPLE_SUB と形式一致**を確認。
  ファイル名集合一致 / 各 tif (1,41,41) float32 非負 / 同梱 CSV バイト一致。`FORMAT_VALID=True`。
- EVAL 入力読込（最新フレームの IR 窓バンド）は `ThreadPoolExecutor`（8 worker）で並列化、**約 3 分**で完走。
  0 フレーム行 29 件は気候値 0.2886 へフォールバック。
- Phase0 定数ベースライン（`--method location_mean` → honest 1.4048）は不変で共存。

---

## (a) 実装の置き場所（Phase0 を壊さない拡張）

| ファイル | 追加 / 変更 | 役割 |
|---|---|---|
| `src/precip/physical.py` | **新規** | 物理 IR 推論器。特徴抽出（最新フレーム→窓 DN 41x41）・lookup 適用・負値0クリップ・気候値フォールバック。`PhysicalIRModel.from_json()` が `outputs/phase1_model.json` を引く |
| `src/precip/phase1_fit.py` | 追加 | `fit_window_lookup_per_satellite` / `cv_rmse_window_lookup_per_satellite`（手設計 fold で閉形式 CV）/ `satellite_means` |
| `src/precip/cv.py` | 追加 | `load_handdesigned_folds`（`conf/folds.yaml` の name_location→fold マップを読む。seed 非依存） |
| `conf/config.yaml` | 追加 | top-level `method: physical_ir` と `physical_ir:` ブロック（model_json / window_hist / folds_yaml / isotonic / global_mean_fallback / n_workers） |
| `src/preprocess_train.py` | 拡張 | `--method physical_ir` で物理特徴キャッシュ（`eda_cache/phase1_window_hist.parquet`）の存在確認 + 0フレーム件数 |
| `src/preprocess_test.py` | 拡張 | `--method physical_ir` で推論マニフェスト `outputs/preprocess_test_manifest.parquet`（最新フレーム basename を前処理で固定）を出力 |
| `src/train.py` | 拡張 | `method=physical_ir` で充足統計から refit + 閉形式 CV、`outputs/phase1_model.json` と `outputs/phase1_train_cv.json` を書く。定数 baseline 分岐は不変 |
| `src/predict.py` | 拡張 | `method=physical_ir` で EVAL に物理モデルを適用（ThreadPool 並列）し提出 zip を生成。定数 baseline 分岐は不変 |

**特徴抽出規約は §10 の充足統計構築（`phase1_suffstats.py`）と厳密一致**（学習/推論の前処理一致）:
最新（リスト最後）フレーム → IR 窓バンド（rasterio 1-based: himawari/goes=13, meteosat=14）を
`cv2.resize(..., INTER_AREA)` で 41x41 へ縮約 → `rint`+`clip(0,255)` で整数 DN → 衛星別 256bin lookup を引く →
負値 0 クリップ。バンド index 定数は `phase1_suffstats` から再利用する（二重定義しない）。

---

## (b) 実行コマンド（E2E）

```bash
# 0. 充足統計キャッシュ（未構築なら。窓 DN / split 差を1パスで集計）
uv run python -m src.build_phase1_suffstats

# 1. 前処理（学習側 / 評価側）— method は config の physical_ir を使用
uv run python src/preprocess_train.py --method physical_ir
uv run python src/preprocess_test.py  --method physical_ir

# 2. 学習 — 充足統計から refit + 手設計 fold で閉形式 CV、phase1_model.json を確定
uv run python src/train.py --method physical_ir

# 3. 予測 — EVAL 29,090 行に物理モデルを適用（ThreadPool 8 worker）し提出 zip 生成
uv run python src/predict.py --method physical_ir --name phase1_physical_ir

# 4. 形式検証（SAMPLE_SUB と照合）
uv run python scratch/validate_submission.py
```

> `conf/config.yaml` に `method: physical_ir` を設定済みのため、`--method` 省略でも physical_ir 系で動く。
> Phase0 定数ベースラインに戻すには `--method location_mean`（または config の method を constant に）。

### 学習ログ（再現値）

```
[train:physical_ir] CV RMSE overall=1.16437  per_fold=[1.2596, 1.2046, 0.6976, 1.2593, 1.3703]
[train:physical_ir] 衛星別 CV RMSE={'goes': 1.28837, 'himawari': 1.4513, 'meteosat': 0.76748}
[train:physical_ir] 全体平均原点 RMSE=1.40606  gain=0.24169
```

§10 の表と一致（n_pixels=67,996,450）。fold2 が低いのは france 支配（乾燥寄り）— §10 (b) のとおり分割の異常ではない。

---

## (c) 提出と形式検証

| 項目 | 値 |
|---|---|
| 提出 zip | `submissions/phase1_physical_ir.zip`（約 72 MB） |
| tif 枚数 | **29,090**（SAMPLE_SUB と同数） |
| ファイル名集合 | SAMPLE_SUB の `test_files/` と**完全一致**（zip のみ / SAMPLE のみ = 0 件） |
| 各 tif | `count=1` / `41x41` / `float32` / NaN・inf 無し / 負値無し |
| 予測値域（全 tif） | `[0.0000, 6.9273]`（最大は himawari の DN=0 ピーク 6.927 mm/hr と整合） |
| 同梱 `evaluation_target.csv` | 提供 EVAL CSV と**バイト一致** |
| 0 フレーム行 | 29 件 → 気候値フォールバック 0.2886 |
| 判定 | **`FORMAT_VALID=True`** |

検証スクリプト出力:

```
zip エントリ: tif=29090 csv=True
ファイル名集合一致: True (zip=29090 sample=29090)
tif 検証: checked=29090 bad=0 値域=[0.0000, 6.9273]
CSV バイト一致: True
FORMAT_VALID=True
```

提出形式は `dataio.write_prediction_tif`（GTiff / float32 / count=1 / 41x41 / CRS なし / nodata なし /
恒等アフィン）で書き出すため、SAMPLE_SUB と同一プロファイル。`NotGeoreferencedWarning` は
恒等変換に対する GDAL の情報通知で、提出形式上は問題ない（SAMPLE も CRS なし）。

---

## 成果物

| パス | 内容 |
|---|---|
| `src/precip/physical.py` | 物理 IR 推論器（特徴抽出 + lookup + フォールバック） |
| `outputs/phase1_model.json` | 確定 lookup（衛星別 256bin, satellite_mean 付与）。train.py が refit して上書き |
| `outputs/phase1_train_cv.json` | CV RMSE（overall / fold 別 / 衛星別）・gain・n_pixels |
| `outputs/preprocess_test_manifest.parquet` | EVAL 推論マニフェスト（最新フレーム basename を固定） |
| `submissions/phase1_physical_ir.zip` | EVAL 提出（29,090 tif + evaluation_target.csv） |
| `scratch/validate_submission.py` | 形式照合スクリプト（使い捨て） |

### 後続フェーズへの引き継ぎ

- 提出パイプラインが3モジュールで通った。**Phase2 GBDT / Phase3 CNN はこの `method` 分岐に新手法を足すだけ**で
  同じ predict→zip→検証フローに乗る（提出形式は `dataio.write_prediction_tif` に集約済み）。
- `preprocess_test.py` の推論マニフェストは、Phase2 で多変量特徴（窓 DN + split 差 + 時間差分 + 近傍統計）の
  EVAL 抽出を前処理側へ寄せる足場として再利用できる（`physical.extract_window_split_dn` も同梱済み）。
- 単バンド基準 = **CV RMSE 1.1644**。Phase2 以降はこれを下回るかで価値を測る。
