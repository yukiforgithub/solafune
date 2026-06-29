# EDA §50 — Phase 0 ベースライン & 提出疎通

> Phase 0（EDA & パイプライン疎通）の締め。本セクションは
> (a) **定数ベースラインの CV RMSE**（全0 / 全体平均 / 地域別平均 × 2 CV スキーム）を確定し、
> (b) 前処理→学習→予測の **3モジュールをエンドツーエンドで通し**、
> (c) 実 EVAL 用の **妥当な提出 zip を生成して sample_submission と形式照合**する。
> 数値はすべて `eda_cache/target_stats.parquet`（TRAIN 各ターゲット tif の per-file 集計、40,686 行）から
> **閉形式で算出**し、生 tif の spot-check（8 件）で突き合わせ検証済み。

---

## TL;DR（結論）

- **主CV（地域ホールドアウト GroupKFold, 5fold）では `地域別平均` は `全体平均` に完全縮退**（ともに overall RMSE **1.4048**）。
  検証 fold の地域は学習 fold に存在しない（DISJOINT）ので、地域別平均は全行が全体平均へフォールバックするため。
- **比較用のランダム KFold（楽観参照）では地域情報が効く**：`地域別平均` **1.3836** < `全体平均` **1.4030** < `全0` **1.4324**。
  この `1.4030 → 1.3836`（−0.0194）の改善は「val 地域も train に居れば」得られる値であり、**本番（未知地域）では使えない**。
  ランダム CV が未知地域汎化を過大評価する具体的な大きさがこれ。
- **EVAL 地域は train と完全 DISJOINT** なので、提出ベースラインは **全体平均 `c = 0.2886` の定数**を採用（地域別平均を使っても全行フォールバックで同値）。
- **提出 zip を生成し sample_submission と完全照合 → `format_valid = True`**。
  ファイル名集合一致（29,090 枚）、全 tif が `1band × 41×41 × float32 / CRS無し / NaN・負値無し`、同梱 CSV は EVAL CSV と完全一致。
  - 成果物: `submissions/baseline_global_mean.zip`（9.6 MB）。

---

## (a) 定数ベースラインの CV RMSE

### A-1. 結果表（手法 × CV スキーム → overall RMSE）

| 手法 | 地域 GroupKFold（主・honest） | ランダム KFold（参照・楽観） |
|---|--:|--:|
| 全0 (`zero`) | 1.4324 | 1.4324 |
| 全体平均 (`global_mean`) | **1.4048** | 1.4030 |
| 地域別平均 (`location_mean`) | **1.4048**（=全体平均へ縮退） | **1.3836** |

- 全 OOF 画素を連結した overall RMSE（各 fold の SSE を画素数で重み付けして合算）。`n_splits=5, seed=42`。
- **`全0` は CV スキームに不変**（1.4324）。予測が定数 0 なので分割に依存しない＝`sqrt(Σvsq / Σnpix)`。
- 主CV の `全体平均` が full-train の 1.4030 より僅かに高い（1.4048）のは、各 fold の平均を**その fold の学習部分集合**で推定するため。20 地域を 5 fold に割ると地域構成の偏りで fold 平均が真の全体平均からずれ、定数予測に小さな汎化ギャップが出る。

### A-2. fold 別 RMSE（参考）

**地域 GroupKFold（主）** — fold 間のばらつきが大きい（地域ごとの気候差がそのまま出る）:

| 手法 | fold0 | fold1 | fold2 | fold3 | fold4 | overall |
|---|--:|--:|--:|--:|--:|--:|
| 全0 | 1.7200 | 1.4670 | 1.5493 | 1.1285 | 1.1682 | 1.4324 |
| 全体平均 | 1.6828 | 1.4238 | 1.5147 | 1.1223 | 1.1627 | 1.4048 |
| 地域別平均 | 1.6828 | 1.4238 | 1.5147 | 1.1223 | 1.1627 | 1.4048 |

→ **地域別平均の per-fold が全体平均と 1 桁まで完全一致**しているのが「縮退」の直接証拠。

**ランダム KFold（楽観参照）** — fold 間が均質（地域が train/val に跨るため）:

| 手法 | fold0 | fold1 | fold2 | fold3 | fold4 | overall |
|---|--:|--:|--:|--:|--:|--:|
| 全0 | 1.4377 | 1.4795 | 1.4480 | 1.4166 | 1.3780 | 1.4324 |
| 全体平均 | 1.4086 | 1.4490 | 1.4178 | 1.3877 | 1.3500 | 1.4030 |
| 地域別平均 | 1.3887 | 1.4301 | 1.3976 | 1.3682 | 1.3317 | 1.3836 |

### A-3. なぜ地域別平均が主CVで全体平均に縮退するか

`train.py` の地域別平均は **学習 fold の地域→平均**を辞書化し、**辞書に無い地域（=未知地域）は全体平均へフォールバック**する。
主CV（GroupKFold, group=`name_location`）では検証 fold の全地域が学習 fold に存在しないため、**検証行は 100% フォールバック**＝全体平均と同一予測になる。
これは欠陥ではなく「未知地域には地域固有の値が原理的に存在しない」という**本コンペの本質を CV が正しく反映**した結果。
EVAL でも全 18 地域が未知なので、提出は地域別平均でも全体平均でも同値であり、後者を採用した（意味的に明示的）。

### A-4. 算出方法と検証（閉形式 + spot-check）

per-file の `vsum`（画素値合計）・`vsq`（二乗和）・`npix`（=1681）から、定数 `c` の SSE を **tif を開かずに**復元できる:

```
SSE = Σ_files ( vsq − 2·c·vsum + c²·npix )
RMSE = sqrt( SSE / Σ npix )
```

- 40,686 ファイルの再読込（数十分規模）を回避し、キャッシュ集計から厳密値を得る設計。
- **生 tif 8 件で突き合わせ**：`vsum / vsq / npix / eq0 / ge01` がすべて誤差 < 1e-2 で一致（ALL MATCH）。
- **既存事実とも一致**：全画素ゼロ率 0.8207、`<0.1mm` 0.8515、画素平均 0.2886、最大 96.51、全0 RMSE 1.4324、全体平均 RMSE 1.4030（§10・§30 と整合）。

> **CV 分割の実装メモ**: 本 Phase の `src/precip/cv.py` は「衛星別に地域をシャッフルしラウンドロビン配分」する seed 依存 GroupKFold を使う（§30 で提案した手設計の fold マップとは別実装）。
> 定数予測は分割の細部に不感（縮退と overall 1.4048 はどのバランスでも成立）なので Phase 0 では問題ない。
> モデルを載せる Phase 2 以降で、§30 の `name_location→fold` 手設計マップを `conf/` に固定して使う運用へ移行する。

---

## (b) エンドツーエンド疎通（3モジュール）

コンペ要件の「前処理 / 学習 / 予測」3分割をそのまま実行し、全段が通ることを確認:

| 段 | スクリプト | 実行結果 |
|---|---|---|
| 前処理(train) | `src/preprocess_train.py` | 40,686 行 / 20 地域 / 3 衛星、フレーム数分布 `{0:235, 1:8, 2:647, 3:39796}` を検証。`outputs/preprocess_train_summary.parquet` 出力。 |
| 前処理(test) | `src/preprocess_test.py` | 29,090 行 / 18 地域 / 3 衛星、`{0:29, 1:8, 2:567, 3:28486}`、`gpm_imerg_filename` 一意を検証。`outputs/preprocess_test_summary.parquet` 出力。 |
| 学習 | `src/train.py` | per-file 集計から CV RMSE を全手法・両スキームで算出。`outputs/baseline_model.json`（method/global_mean/地域別平均）・`cv_scores.json`・`folds.parquet` を保存。 |
| 予測 | `src/predict.py` | EVAL 29,090 行に定数を当て 41×41 float32 tif を生成、`evaluation_target.csv` を同梱して zip 化。 |

再現コマンド:

```bash
uv run python src/preprocess_train.py
uv run python src/preprocess_test.py
uv run python src/train.py  --cv-scheme location_group --method global_mean   # 主CV（保存される CV スコア）
uv run python src/train.py  --cv-scheme random         --method location_mean # 楽観参照（A-1 右列）
uv run python src/predict.py --name baseline_global_mean --method global_mean # 提出生成
```

---

## (c) 提出と sample_submission との形式照合

**生成物**: `submissions/baseline_global_mean.zip`（9.6 MB）
内訳: `evaluation_target.csv`（EVAL CSV のコピー）+ `test_files/`（29,090 枚の 41×41 float32 tif、全画素 = 0.2886）。

**照合結果（`format_valid = True`、全チェック通過）**:

| # | チェック | 結果 |
|--:|---|---|
| 1 | zip 内 tif 数 = EVAL 行数 = 29,090、ファイル名集合が EVAL `gpm_imerg_filename` と完全一致 | OK |
| 2 | sample_submission `test_files/` の名集合（29,090）とも完全一致 | OK |
| 3 | 全 29,090 tif の profile = `GTiff / float32 / count=1 / 41×41 / CRS None`（profile 不一致 0 件） | OK |
| 3' | 画素サンプル 200 枚: 形状 `(1,41,41)` / dtype float32、NaN・負値 0 件 | OK |
| 4 | 同梱 `evaluation_target.csv` が EVAL CSV と列・行・内容まで完全一致 (29090×6) | OK |

- 提出 tif の profile は sample_submission の tif と一致（`transform = Affine.identity()`、`nodata = None`）。書き出し時の `NotGeoreferencedWarning` は単位行列由来の GDAL 警告で、sample_submission も同条件のため無害。
- 後処理として負値→0 クリップを出力直前に適用（定数は非負なので実害なしだが配線として明示）。

---

## 確定事項（後続フェーズへの引き継ぎ）

1. **超えるべき原点**: 未知地域汎化の honest なベースライン = **RMSE 1.4048**（地域 GroupKFold, 定数）。
   ランダム CV で出る 1.3836 は楽観値で、**LB（Private）の目安には 1.40 台前半を採る**。モデルはこれを下回って初めて意味がある。
2. **地域別平均は単独では eval に効かない**（全 18 地域が未知 → 全体平均へ縮退）。地域差を当てるには **入力衛星画像から地域非依存に降水を推定**する必要がある（=本コンペの主題）。
   気候値フォールバック（§30 (b)）も、未知地域では「地域別」ではなく**衛星×季節×時刻**などの転移可能な軸で組む。
3. **提出パイプラインは確定**: `predict.py` 1 本で `gpm_imerg_filename` 完全一致・正しい profile の zip を生成でき、形式検証も通過。Phase 2 以降はこの予測値を定数からモデル出力へ差し替えるだけ。
4. **CV 実装の移行**: Phase 0 の seed 依存 GroupKFold → Phase 2 以降は §30 の手設計 `name_location→fold` マップを `conf/` に固定し、衛星・行数・降水強度のバランスを担保する。
