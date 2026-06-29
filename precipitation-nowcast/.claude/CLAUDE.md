# Precipitation Nowcast — Solafune コンペティション

> **👉 現在の進捗・引き継ぎは [docs/STATUS.md](../docs/STATUS.md) を最初に読むこと。**
> 直近のフェーズ結果・LB順位・実行コマンド・環境の制約(RAM 7.7GB/OOM対策)・次の一手・全ドキュメント地図が集約されている。

## プロジェクト概要

**コンペティション名**: 宇宙からの降水ナウキャスト（衛星データを用いた広域降水ナウキャスティング）
**プラットフォーム**: [Solafune](https://solafune.com)
**賞金**: $12,000 USD（1位 $4,000、2位 $2,500、3位 $2,000）

静止衛星データ（ひまわり・GOES・Meteosat）から直接、高精度なリアルタイム降水量を推定するモデルを構築する。地上レーダー不要で地域を越えて汎化することが求められる。

---

## タスク定義

**タスク種別**: 時系列回帰（Time-series Regression）

- **入力**: 直近30分の衛星観測画像（最大3枚、10分間隔） × 16バンド（マルチスペクトル）
- **出力**: GPM-IMERG 降水量 GeoTIFF（1バンド）
- **評価指標**: RMSE（小さいほど良い）

```
RMSE = sqrt( (1/n) * sum( (y_i - y_hat_i)^2 ) )
```

リーダーボードは Public（評価データの35%）と Private（100%）の2段階。**Private スコアが最終順位**。

---

## データ

### ファイル構成

| ファイル名 | サイズ | ファイル数 | 説明 |
|---|---|---|---|
| `train_dataset.zip` | 18 GB | 161,377 | 訓練データセット（表形式 + 画像） |
| `evaluation_dataset.zip` | 13 GB | 115,692 | 評価データセット（表形式 + 画像） |
| `sample_submission.zip` | 179 MB | 29,091 | 提出サンプル |

データは `data/` 以下に展開して使用する（gitignore 済み）。

### CSVフォーマット（train/evaluation 共通）

```
data_id                              : 各データの一意の識別子
name_location                        : 地域名 (例: kanto_region)
satellite_target                     : 衛星名 (himawari / goes / meteosat)
datetime                             : 観測日時
last_30_minutes_observation_filename : 入力衛星画像リスト（最大3ファイル）
gpm_imerg_filename                   : 予測対象ファイル名（ターゲット）
```

### 衛星データ仕様

| 衛星 | バンド数 | バンド名 |
|---|---|---|
| ひまわり8/9（Himawari） | 16 | B01〜B16 |
| GOES | 16 | C01〜C16 |
| Meteosat | 16 | vis_04, vis_05, vis_06, vis_08, vis_09, nir_13, nir_16, nir_22, ir_38, wv_63, wv_73, ir_87, ir_97, ir_105, ir_123, ir_133 |
| GPM-IMERG（ターゲット） | 1 | precipitation |

画像はすべてフルディスク画像から対象地域を切り出した GeoTIFF 形式。

---

## 提出フォーマット

```
your_submission.zip
├── evaluation_target.csv
└── test_files/
    ├── {location_X}_GPM_IMERG_{datetime_X}.tif
    ├── {location_X}_GPM_IMERG_{datetime_X}.tif
    └── ...
```

---

## コードの実装規約

入賞時の提出コードは3モジュールに分割すること:

1. **前処理** (`preprocess_train.py`, `preprocess_test.py`): 提供データを読み込み、モデル入力形式に変換してファイルを出力
2. **学習** (`train.py`): 前処理済みデータを読み込みモデルを学習、重みを保存
3. **予測** (`predict.py`): テストデータとモデルを読み込み、提出ファイルを出力

コンテナ化（Dockerfile）推奨。GPU 使用時は CUDA 11.8 以上。

---

## 制約・ルール

- **外部データセット禁止**（今回のコンペ固有の制約）
- 利用可能モデル・重み: CC0, CC BY, MIT, BSD, Apache 2.0 のオープンソースライセンスのみ
- 商用利用不可なモデルは使用不可
- 実装言語: Python, R 等のオープン且つ無料なツールのみ
- 1チーム1日の提出上限: チームメンバー数 × 5回（JST 9:00 リセット）
- チーム上限: 5人

---

## 開発環境

```toml
# pyproject.toml 主要依存
python = ">=3.13"
rasterio       # GeoTIFF 読み書き
scikit-learn   # ML ベースライン
lightgbm       # 勾配ブースティング
catboost       # 勾配ブースティング
opencv-python  # 画像処理
scikit-image   # 画像処理
pandas         # データ操作
matplotlib / seaborn  # 可視化
```

パッケージ管理: `uv`（`uv sync` で環境構築）

---

## ディレクトリ構成

```
precipitation-nowcast/
├── .claude/CLAUDE.md      # このファイル
├── data/                  # データ置き場（gitignore）
├── docs/project-desc/     # コンペ公式PDFドキュメント
├── notebooks/             # 探索・実験用Jupyter notebook
├── src/                   # ソースコード
├── conf/                  # 設定ファイル
├── main.py                # エントリーポイント
└── pyproject.toml
```

---

## 評価・スコア確認の注意点

- Public LB は評価データの約 **35%** で計算（過学習しないよう注意）
- **Private LB（全評価データ）が最終順位**
- 効率性スコア（Efficiency Score）は試験的指標で最終順位には影響しない

### 用語

- **LB** = **Leaderboard（リーダーボード）**: 提出時に運営の評価データで算出されるスコア/順位表。
  - **Public LB**: 評価データの約 **35%** で計算、開催中に表示。
  - **Private LB**: **全評価データ（100%）** で計算、終了後に確定する **最終順位**。
  - Public LB だけを見て調整すると Public への過学習になるため注意。
- **CV** = **Cross Validation（交差検証）**: 手元の訓練データを fold 分割して汎化性能を測る **ローカル評価**（提出不要）。

### CV 戦略

- **CV と LB の相関** を確認しながら進める。乖離が小さければ CV を信頼し、最終的には過学習しにくい CV を基準に Private LB に備える。
- 本コンペは **地域を越えた汎化** が問われるため、CV 分割は地域（`name_location`）や衛星（`satellite_target`）で **GroupKFold** 等のグループ分割を行い、未知地域への汎化を測ること。
