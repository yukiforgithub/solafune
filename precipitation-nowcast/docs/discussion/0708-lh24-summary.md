# ディスカッション要約: LH24「0.708 public LB + clean modular code」

> 出典: Solafune コンペ ディスカッション（投稿者 LH24、PDF: `docs/discussion/Sharing a 0.708 public LB submission and clean, mo... - Solafune.pdf`）
> 本ファイルは投稿内容の要約と、我々の Phase 0 EDA（[`docs/eda/README.md`](../eda/README.md)）との突き合わせ。

## TL;DR（我々への含意）

1. **バンド選択がモデルサイズより効く** ── 可視のみ(1.286)→split-window(0.872) で **−0.41** 改善。これが最大のレバー。
2. **小さな U-Net（1.9M params, 9ch = 3バンド×3フレーム, log1p損失, 5fold地域CV+アンサンブル）で public LB 0.708**。安価で再現可能な到達目標が判明。
3. 我々の Phase 0 の主要結論（ゼロ率~82%、地域ホールドアウトCV、log1p、IR窓+WV中核、強雨の裾がRMSE支配）は **第三者の独立分析と一致** → 戦略の方向性が裏付けられた。
4. **我々の上回り筋**: 彼は単一U-Net。我々の戦略の **2部構成(rain/no-rain分類×有雨回帰) + 強雨のisotonic較正** は、彼が報告した「弱雨で過大・強雨で過小」を直接叩ける差別化点。

---

## 投稿の中身

### タスク認識（我々と一致）
41×41 GPM(mm/hr) を30分先予測、最大3フレーム×16バンド、衛星別解像度→ターゲットへリサンプル。

### EDA所見
- **ゼロ率 83%**（投稿; グラフ表記 83.0%、mean=0.251）。
- **地理ホールドアウト**: train/test 地域は設計上 disjoint。地域別平均降水は桁違いに変動。**最湿 = hat_yai(タイ)、最乾 = dhaka(バングラ)**。→ 地域グループCVを推奨。
- **3衛星でバンド対応が異なる** → 衛星別マッピング必須。
- **バンド重要度(Spearman ρ, 1500サンプル)**: IR系 ρ≈−0.3（冷たい雲頂=強雨で負相関）、WV上層(6.2µm) −0.221、可視赤(0.64µm) +0.129。**IR支配・可視ほぼゼロ**。昼間主導で夜間は可視が弱る。
- **Split-window差(BT10.4−BT12.3)**: 氷水経路/雲微物理の物理プロキシ。深い対流の氷晶で差が開く → dry/rainy で明瞭にシフト。
- **雲頂冷却率(BT_t0−BT_t2)**: 対流発達(冷却)が強雨に先行。訓練の **97.5% が3フレーム揃い**。

### モデル（0.708 の中身）
- 小型 U-Net（encoder 9→32→64→128 / bottleneck 256 / decoder skip付 / head→1）、**総params 1,928,737**。
- 入力 **9ch = 3バンド × 3フレーム**、出力 (1,41,41)。41→**48にパディング**(8で割れる)後クロップ。
- **損失 = log1p空間のMSE**: `loss=MSE(model(x), log1p(y))` / 推論 `pred=expm1(model(x)).clamp(0)`。評価は元mm/hr。
- **CV2モード**: holdout(4地域固定: florida/france/jakarta/kinshasa) と kfold(5fold地域ローテ)。**提出は5fold各モデルのraw予測を平均**(アンサンブル)。
- コード: Hydra(設定) + W&B(実験管理)、models registry(自作unet / timm_unet)、dataset.py に **5つの BAND_COMBOS**(visible / ir_classic / split_window / wv_moisture / ice_proxy)。単一GPU(4GB VRAMでも)動作。

### 結果
**Trivial baselines（彼の4地域holdout val, raw mm/hr RMSE）**
| 手法 | RMSE |
|---|---|
| 全0 | 1.322 |
| 訓練平均 | 1.298 |
| 公式CNN(可視, MSE, **public LB**) | 0.913 |

**バンド比較（U-Net 1.9M, log1p, holdout val 4地域）**
| バンド組合せ | Val RMSE |
|---|---|
| Visible（公式手法） | 1.2857 |
| IR classic | 0.8859 |
| WV moisture | 0.8825 |
| Ice proxy(3.9µm) | 0.8809 |
| **Split-window** | **0.8724**（最良） |

**強雨依存の誤差（split-window）** — 弱雨で過大・強雨で過小、裾がRMSE支配
| 真値(mm/hr) | 画素% | RMSE |
|---|---|---|
| 0–0.1 | 88.1% | 0.230 |
| 0.1–0.5 | 4.7% | 0.630 |
| 0.5–1 | 2.3% | 0.814 |
| 1–2 | 2.1% | 1.162 |
| 2–5 | 2.0% | 2.322 |
| **>5** | 0.9% | **7.968**（全体の約9倍） |

**衛星別（val RMSE / zero%）**: Himawari 1.126/68.0% ・ GOES 1.165/75.9% ・ **Meteosat 0.757/89.3%**（乾燥で容易）。

---

## 我々の Phase 0 EDA との突き合わせ

| 項目 | LH24 | 我々（全40,686件スキャン） | 評価 |
|---|---|---|---|
| ゼロ率 | 83% | **82.07%** | ほぼ一致（彼はサンプル、我々は全件） |
| 平均降水 | 0.251 | **0.2886** | 近い（サンプル差） |
| 最大値 | 77.6 | **96.5** | 我々が全件で裾を捕捉 |
| 最乾/最湿地域 | dhaka / hat_yai | **dhaka 99.77%ゼロ / hat_yai 全0RMSE2.60で最難** | **完全一致** |
| 衛星難易度 | Meteosat易 | **Meteosat平均0.134で最易** | **一致** |
| CV | 地域グループ推奨 | **地域GroupKFold主CV確定** | **一致** |
| 全0 / 平均 baseline | 1.322 / 1.298 | **1.4324 / 1.4030** | 値が違う→下記注意 |

> **数値差の注意**: 彼の 1.322/1.298 は **4地域(florida/france/jakarta/kinshasa)holdout val** 上、我々の 1.4324/1.4030 は **train全件画素**上。母集団が違うだけで矛盾ではない。我々の追跡用原点は **全件 1.4324/1.4030**（および地域honest 1.4048）を採用し、彼の数値は「別splitの参考」として扱う。

---

## 我々の戦略への反映（推奨アクション）

採用・前倒し:
- **[最優先] バンド選択を最初のレバーに。** 可視を捨て IR窓/WV/split-window を中核に。`BAND_COMBOS`（visible / ir_classic / split_window / wv_moisture / ice_proxy）を `conf/` と前処理に実装し、同条件でCV比較（[`03_strategy.md`](../survey/03_strategy.md) §01 Tier1 と整合）。
- **[Phase 3 短絡] 小型U-Net(9ch=3バンド×3フレーム, log1p MSE, 41→48パディング, 5fold地域CV+fold平均)** を再現 → これ自体が public ~0.708 級の足場。我々の §54 と一致、まず素早く再現してLB相関を取る。
- **特徴量追加**: split-window差(BT10.4−BT12.3)、雲頂冷却率(BT_t0−BT_t2)。GBDT(Phase2)とCNN入力チャネルの両方に。
- **提出はfoldアンサンブル平均**（holdout 0.87→LB 0.708 の差はアンサンブル＋全データ効果が大きい）。

我々の差別化（彼を上回る筋）:
- **2部構成(分類P(rain)×有雨log1p回帰)** + **強雨のisotonic較正**。彼の弱点（弱雨過大ドリズル/強雨過小、>5mmでRMSE7.97）を直接攻める。彼は単一U-Net・較正なし。
- **GBDT×CNN アンサンブル**（系統の異なるモデル混合）。
- 後処理（負値クリップ・ゼロ丸め閾値・上限クリップ）は彼も最小限 → ここで詰められる。

注意点:
- **timm 事前学習重みはライセンス個別確認**（CC0/CC-BY/MIT/BSD/Apache のみ可）。`timm_unet` を使うなら encoder の重みライセンスを確認・記録。
- W&B / Hydra はツールとして利用可（我々は Hydra 採用方針と整合）。
- 公開コードの直接利用は規約に注意。**手法・知見の参考に留め、実装は自前**（コンペ規約: 提供データの2次創作禁止、他者コードの最終提出不可）。
