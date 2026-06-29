# サーベイ② 推定手法（古典アルゴリズム → 機械学習 → 深層学習）

> 静止衛星 → GPM-IMERG 降水量（mm/hr, RMSE 評価）の回帰手法を、(A) 物理・経験式の運用アルゴリズム、(B) タブ型機械学習、(C) CNN/U-Net 系、(D) 時空間深層学習、(E) 不均衡対策・損失設計、の5層で整理する。
> 入力特徴は [`01_input-features.md`](./01_input-features.md)、勝ち筋は [`03_strategy.md`](./03_strategy.md)。

---

## 0. 手法マップ（全体像）

```
                     精度↑ / 計算コスト↑ / データ要求↑
  物理・経験式 ──→ タブ型ML ──→ CNN/U-Net ──→ 時空間DL（ConvLSTM/Transformer/Diffusion）
  (GPI,A-E,H-E,   (RF, LightGBM   (Oya, PERSIANN  (MetNet系, Earthformer,
   GMSRA,CCS,      /CatBoost,      -CNN, GK2A,      PreDiff, ConvLSTM)
   SCaMPR,CRR)     hurdle)         U-Net IMERG-ER)
        │              │               │                  │
   解釈性・即運用   特徴量設計が物言う  空間文脈を自動学習   時間発展・確率予測
```

共通する設計原則は **2段階（rain/no-rain 判別 → 強度回帰）**。ゼロ過剰のため、ほぼ全ての成功手法がこの分離を明示的・暗黙的に行う。

---

## A. 古典・運用アルゴリズム（物理・経験式）

「外部データ禁止」の本コンペでそのまま使えるものは少ないが、**特徴量設計と2段階構成の発想源**として極めて有用。ベースラインや後処理ルールの着想にも使える。

| アルゴリズム | 入力 | 核心ロジック | 本コンペへの示唆 |
|---|---|---|---|
| **GPI**（GOES Precipitation Index） | IR 窓 Tb のみ | Tb<235K の画素に固定降水率を割当て（面積×時間） | 最単純ベースライン。閾値の発想 |
| **Auto-Estimator**（Vicente 1998） | IR 窓 Tb（＋NWP の PW/RH） | RR=f(T10.7) のべき乗則＋水分・地形・天頂角補正。連続画像の Tb 変化で巻雲除去 | 「Tb→RR べき乗則」「時間変化で巻雲除去」の原型 |
| **Hydro-Estimator**（2002〜） | IR 窓 Tb（＋NWP） | 画素 vs 近傍の **Z スコア**で対流コア/非コア/巻雲を分離。マルチスケール半径。視差・天頂角補正 | **空間文脈で巻雲を消す**設計の教科書。特徴量に直輸入可 |
| **GMSRA**（Ba & Gruber 2001） | VIS, 3.9, 6.7, 11, 12μm | IR−WV<0 で深い対流の降雨域同定、3.9μm 粒径と勾配で巻雲除去、Tb 10K ビンごとに降水確率×平均降水率 | **マルチスペクトル＋確率×強度**の2段階。BTD 設計の源 |
| **PERSIANN-CCS** | IR（10.7μm） | 可変閾値で雲パッチ分割→coldness/geometry/texture で 400 雲型分類→雲型ごとに Tb-RR 関係 | 雲パッチ・テクスチャ特徴の有効性。分類→回帰 |
| **SCaMPR**（Kuligowski） | VIS/IR 複数予測子 | MW 降水率を較正ターゲットに、判別分析で rain/no-rain＋前進ステップ回帰で強度。自己較正 | **2段階（判別→回帰）＋予測子自動選択**。GOES 運用 rain-rate の基盤 |
| **CRR-Ph**（MSG/NWCSAF） | 雲微物理（COT, reff, phase） | 雲物理量から対流性降水率 | 微物理特徴の有効性 |
| **GSMaP / IMERG** | MW＋IR ブレンド | MW で高精度、IR で時空間補間（morphing） | ターゲット IMERG の生成原理の理解に重要 |

要点：これらは **NWP・DEM を前提**にしている部分が多く、そのままは本コンペで使えない。だが「Tb→RR の非線形性」「空間 Z スコア」「BTD による巻雲除去」「確率×強度の2段階」という**設計知見**は、ML 特徴量・損失・後処理にそのまま移植できる。

---

## B. タブ型機械学習（画素単位回帰）★最初の本命ベースライン

各画素を 1 サンプルとし、§01 の特徴量（生バンド＋BTD＋空間統計＋メタ）から降水率を回帰する。プロジェクトの依存（`lightgbm`, `catboost`, `scikit-learn`）と整合し、**少ない実装コストで強い**。

### B.1 ランダムフォレスト（RF）
- **HRA（Hirose 2019）**：Himawari-8 の 9 IR バンド＋36 BTD を RF に投入し、0.04°・10分の降水率を推定。3ステップ（rain/no-rain 分離 → 降水カテゴリ分類 → 強度推定）。MW 不在時間帯でも GEO 単独で温暖型大雨を捕捉。
- **物理拡張 RF**：対流初生ナウキャストで「物理特徴で拡張した RF」が GOES/FY-4 で成果（Yang 2024 等）。

### B.2 勾配ブースティング（LightGBM / CatBoost / XGBoost）
- 表形式で **RF より高精度・高速**になりやすく、欠測・カテゴリ（衛星種別）も扱いやすい。
- 降水×衛星マージで LightGBM が広く使われ、**分位点回帰**で不確実性・極端値も扱える（Papacharalampous 2023 等）。
- **2部（hurdle）構成**：①分類器（降る確率）×②有雨画素のみで強度回帰、を別々の GBDT で。RMSE 最適化と相性が良い。
- 目的関数：`y` を log1p 変換、または Tweedie 損失（ゼロ過剰連続値に適合）を直接使う手も。

### B.3 タブ ML の長所・短所
- ◎ 実装容易・学習高速・解釈性（feature importance）・少データで安定・3衛星混在に強い。
- △ **空間文脈は手作業で特徴展開が必要**（近傍統計を作り込む必要）。画像の大局構造は CNN に劣る。

> 戦略的位置づけ：**まず GBDT 2部モデルで堅実な LB を取り**、特徴量の効き方を feature importance で把握 → その知見を CNN へ持ち込むのが王道。

---

## C. CNN / U-Net 系（単時刻・空間文脈を自動学習）

画像 → 降水マップの **画像対画像回帰**。近傍統計を手で作らずとも、畳み込みが空間文脈（対流コア vs 巻雲）を学習する。

| 研究 | 構成 | 要点・数値 |
|---|---|---|
| **Oya**（2025, Google/NASA） | **2つの U-Net**（①降水検出分類器＋②log 降水量回帰）。全 VIS/IR チャネル入力。GPM CORRA を真値、IMERG で事前学習 | 全チャネル＞IR 窓単独（CSI 全帯で +5〜10pt）。log 変換＋**LDS による逆密度重み**で不均衡対策。リムダーケニングが高緯度劣化要因 |
| **GOES-16 → IMERG-ER U-Net**（2025） | U-Net CNN、GOES-16 IR バンドの最適組合せを選択 | RMSE **0.46 mm/h**、CSI 0.53。低強度（<3mm/h）は良好、**高強度はデータ不均衡で苦戦** |
| **PERSIANN-CNN** | CNN で雲特徴抽出→降水 | 古典 CCS の DL 化 |
| **GK2A 降水マップ**（2024） | 韓国 GEO（GK2A）から DL で降水マップ生成 | GEO 単独・多チャネル DL の有効性 |
| **SmaAt-UNet / RainNet** | 軽量 attention U-Net 等 | 効率重視の画像回帰（ナウキャスト文脈で多用） |

CNN の鍵：
- **2段階を分けるか**（Oya のように分類 U-Net＋回帰 U-Net）／単一ネットで連続値（ゼロ含む）回帰するか。前者が不均衡に強い。
- **マルチチャネル入力**で多バンドを活かす（IR 窓のみは弱い）。
- **パッチ学習**：ROI を固定サイズパッチに切り、ミニバッチ化。境界・欠測のマスク処理。

---

## D. 時空間深層学習（直近3フレームの時間発展を使う）

本コンペは「直近30分・最大3フレーム」入力なので、時間発展を明示モデル化できる。ナウキャスト文献の手法が応用可能。背景で言及される MetNet-3・Pangu はこの系統。

| モデル | 特徴 | 本コンペ適性 |
|---|---|---|
| **ConvLSTM / Convcast** | CNN＋LSTM で時空間。降水ナウキャストの定番 | 3フレームを系列入力。実装しやすい |
| **U-Net + ConvLSTM**（IMERG nowcasting 2024） | U-Net 構造に ConvLSTM セル、GFS ドライバ併用で 4h リード | 時空間統合の代表。ただし GFS は外部データ＝本コンペ不可 |
| **MetNet / MetNet-2/3** | 空間ダウンサンプル＋ConvLSTM エンコーダ＋軸方向 attention | 大規模・高性能だが計算重い。衛星が長リードで効くと報告 |
| **Earthformer** | 時空間 Transformer（cuboid attention） | 長距離依存に強い。データ・計算を要する |
| **PreDiff / 拡散モデル** | 潜在拡散で確率的ナウキャスト | 極端値・不確実性に有利だが RMSE 最小化には過剰なことも |
| **3D U-Net diffusion（GEO IR）2025** | GEO IR 輝度温度の決定論的ナウキャスト | 衛星画像系列の時間外挿 |

注意：時空間 DL は強力だが、**RMSE 単一指標・3フレームのみ・外部 NWP 不可**という制約では、過剰投資になりやすい。まず単時刻 CNN を固め、時間情報は「特徴（dTb/dt 等）」や「軽い時間 stack（3フレームをチャネル方向に連結して 2D-CNN）」で取り込み、本格的 ConvLSTM/Transformer は終盤の伸びしろとするのが堅実。

---

## E. 不均衡対策・損失設計（RMSE を直接動かす最重要レバー）

降水データの「ゼロ過剰 × ロングテール」をどう損失で扱うかが、RMSE を最も大きく左右する。**Hurdle-IMDL（2025）**の分解整理が指針になる。

### E.1 問題の分解（Hurdle-IMDL）
- **ゼロ過剰（zero inflation）**：無降水が支配的 → **hurdle（2部）モデル**で「降る/降らない」を分離。
- **ロングテール（long tail）**：弱い雨が多く強雨が希少 → **debiasing/再重み付け**（系統的過小評価の補正）。

### E.2 具体的手法

| 手法 | 内容 | 効果・注意 |
|---|---|---|
| **2部 / hurdle モデル** | 分類（P(rain)）×回帰（強度｜rain） | ゼロ過剰の王道。Oya・HRA・SCaMPR が採用 |
| **log/log1p ターゲット変換** | 右歪みを対称化、評価時に逆変換 | 回帰安定。RMSE は元スケールで効くので逆変換後の偏りに注意 |
| **Tweedie 損失** | ゼロ過剰連続値を直接モデル化 | GBDT で 1 段でゼロ過剰回帰が可能 |
| **重み付き MSE** | 降水画素に高重み（例 w=3）、無降水に低重み（w=1） | 強雨を取りこぼさない |
| **逆密度重み（LDS）** | ラベル密度の逆数で重み付け（Oya） | 強雨・極端雨の精度を改善、弱雨も維持 |
| **focal / Tversky（分類側）** | 偽陰性に重いペナルティ | rain/no-rain 判別の再現率向上 |
| **分位点回帰** | 複数分位を予測 | 不確実性・極端値。LightGBM で容易 |

### E.3 RMSE 特有の注意
- RMSE は **大外しの強雨**を二乗で罰する一方、**画素数の多い無降水**の系統誤差も効く。→ 「強雨を当てる」と「無降水でゼロを返す」の両立が肝。
- **閾値の最適化**：2部モデルの rain/no-rain 閾値や、微小予測値のゼロ丸めは、検証データで RMSE 最小化するよう調整。
- **負値クリップ**：物理的に降水≥0。後処理で負を 0 に。
- **過小評価の補正**：log 回帰や重み無しだと強雨を系統的に過小評価しがち（Hurdle-IMDL が指摘）。較正（isotonic 等）や逆密度重みで補正。

---

## F. 手法選定マトリクス（本コンペ視点）

| 手法 | 精度ポテンシャル | 実装/計算コスト | 不均衡耐性 | 3衛星混在耐性 | 推奨フェーズ |
|---|---|---|---|---|---|
| 経験式ベースライン（GPI/Tb-RR） | 低 | 極小 | 低 | 中 | 即・基準値 |
| GBDT 2部（LightGBM/CatBoost） | 中〜高 | 小 | 高（hurdle） | 高 | **本命ベース** |
| 単時刻 CNN/U-Net（2段階） | 高 | 中 | 中〜高 | 中 | 主力 |
| 時間 stack 2D-CNN（3フレーム連結） | 高 | 中 | 中 | 中 | 主力＋ |
| ConvLSTM/Transformer/Diffusion | 高（要データ） | 大 | 中 | 中 | 終盤の伸びしろ |
| アンサンブル（GBDT＋CNN） | 最高 | 中〜大 | 高 | 高 | 仕上げ |

---

## 参考文献・出典

- NOAA STAR：Hydro-Estimator Technique — <https://www.star.nesdis.noaa.gov/smcd/emb/ff/HEtechnique.php> ／ SCaMPR — <https://www.star.nesdis.noaa.gov/smcd/emb/ff/SCaMPR.php>
- Ba & Gruber (2001) GMSRA — <https://journals.ametsoc.org/jamc/article/40/8/1500/16163/GOES-Multispectral-Rainfall-Algorithm-GMSRA>
- Hirose et al. (2019) Himawari-8 Random Forest（HRA）— <https://www.jstage.jst.go.jp/article/jmsj/97/3/97_2019-040/_article>
- PERSIANN-CCS（CHRS）— <https://chrs.web.uci.edu/SP_activities01.php>
- Oya: Deep Learning for Accurate Global Precipitation Estimation (2025) — <https://arxiv.org/html/2511.10562>
- Advancing timely satellite precipitation for IMERG-ER using GOES-16 and U-Net (2025) — <https://www.sciencedirect.com/science/article/abs/pii/S1364815225001410>
- Global Precipitation Nowcasting of IMERG: U-Net ConvLSTM (2024) — <https://journals.ametsoc.org/view/journals/hydr/25/6/JHM-D-23-0119.1.xml>
- Hurdle-IMDL: An Imbalanced Learning Framework for Infrared Rainfall Retrieval (2025) — <https://arxiv.org/abs/2510.20486>
- Huayu: Advanced Real-Time Precipitation Estimation from Geostationary Satellite (2025) — <https://arxiv.org/abs/2512.15222>
- Deep learning for precipitation nowcasting: A survey (2024) — <https://arxiv.org/html/2406.04867v1>
- Earthformer — <https://arxiv.org/pdf/2207.05833> ／ PreDiff — <https://arxiv.org/pdf/2307.10422>
- LightGBM で降水マージ・分位点 — <https://arxiv.org/pdf/2302.03606>, <https://arxiv.org/pdf/2311.07511>
- GK2A 降水マップ DL (2024) — <https://www.mdpi.com/2072-4292/18/2/188>
