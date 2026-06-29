# 調査サーベイ：衛星データからの降水量推定

> 「宇宙からの降水ナウキャスト」（Solafune）に向けた、**静止衛星マルチスペクトル画像 → GPM-IMERG 降水量（mm/hr, RMSE）** の手法調査と戦略。
> 背景：気候変動下で激甚化する鉄砲水に対し、地上レーダーに依存しない**衛星のみ・遅延ゼロ・地域汎化**する降水推定への需要が高まっている。本サーベイはその実現に必要な特徴量設計と推定手法を体系化する。

## 構成

| ファイル | 内容 | 一言で |
|---|---|---|
| [`01_input-features.md`](./01_input-features.md) | 回帰のための**入力変数・特徴量エンジニアリング** | 何を入力するか：IR窓Tb・BTD・VIS/NIR・空間/時間/微物理特徴 |
| [`02_estimation-methods.md`](./02_estimation-methods.md) | **推定手法**（古典→ML→DL） | どう推定するか：経験式・GBDT・U-Net・時空間DL・損失設計 |
| [`03_strategy.md`](./03_strategy.md) | **コンペ戦略**とロードマップ | どう勝つか：段階的ロードマップ・CV設計・後処理 |

データ仕様は [`../dataset/data-specification.md`](../dataset/data-specification.md)（16バンドの意味つき）を参照。

## 3つの核心メッセージ

1. **2段階構成（rain/no-rain 判別 → 強度回帰）がほぼ必須**。降水は「ゼロ過剰×ロングテール」で、これを分けて扱う hurdle/2段階が古典（SCaMPR・GMSRA）から最新DL（Oya 2025・Hurdle-IMDL 2025）まで一貫して有効。

2. **多バンド（特に水蒸気バンドと BTD）が単一 IR を大きく上回る**。冷たい雲頂＝雨ではない（巻雲問題）。WV−IR・split-window・位相・粒径の BTD と空間文脈で巻雲を除き、温暖型大雨も捕捉する（Hirose 2019・Oya 2025）。

3. **外部データ禁止が効く**。古典運用法が頼る NWP（可降水量・湿度）や DEM（地形）は本コンペで使用不可。その分を **WV マルチバンド＋空間/時間特徴で内製**できるかが順位を分ける。

## 推奨アプローチ（要約）

```
Phase0 EDA/疎通 → Phase1 Tb-RR経験式 → Phase2 GBDT 2部(本命) →
Phase3 U-Net 2段階(主力) → Phase4 時間特徴/3フレーム → Phase5 アンサンブル&後処理
```

まず **LightGBM/CatBoost の2部モデル**で堅実な土台と特徴知見を得て、**U-Net**で空間文脈、**3フレーム**で時間発展を積み増し、最後に**アンサンブル＋後処理（負値クリップ・ゼロ閾値・強雨較正）**で RMSE を詰める。検証は**地域 GroupKFold** で汎化を担保（Private 100% が最終順位）。

## 主要参考（抜粋）
- Hirose et al. (2019) Himawari-8 多バンド Random Forest（HRA）
- Oya (2025) 2段階 U-Net・全 VIS/IR チャネル・LDS 重み
- Hurdle-IMDL (2025) ゼロ過剰／ロングテールの分解学習
- NOAA Hydro-Estimator・SCaMPR・GMSRA（運用アルゴリズムと特徴量設計）
- GOES-16 → IMERG-ER U-Net (2025)・Huayu (2025, FY-4B IR 単独リアルタイム)

各ドキュメント末尾に URL 付き出典一覧あり。
