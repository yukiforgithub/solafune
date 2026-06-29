# データ仕様（各バンドの意味つき）

> 出典:
> - コンペ概要: `docs/project-desc/宇宙からの降水ナウキャスト - Solafune-概要.pdf`
> - 各バンドの中心波長・用途は下記の公式/一次資料を調査して記載（各セクション末尾の出典参照）

本コンペティションでは、静止衛星のマルチスペクトル画像を入力として、降水量（GPM-IMERG）を推定する。

> **注意**: 本コンペティションでは **外部データセットの使用は禁止** されています（過去コンペとはルールが異なる点に注意）。

---

## データセットの出典・ライセンス

| データ | 提供元 | ライセンス |
|---|---|---|
| ひまわり8/9号 | 気象庁 | 日本政府標準利用規約（バージョン2.0） |
| GOES | NOAA Geostationary Operational Environmental Satellites | Creative Commons CC0 1.0 Universal |
| Meteosat | EUMETSAT | EUMETSAT Data Policy |
| GPM-IMERG | NASA/JAXA | Free and Open Archive |

---

## 入力データ（衛星マルチスペクトル画像）

3種類の静止衛星のいずれかの観測画像が入力。いずれも **16バンド**。
画像は全円盤（フルディスク）画像から、各視点位置と固有の日時に基づいて関心領域（ROI）を抽出済み。

降水推定の観点では、おおまかに次の役割を持つバンド群がある:

- **可視/近赤外（反射）**: 雲の有無・厚さ・粒径、昼間のみ有効
- **水蒸気（WV）**: 中〜上層の大気中の水分量
- **赤外窓（IR window）**: 雲頂温度 → 雲頂高度 → 対流の強さ（背の高い冷たい雲ほど強い降水と相関）

### ひまわり8/9号（Himawari / AHI）

- **機器**: マルチスペクトル機器（AHI）レベル **1B** 画像 / **16バンド**（バンド名 `B01`〜`B16`）

| バンド | 中心波長(μm) | 区分 | 意味・主な用途 |
|---|---|---|---|
| `B01` | 0.47 | 可視 (Blue) | エアロゾル、植生、可視画像 |
| `B02` | 0.51 | 可視 (Green) | 可視画像、植生 |
| `B03` | 0.64 | 可視 (Red) | 雲・地表の高解像度可視画像（最高解像度 0.5km） |
| `B04` | 0.86 | 近赤外 | 植生、エアロゾル（陸/水の判別） |
| `B05` | 1.6 | 近赤外 (SWIR) | 雲相（水/氷）、雪氷判別 |
| `B06` | 2.3 | 近赤外 (SWIR) | 雲粒径、雲相 |
| `B07` | 3.9 | 短波赤外 | 下層雲・霧、夜間雲、火災検出 |
| `B08` | 6.2 | 赤外 | 上層水蒸気 |
| `B09` | 6.9 | 赤外 | 中層水蒸気 |
| `B10` | 7.3 | 赤外 | 下層水蒸気、SO₂ |
| `B11` | 8.6 | 赤外 | 雲相（薄い雲）、SO₂ |
| `B12` | 9.6 | 赤外 | オゾン |
| `B13` | 10.4 | 赤外 | クリーンな赤外窓、**雲頂温度**（降水と強く相関） |
| `B14` | 11.2 | 赤外 | 赤外窓、雲頂・海面水温 |
| `B15` | 12.4 | 赤外 | ダーティ赤外窓、水蒸気補正 |
| `B16` | 13.3 | 赤外 | CO₂、雲頂高度 |

> 出典: [JMA 気象庁 Himawari AHI バンド仕様](https://www.data.jma.go.jp/mscweb/en/general/sample.html) / [eoPortal: Himawari-8/9](https://www.eoportal.org/satellite-missions/himawari-8-9)
> （B07 中心波長はひまわり8号で3.9μm、9号で3.8μm）

### GOES（ABI）

- **機器**: マルチスペクトル機器（ABI）レベル **1B** 画像 / **16バンド**（バンド名 `C01`〜`C16`）

| バンド | 中心波長(μm) | 区分 | ニックネーム | 意味・主な用途 |
|---|---|---|---|---|
| `C01` | 0.47 | 可視 | Blue | 可視画像、エアロゾル |
| `C02` | 0.64 | 可視 | Red | 高解像度の可視画像（詳細観測） |
| `C03` | 0.86 | 近赤外 | Veggie | 植生モニタリング |
| `C04` | 1.37 | 近赤外 | Cirrus | 薄い巻雲（シーラス）検出 |
| `C05` | 1.6 | 近赤外 | Snow/Ice | 雪氷検出 |
| `C06` | 2.2 | 近赤外 | Cloud particle size | 雲粒径、雲特性 |
| `C07` | 3.9 | 赤外 | Shortwave window | 下層雲・霧、火災、雲/地表解析 |
| `C08` | 6.2 | 赤外 | Upper-level water vapor | 上層水蒸気 |
| `C09` | 6.9 | 赤外 | Midlevel water vapor | 中層水蒸気 |
| `C10` | 7.3 | 赤外 | Lower-level water vapor | 下層水蒸気 |
| `C11` | 8.4 | 赤外 | Cloud-top phase | 雲頂相（水/氷の判別） |
| `C12` | 9.6 | 赤外 | Ozone | オゾン濃度 |
| `C13` | 10.3 | 赤外 | "Clean" longwave window | クリーンな赤外窓、晴天時観測、**雲頂温度** |
| `C14` | 11.2 | 赤外 | Longwave window | 標準的な赤外画像 |
| `C15` | 12.3 | 赤外 | "Dirty" longwave window | エアロゾル越しの地表観測 |
| `C16` | 13.3 | 赤外 | CO2 longwave | 上層の気温解析、雲頂高度 |

> 出典: [NOAA GOES-R ABI Bands Technical Summary](https://www.goes-r.gov/spacesegment/ABI-tech-summary.html)

### Meteosat（MTG / FCI）

- **機器**: マルチスペクトル観測装置（FCI）レベル **1C** 画像 / **16バンド**
- **命名規則**: `<区分>_<中心波長×100 の整数部>`。
  例: `vis_04` ≈ 0.44μm、`wv_63` ≈ 6.3μm、`ir_105` ≈ 10.5μm。
  区分 — `vis`=可視, `nir`=近赤外, `wv`=水蒸気, `ir`=赤外。

| バンド | 中心波長(μm) | 区分 | 意味・主な用途 |
|---|---|---|---|
| `vis_04` | 0.444 | 可視 | 可視画像、エアロゾル、雲検出 |
| `vis_05` | 0.510 | 可視 | 可視画像、雲検出 |
| `vis_06` | 0.640 | 可視 | 薄い巻雲・エアロゾル・局地的な火災イベント検出 |
| `vis_08` | 0.865 | 可視/近赤外境界 | 植生、可視観測 |
| `vis_09` | 0.914 | 近赤外 | 水蒸気影響を受ける近赤外観測 |
| `nir_13` | 1.380 | 近赤外 | 薄い巻雲（シーラス）検出 |
| `nir_16` | 1.610 | 近赤外 | 雪氷・雲相、近赤外画像 |
| `nir_22` | 2.250 | 近赤外 | 火災検出、地表特性、雲粒径 |
| `ir_38` | 3.800 | 赤外 | 火災検出、下層雲・霧、雲特性 |
| `wv_63` | 6.300 | 水蒸気 | 上層水蒸気、大気中の水分追跡 |
| `wv_73` | 7.350 | 水蒸気 | 上〜中層水蒸気 |
| `ir_87` | 8.700 | 赤外 | 雲相、雲・地表温度解析 |
| `ir_97` | 9.660 | 赤外 | オゾン検出 |
| `ir_105` | 10.50 | 赤外 | **雲頂温度**・地表温度（赤外窓、降水と相関） |
| `ir_123` | 12.30 | 赤外 | 熱赤外画像、水蒸気補正 |
| `ir_133` | 13.30 | 赤外 | CO₂、長波赤外、雲頂高度 |

> 出典: [eoPortal: Meteosat Third Generation (FCI)](https://www.eoportal.org/satellite-missions/meteosat-third-generation) / [EUMETSAT MTG FCI Level 1c Data Guide](https://user.eumetsat.int/resources/user-guides/mtg-fci-level-1c-data-guide)

#### 3衛星のバンド対応（ほぼ共通の観測波長）

3衛星は設計が近く、波長帯はおおむね対応する。チャンネル横断で特徴量を統一する際の対応表:

| 観測波長帯(μm) | Himawari | GOES | Meteosat | 主な用途 |
|---|---|---|---|---|
| ~0.47 | B01 | C01 | vis_04(0.44) | 可視・エアロゾル |
| ~0.51 | B02 | — | vis_05 | 可視 |
| ~0.64 | B03 | C02 | vis_06 | 高解像度可視 |
| ~0.86 | B04 | C03 | vis_08 | 植生・近赤外 |
| ~0.91 | — | — | vis_09 | 近赤外 |
| ~1.38 | — | C04 | nir_13 | 巻雲検出 |
| ~1.6 | B05 | C05 | nir_16 | 雪氷・雲相 |
| ~2.2 | B06 | C06 | nir_22 | 雲粒径・火災 |
| ~3.9 | B07 | C07 | ir_38 | 下層雲・火災 |
| ~6.2 | B08 | C08 | wv_63(6.3) | 上層水蒸気 |
| ~7.0 | B09 | C09 | wv_73(7.3) | 中層水蒸気 |
| ~7.3 | B10 | C10 | — | 下層水蒸気 |
| ~8.5 | B11 | C11 | ir_87 | 雲相 |
| ~9.6 | B12 | C12 | ir_97 | オゾン |
| ~10.4 | B13 | C13 | ir_105 | 雲頂温度（赤外窓） |
| ~11.2 | B14 | C14 | — | 赤外窓 |
| ~12.3 | B15 | C15 | ir_123 | ダーティ赤外窓 |
| ~13.3 | B16 | C16 | ir_133 | CO₂・雲頂高度 |

> 注: 波長は衛星ごとに数十nmの差がある。上表は近い波長を同じ行にまとめた概略対応であり、厳密な等価ではない。

---

## 目的変数（ターゲット）: GPM-IMERG 降水情報

> 本コンペティションで予測する **目的変数** が GPM-IMERG の降水量である。

### 降水情報の説明（概要.pdf より）

GPM-IMERG は、**較正済み（calibrated）** および **未較正（uncalibrated）** の降水量データを含む、マルチバンドの全球降水量データセットである。

本コンペティションのデータセットでは、このうち **較正済みデータのみ** を使用しており、バンド名は **「降水量」（precipitation）の1バンドのみ** である。

- **データセット種別**: マルチバンドの全球降水量データセット（GPM-IMERG）
- **使用バンド**: 較正済みの「降水量」バンドのみ（**1バンド**）
- **役割**: **これが目的変数（予測対象）**
- **画像処理**: 各観測地点と固有の日時に基づいて、対象領域に合わせて切り出し処理されている
- **単位**（IMERG の標準）: 降水強度 mm/hr

#### 使用できるバンド名（GPM-IMERG）

| バンド名 | 意味・説明 |
|---|---|
| `降水量`（precipitation） | 較正済み（multi-satellite + ゲージ較正）の降水強度。GPM-IMERG の `precipitationCal` に相当。**唯一の使用バンドであり目的変数** |

> 補足: GPM (Global Precipitation Measurement) は GPM Core 衛星（GMI / DPR）を基準に複数の受動マイクロ波・赤外衛星を統合し、IMERG (Integrated Multi-satellitE Retrievals for GPM) アルゴリズムで全球の降水を 0.1°・30分グリッドで推定したプロダクト。「較正済み」は地上雨量計データで月単位較正された `precipitationCal` を指す。
> 出典: [NASA GPM IMERG](https://gpm.nasa.gov/data/imerg)

---

## バンド構成まとめ

| データ種別 | 衛星/データ | バンド数 | バンド名 |
|---|---|---|---|
| 入力 | ひまわり8/9号 | 16 | `B01`〜`B16` |
| 入力 | GOES | 16 | `C01`〜`C16` |
| 入力 | Meteosat | 16 | `vis_04`, `vis_05`, `vis_06`, `vis_08`, `vis_09`, `nir_13`, `nir_16`, `nir_22`, `ir_38`, `wv_63`, `wv_73`, `ir_87`, `ir_97`, `ir_105`, `ir_123`, `ir_133` |
| **目的変数** | GPM-IMERG | **1** | `降水量`（precipitation、較正済みのみ） |

---

## 出典一覧

- ひまわり8/9（AHI）: [JMA MSC](https://www.data.jma.go.jp/mscweb/en/general/sample.html), [eoPortal](https://www.eoportal.org/satellite-missions/himawari-8-9)
- GOES（ABI）: [NOAA GOES-R ABI Technical Summary](https://www.goes-r.gov/spacesegment/ABI-tech-summary.html)
- Meteosat（MTG/FCI）: [eoPortal MTG](https://www.eoportal.org/satellite-missions/meteosat-third-generation), [EUMETSAT FCI L1c Data Guide](https://user.eumetsat.int/resources/user-guides/mtg-fci-level-1c-data-guide)
- GPM-IMERG: [NASA GPM IMERG](https://gpm.nasa.gov/data/imerg)
