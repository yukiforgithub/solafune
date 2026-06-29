# Phase 0 — EDA & パイプライン疎通：統合レポート

> Solafune「宇宙からの降水ナウキャスト」。静止衛星16バンド（直近30分・最大3フレーム）→ GPM-IMERG 降水量（mm/hr）を回帰、**評価は元スケール RMSE（小さいほど良い）**。
> 本書は [`docs/survey/03_strategy.md` の Phase 0](../survey/03_strategy.md#phase-0--eda--パイプライン疎通最優先) を実行した結果の統合版。
> 各セクションの全文は [`docs/eda/sections/`](./sections/)、数値の生データは [`eda_cache/`](../../eda_cache/)、図は [`docs/eda/figures/`](./figures/)。
> 全数値は `eda_cache/target_stats.parquet`（TRAIN ターゲット tif 1ファイル=1行、**40,686 行 / 68,393,166 画素**）から閉形式算出、生 tif の spot-check で突合済み。

---

## エグゼクティブサマリ

Phase 0 のゴールは「降水分布の把握・格子対応の確定・CV 設計・提出フォーマット確定・ベースライン取得」。以下が**決定的所見**と、それに基づく**決定事項**。

### 決定的所見（数値）

- **極端なゼロ過剰**：正確に 0 の画素 **82.07%**、<0.1mm まで含めると **85.1%**。有雨（≥0.1mm）はわずか **14.85%**。
- **ロングテール**：有雨の歪度 5.35・尖度 47.4、画素最大 **96.5 mm/hr**。`log1p` で歪度 5.35→1.28・尖度 47.4→1.43 とほぼ対称化。
- **ベースライン（画素レベル, train 全体）**：全0 **RMSE=1.4324** / 全画素=平均(c=0.2886) **RMSE=1.4030**。平均で埋めても全0比で 2% しか下がらない＝**ゼロ過剰下では全0 が強い原点**。
- **ターゲット tif は train/eval/sample 全てで `1band × 41×41 × float32`**、NaN無し・負値無し・CRS無し。eval の `test_files/` は正解ではなく提出雛形（自分の予測で上書きする）。
- **入力は uint8・16band・CRS無し**。衛星別サイズ himawari 81×81 / goes 141×141 / meteosat 144×144。入力/ターゲット比は非整数（1.976 / 3.439 / 3.512）。死にバンドは0本、16本すべて有効。
- **TRAIN 20地域 と EVAL 18地域 は完全 DISJOINT**（交差0）。各地域は単一衛星に1:1対応。衛星別に雨量レジームが約3倍違う（Meteosat は乾燥側、全0 RMSE 0.85 / Himawari・GOES は湿潤側、1.6〜1.84）。
- **欠測フレーム**：train 0枚235・1枚8・2枚647、**eval にも 0枚29・1枚8・2枚567 が存在**（除外不可、推論器は0〜3枚で動く必要）。
- **地域別平均は主CV・eval で全体平均に縮退**：検証地域が学習に存在しないため全行が全体平均へフォールバック。主CV(地域GroupKFold)の honest なベースライン **RMSE=1.4048**（ランダムKFoldの楽観値 1.3836 は本番では使えない）。

### 決定事項

1. **回帰ターゲットは `log1p(降水量)`**、評価は元スケールなので逆変換後に **isotonic 較正（特に強雨の過小評価補正）**を CV で行う。
2. **モデル構成は2部（分類 P(rain) × 有雨 log1p 回帰）**を本命、Tweedie 損失を比較対象とする（ゼロ過剰対策）。
3. **後処理は必須**：負値→0クリップ、微小値のゼロ丸め（しきい値は CV 最適化）、強雨の上限クリップ。
4. **提出 tif は `1band × 41×41 × float32`、ファイル名は `gpm_imerg_filename` 完全一致**、CRS/transform 不要、負値→0クリップを出力直前に適用。`predict.py` 1本で生成・形式検証通過済み（`format_valid=True`）。
5. **主CV = 地域ホールドアウト `GroupKFold(group=name_location)` 5fold**。ランダムKFoldは未知地域汎化を過大評価するため使わない。**衛星×行数×降水強度をバランスした手設計 `name_location→fold` マップ**を [§30 C-3](./sections/30_grid_and_cv.md#c-3-推奨-5-fold-割当groupkfold-groupname_location) に確定、Phase 2 以降は `conf/` に固定。
6. **リサンプリング決定**：GBDT 向けは各16バンドを **`INTER_AREA`（面積平均）で 41×41 に縮約**して画素テーブル化。CNN 向けは **native 解像度を保持**し出力で 41×41 に整合（損失計算は 41×41 グリッド）。
7. **欠測フレーム扱い**：3枚=時間差分特徴フル / 1〜2枚=最新フレーム主入力＋差分は欠測（捨てない） / 0枚=**気候値フォールバック**（未知地域では地域別でなく**衛星×季節×時刻**の転移可能な軸で組む）。フレーム数はカテゴリ特徴、欠測は GBDT で NaN・CNN でマスクチャネル。
8. **サンプル重みは行数で取らない**（france 7,167行=17.6% が支配する）。難所（hat_yai/aceh/jamaica 等、全0 RMSE 2.2〜2.6）に学習リソースを寄せ、無降水画素はダウンサンプル。ただし評価分布は保持。
9. **月は層化キーに使わない**（衛星・地域と分離不能に交絡、月2/5/6 は欠落）。昼夜は UTC でなく**ローカル太陽時／太陽天頂角**を特徴に。

> **超えるべき原点**：未知地域汎化の honest なベースライン **RMSE 1.4048**。Private LB の目安は 1.40 台前半。モデルはこれを下回って初めて意味がある。

---

## 目次

| § | セクション | 主題 | 全文 |
|--:|---|---|---|
| 10 | ターゲット降水分布 | ゼロ過剰・log1p対称化・裾・RMSE含意 | [10_target_distribution.md](./sections/10_target_distribution.md) |
| 20 | 層別統計と分布シフト | 衛星/地域/季節/時刻の不均衡、重み・層化設計 | [20_stratification.md](./sections/20_stratification.md) |
| 30 | 格子対応・欠測・CV設計 | リサンプリング、欠測フレーム、★地域ホールドアウトCV | [30_grid_and_cv.md](./sections/30_grid_and_cv.md) |
| 40 | 入力16バンドの特性 | DNレンジ・死にバンド・昼夜0・波長対応 | [40_input_bands.md](./sections/40_input_bands.md) |
| 45 | バンド×降水の関係 | 全16バンド相関・昼夜別・Tb→RRキャリブ・split-window | [45_band_target_relationship.md](./sections/45_band_target_relationship.md) |
| 50 | ベースライン & 提出疎通 | 定数CV RMSE、3モジュールE2E、提出形式照合 | [50_baselines.md](./sections/50_baselines.md) |

関連: 戦略 [`03_strategy.md`](../survey/03_strategy.md) ／ 特徴量調査 [`01_input-features.md`](../survey/01_input-features.md) ／ 手法調査 [`02_estimation-methods.md`](../survey/02_estimation-methods.md)。

---

## §10 ターゲット降水分布の特性化

[全文 →](./sections/10_target_distribution.md)

TRAIN 全 40,686 ファイル（41×41=1,681 画素／合計 68,393,166 画素）でターゲット GPM-IMERG（mm/hr）の分布を確定。

- **ゼロ過剰**：`==0` 82.07%、`(0,0.1)` 3.08%、`[0.1,1)` 8.16%、`[1,5)` 5.32%、`[5,10)` 0.99%、`[10,20)` 0.31%、`≥20` 0.079%。強雨(≥5mm)は全画素の 1.37% だが二乗誤差では裾が RMSE を支配する。
- **log1p 対称化**：有雨条件付き（平均1.80・SD3.06 mm/hr）の歪度 5.35→1.28・尖度 47.4→1.43。ただし全画素（ゼロ込み）は log1p でも歪度が残る＝**2部構成（分類×有雨回帰）が log と相性が良い**。
- **裾**：全画素 p99=5.40 / p99.9=17.11、全数最大 96.5 mm/hr。**強雨の過小評価を isotonic で較正する余地が最大の改善源**。
- **画像単位**：全面無降水画像 13.26%、有雨被覆率の中央値 6.19%＝降っている領域は局所的（空間モデルでまばらな降雨域を見落とさないことが重要）。
- **RMSE の正体**：全0で 1.4324 になるのは「82%を完璧に当てても残り18%、特に上位0.1%の強雨を外すと二乗で罰される」構造。ゼロ過剰は RMSE を下げてくれない。

図：[ビン別割合](../../eda_cache/fig_target_bins.png) ／ [log1p ヒストグラム](../../eda_cache/fig_target_log1p_hist.png)。数値：[`target_dist_summary.json`](../../eda_cache/target_dist_summary.json)。

---

## §20 層別統計と分布シフト

[全文 →](./sections/20_stratification.md)

衛星・地域・季節・時刻の不均衡を定量化し、サンプル重み・CV層化に落とす。層別の全0 RMSE は「その層の難易度（雨の多さ）」の代理。

- **衛星別（最大の分布シフト軸）**：Himawari 平均0.415・全0 RMSE 1.837 / GOES 0.385・1.603 / **Meteosat 0.134・0.853**（別レジーム、乾燥側）。全体 RMSE は湿潤衛星（Himawari/GOES）に支配される。図：[strat_satellite.png](./figures/strat_satellite.png)。
- **地域別**：20地域で雨量レジームが2〜3桁の幅。難所＝hat_yai(全0 RMSE 2.601)/jamaica(2.481)/central_vietnam(2.509)/aceh(2.241)。ほぼ無降水＝dhaka/bihar/cape_town（平均<0.004）。**france が 7,167行(17.6%)と突出かつ乾燥側(0.099)**。図：[strat_location.png](./figures/strat_location.png)。
- **季節（月）⚠️交絡**：月は衛星・地域とほぼ分離不能（月2/5/6 欠落、2025に偏在）。見かけの季節性は標本構成効果。**層化キーにもサンプル重みにも使わない**。図：[strat_month.png](./figures/strat_month.png) ／ [strat_month_coverage.png](./figures/strat_month_coverage.png)。
- **UTC時刻**：行数はほぼ均等（時刻不均衡なし）。日変化は ±12% と弱いが、UTC は3衛星の経度差で位相がバラバラ＝**ローカル太陽時／太陽天頂角を使うべき**。図：[strat_hour.png](./figures/strat_hour.png)。
- **設計への落とし込み**：(1)重みは行数でなく誤差寄与ベース、無降水ダウンサンプル＋有雨オーバーサンプル、無降水3地域はダウンウェイト（評価分布には残す）。(2)第一キーは地域GroupKFold、fold間で衛星・雨量レジームを揃える、france は単独 fold 固定。(3)手元CVは未知地域での挙動に振り切る。

---

## §30 格子対応・欠測フレーム・CV設計 ★最重要

[全文 →](./sections/30_grid_and_cv.md)

- **格子対応**：ターゲットは train/eval/sample 全て `1band×41×41×float32`（CRS無）。入力は uint8 16band（CRS無、単位行列）、himawari 81² / goes 141² / meteosat 144²。同一ROI・異格子と解釈、入力/ターゲット比は非整数。
- **リサンプリング**：GBDT＝各16バンドを `INTER_AREA` で 41×41 に縮約し画素テーブル化（複数窓の近傍統計も同時計算）。CNN＝native保持＋出力で 41×41 整合。
- **欠測フレーム**：train {0:235,1:8,2:647,3:39796} / eval {0:29,1:8,2:567,3:28486}。**eval にも欠測があるため除外不可**。0枚＝気候値フォールバック（衛星×季節×時刻）、1〜2枚＝最新フレーム主入力で捨てない、フレーム数はカテゴリ特徴。
- **CV ★**：train20地域・eval18地域が完全 DISJOINT。ランダムKFoldは地域固有の気候を覚えて楽観化するので **主CV = `GroupKFold(group=name_location)` 5fold**。[手設計 fold マップ（C-3）](./sections/30_grid_and_cv.md#c-3-推奨-5-fold-割当groupkfold-groupname_location) で全 fold に3衛星を確保し行数(7,235〜10,095)・平均RR(0.117〜0.404)をバランス。france は fold2 に固定。`name_location→fold` 辞書を `conf/` に固定（seed非依存・再現可能）。地域内 hold-out は日/イベント単位でグループ化し時間リーク回避。

---

## §40 入力16バンドの特性

[全文 →](./sections/40_input_bands.md)

train 各衛星から入力 tif をランダム400枚（seed=42）で48バンド（衛星×16）を集計。

- **uint8 フルレンジ利用**：48本全てで min=0、28本で max=255 到達。線形8bit量子化済みと推定→ 反射系は /255、放射系は p1–p99 ロバストスケールが素直。**真の物理単位への逆変換係数は GeoTIFF に無い**（衛星内の相対量でフィットせざるを得ない）。
- **死にバンド0本**：is_constant/is_empty とも0。16本すべて情報を持つ。
- **0は no-data 番兵でない**：全16band同時0はほぼ皆無。可視/近赤外の高い frac0 は**夜間**（反射ゼロ）。→ **昼夜フラグ／太陽天頂角を必須特徴に**。放射系の数%の0はスキャン端/散在欠測。
- **バンド群**：反射系(VIS/NIR/SWIR)＝平均DN低・分散大・昼のみ／放射系(WV/IR)＝平均DN 140–180で安定。降水と直結するのは**赤外窓(10.4μm)＋水蒸気(6–7μm)**。
- **波長対応**：3衛星でバンドの並び順が揃わない（Himawari/GOES は波長昇順、Meteosat は反射群→放射群）。衛星横断には[波長対応表](./sections/40_input_bands.md#3衛星のバンド対応波長で揃うか)で共通スロットへマップ、欠落バンドはマスク扱い。

図：[fig_bands.png](./figures/fig_bands.png)。数値：[`band_stats.parquet`](../../eda_cache/band_stats.parquet)。

---

## §50 ベースライン & 提出疎通

[全文 →](./sections/50_baselines.md)

定数ベースラインの CV RMSE を確定し、前処理→学習→予測の3モジュールを E2E で通し、提出 zip を sample_submission と形式照合。

- **定数CV RMSE（手法 × CVスキーム）**：

  | 手法 | 地域GroupKFold（主・honest） | ランダムKFold（参照・楽観） |
  |---|--:|--:|
  | 全0 | 1.4324 | 1.4324 |
  | 全体平均 | **1.4048** | 1.4030 |
  | 地域別平均 | **1.4048**（全体平均へ縮退） | **1.3836** |

- **地域別平均の縮退**：主CV では検証地域が学習に無いため全行が全体平均へフォールバック＝同値。これは欠陥でなく「未知地域には地域固有値が原理的に無い」本質を CV が正しく反映。**ランダムCVの 1.3836（−0.0194）は本番では使えない楽観値**。
- **提出は全体平均 c=0.2886 の定数**（eval全18地域が未知なので地域別でも同値）。
- **E2E 疎通**：`preprocess_train.py`（40,686行検証）／`preprocess_test.py`（29,090行・gpm一意検証）／`train.py`（CV RMSE算出・モデルJSON保存）／`predict.py`（提出zip生成）全段通過。
- **形式照合 `format_valid=True`**：tif数=eval行数=29,090、ファイル名集合完全一致、全tif `GTiff/float32/count=1/41×41/CRS None`、同梱CSVがeval CSVと完全一致。成果物 [`submissions/baseline_global_mean.zip`](../../submissions/baseline_global_mean.zip)（9.6MB）。

再現コマンド（§50 (b) より）：

```bash
uv run python src/preprocess_train.py
uv run python src/preprocess_test.py
uv run python src/train.py  --cv-scheme location_group --method global_mean   # 主CV
uv run python src/train.py  --cv-scheme random         --method location_mean # 楽観参照
uv run python src/predict.py --name baseline_global_mean --method global_mean # 提出生成
```

---

## Phase 1 以降への引き継ぎ

[戦略ロードマップ](../survey/03_strategy.md#3-推奨ロードマップ段階的に積む)に沿って Phase 0 の確定事項を渡す。

- **Phase 1（物理ベースライン）**：IR窓 Tb(10.4/11.2μm)単独の Tb→RR べき乗則を**衛星内の相対DN**でフィット（§40：物理単位の逆変換係数が無いため）。
- **Phase 2（GBDT 2部モデル・本命）**：area縮約 41×41 画素テーブル＋近傍統計＋BTD＋昼夜/太陽天頂角/衛星種別。分類×log1p有雨回帰 or Tweedie。後処理（ゼロ丸め・isotonic較正）を CV で詰める。CV実装を §30 手設計 fold マップへ移行。
- **Phase 3（U-Net）**：native保持16ch、2段階、重み付き/逆密度損失、欠測・昼夜マスク。
- **Phase 4/5**：3フレーム時間特徴（dTb/dt）→ GBDT×CNN アンサンブル＋後処理仕上げ。
