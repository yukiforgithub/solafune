# 開発ステータス / 引き継ぎドキュメント

> このファイルは「次のセッション（または別の人）がスムーズに開発を引き継ぐ」ための単一の入口。
> 最終更新: 2026-06-23。コンペ概要・タスク定義は [.claude/CLAUDE.md](../.claude/CLAUDE.md) を、詳細は各 docs を参照。

---

## 0. TL;DR（今どこにいるか）

- Solafune「宇宙からの降水ナウキャスト」: 静止衛星16band(uint8) → GPM-IMERG 降水量(mm/hr, 41×41) を画素回帰。評価 **RMSE**。
- 現在 **Phase 2（GBDT中心）** を進行中。方針は **GBDT 主軸**（ユーザー選好。NN/CNN は将来のアンサンブル要員に格下げ）。
- **★提出ベスト LB = 0.69198（21/52位）＝ Phase3 CNN（Kaggle GPU, 衛星別 U-Net 5fold全データ）**。GBDT best(0.70688)から **−0.015・+8ランク**の大躍進。公開解法0.6947(ConvLSTM)すら上回る。
- **方針転換が成功**: ローカルCPU(7.7GB)ではGBDTが0.707で頭打ち＋小型CNNはGBDT未満だったが、**Kaggle無料GPU(T4)に前処理済みテンソル(41×41,51ch,12GB)を上げて本物のU-Netを学習**したら明確にGBDTを超えた。CNNの空間文脈モデリングが未知地域で効く。
- **CNN進捗**: v1 single(80%)=0.69640 → v1 full(100%)=0.69198。**v2（dual head強度×降水確率 + 衛星別正規化 + 6-way TTA + seed平均）実行中**＝cond≥5(強雨)の弱点狙い。さらに CNN×GBDT アンサンブル余地あり。
- 旧 GBDT best: phase2_gbdt_hp 0.70688（HPチューニングはCV−0.0132改善もLB転移は僅少。GBDTは頭打ち確定）。
- **HPチューニング(reg_strong+power1.7)が大当たり**: 「CV≫LBの大乖離＝訓練地域への過適合」仮説を検証 → 正則化強化(num_leaves63→31, min_child200→500, L1/L2, colsample0.8→0.6) ＋ tweedie_variance_power 1.5→1.7 で **CV −0.0118**。HP探索2ラウンドで確定（§5-D）。学習依存だが地域固有補正でないので transfer する。
- **時間特徴(97)**: CV 1.1757→1.1629, LB→0.70705（24位）。空間統計とは直交する勝ち筋。
- **棄却が確定したもの**（全て fold-out/ablation で確認）: パッチ統計(空間, ~0)・後処理較正(地域非転移)・ブレンド(旧HPでは非転移※新HPでは僅か採用)・入力適応 scene-norm(+0.0002)・小型CNN(GBDT未満)・**物理/対流特徴(冷却分解+雲相BTD, 実験2で純効果+0.0010=無効)**。→ **効くのは「特徴の新軸(時間)」と「正則化HP」**。
- 実行分担: **重い計算はユーザーがローカル実行**（RAM 7.7GB 制約）。**Claude はコード実装 + RUNBOOK 提供のみ、原則として実行しない**。

### リーダーボード状況（2026-06-23 時点）
| | スコア(RMSE) |
|---|---|
| 1位 | 0.6441 |
| **自分(24位)** | **0.70705** |

→ 上位は密集、僅差勝負。1位まで約0.063。**次アクション = `phase2_gbdt_hp.zip`（97特徴・新HP, CV1.1497）を提出**して LB確認（HPチューニングの大改善が転移するはず, 見込み~0.7056）。物理特徴版 `phase2_gbdt_physics.zip` は実験2で無効と判明したので提出不要。較正/ブレンド/scene-norm/小型CNN/物理特徴は全て棄却済（§5）。**残る軸: 更なる時間特徴 or HP微調整 or（GPUあれば）本格CNN**。

### LB 推移（best 更新の履歴）
| 提出 | 特徴 | CV(tweedie) | LB | 順位 |
|---|---|---|---|---|
| phase2_gbdt | 63 | 1.1767 | 0.7086 | 24/37 |
| phase2_gbdt_patch | 83(+空間パッチ) | 1.1757 | 0.70845 | 25/39 |
| **phase2_gbdt_temporal** | **97(+時間3フレーム)** | **1.1629** | **0.70705（best）** | **24/40** |
| phase2_gbdt_temporal_cal | 97 + isotonic較正 | 1.1790(悪化) | 0.709023(悪化) | — |
| phase3_cnn_holdout | CNN小型FCN | 1.2090(fold0; 同fold0 GBDT=1.1755) | 0.724842(悪化) | — |
| phase2_gbdt_scenenorm | 97+シーン正規化8(2.0M) | 1.1593(交絡: 2.0M) | 0.70790(悪化) | ❌棄却 |
| **phase2_gbdt_hp** | **97・新HP(reg_strong+pw1.7)** | **1.1497(blend)** | **0.70688（best, 29/50）** | ✅新best(僅差) |
| phase2_gbdt_physics | 102・新HP(+物理5) | 1.1503(blend) | 0.70820(悪化) | ❌物理棄却 |

→ **★CV→LB 転移率は施策の種類で大きく違う（重要）**:
- **特徴の新軸（時間）: 転移率 ~0.11**（CV −0.0128 → LB −0.0014）。地域非依存の新情報なので強く transfer。
- **後処理較正: 転移率 ~0.12 だが逆効果**（fold-out +0.016 → LB +0.0020、符号一致）。
- **HP/正則化: 転移率 ~0.013（約1/8）**（CV −0.0132 → LB **−0.00017**）。正則化は訓練地域OOFを最適化するが eval 地域へ carry しにくい。
- 教訓: **fold-out CV は「特徴の新軸」の良し悪し判定には信頼できるが、「HP最適化」のLBゲインは大幅に割り引く**こと。CVが大きく動いても、それがHP起因ならLBは僅かしか動かない。

---

## 1. フェーズ別の結果

| Phase | 内容 | CV(地域GroupKFold) | LB(Public) | 状態 |
|---|---|---|---|---|
| 0 | EDA + 3モジュール雛形 + ダミー提出 | 全0:1.4324 / 全体平均:1.4030 / honest:**1.4048** | — | ✅完了・検証済 |
| 1 | 物理: IR窓DN→RR lookup(衛星別) | **1.1644** | **0.7630** | ✅完了 |
| 2 | GBDT 2部/Tweedie, Tier-1特徴(63) | **1.1767**(tweedie採用, 衛星別 himawari1.378/goes1.281/meteosat0.778, cond≥5=8.08) | **0.7086**(24/37位) | ✅完了 |
| 2+ | + サブグリッド パッチ統計(83特徴) | **1.1757**(tweedie, 衛星別 himawari1.377/goes1.282/meteosat0.771, cond≥5=8.07) | **0.70845**(25/39位) | ✅完了・**効果は僅少と確定** |
| 2t | + 時間特徴 直近3フレーム(97特徴) | **1.1629**(tweedie, 衛星別 himawari1.374/goes1.256/meteosat0.761, **cond≥5=7.96**, bias-0.023) | **0.70705**(24/40位, 新ベスト) | ✅完了・**GBDT期で最大の改善・順位上昇** |
| 3 | CNN 衛星別小型FCN(C=51, 3frame×16band+presence) | **1.2090**(holdout fold0)。同fold0のGBDT=**1.1755** → **GBDTに−0.034負け** | 未提出(劣るため) | ⚠️実行済・**小型FCNはGBDTに及ばず** |
| 2h | + HPチューニング(reg_strong+pw1.7, 97特徴) | **1.1511**(tweedie)/**1.1497**(blend), cond5 7.98 | 未提出(本命) | ✅完了・**最大級の改善・要提出** |

### 重要: CV と LB の関係（絶対値で比較しない）
- **CV ≫ LB**（Phase1: CV1.16→LB0.763 / Phase2: CV1.18→LB0.709）。約 **0.4 の有利な乖離**がある。
- 原因: train(20地域)/eval(18地域) は **DISJOINT** で、train CV は激甚降水地域(hat_yai等)に支配され悲観的。入力の雲頂分布は train/eval でほぼ同等（ドメインシフトではない）。LH24 ディスカッションも同方向 → **コンペ構造的性質**でバグではない。
- **使い方: CV は絶対値でなく「モデルの順位付け」に使う**。CV改善 → LB転移を提出で定期確認（提出枠: 人数×5/日, JST9:00リセット）。強雨条件付きRMSE(cond≥1,≥5)・衛星別も併記し、極端地域への過剰最適化を監視。

---

## 2. パイプラインと実行方法

3モジュール構成（コンペ要件）。すべて `uv run`、リポジトリルートから。

### コード地図
```
src/precip/
  config.py        # パス定数・衛星バンド名・TARGET_SIZE(41,41)
  dataio.py        # read_target/read_satellite/write_prediction_tif/load_*_df
  cv.py            # load_handdesigned_folds()=conf/folds.yaml の地域5fold（正準）
  metrics.py       # rmse
  phase1_suffstats.py # 物理: 窓DN/split充足統計, _resize_area(INTER_AREA→41x41), WINDOW/SPLIT_BAND_INDEX
  phase1_fit.py    # 物理: lookup/powerlaw/exp fit, 衛星別CV
  physical.py      # 物理: PhysicalIRModel(推論器)
  features.py      # ★Phase2 画素特徴抽出(97特徴=16band+BTD+空間32+構造3+メタ7+位置3+パッチ20+時間14, 学習/予測で一元化)
  gbdt.py          # ★Phase2 2部(P(rain)·E[y|rain]) / Tweedie の fit/predict/save/load
  calibrate.py     # ★Phase2 後処理較正(isotonic): OOFからfold-out評価+衛星別LUT fit/apply（※棄却済,既定off）
  ensemble.py      # ★Phase2 ブレンド: two_part×tweedie をper-sat OOF重みで結合(fold-out評価+閉形式重み)
  cnn.py           # ★Phase3 CNN: 衛星別小型FCN(C=51,プーリング無,dilation) + memmap前処理 + holdout/full学習
src/preprocess_train.py  # --method {constant,physical_ir,gbdt}
src/preprocess_test.py   #   gbdt はスタブ(特徴は predict が実行時抽出)
src/train.py             # --method ... gbdt分岐=train_gbdt(衛星別遅延ロード+CV+最終fit+importance)
src/predict.py           # --method ... gbdt分岐=predict_gbdt(チャンク並列, 提出zip)
src/build_phase1_suffstats.py # 物理キャッシュ構築エントリ
conf/config.yaml         # cv/baseline/method/physical_ir/gbdt/postprocess
conf/folds.yaml          # 正準 name_location→fold(5fold, 地域GroupKFold)
run-phase2.sh            # Phase2(patch無し) 一括: preprocess→check→train→predict
run-phase2_patch.sh      # Phase2(patchあり,83特徴) 一括
run-phase2_temporal.sh   # Phase2(時間特徴あり,97特徴) 一括 ※提出名 phase2_gbdt_temporal
run-phase2_calibrate.sh  # Phase2t + 後処理較正(isotonic) train+predict ※前処理不要・提出名 phase2_gbdt_temporal_cal（※較正は棄却済）
run-phase2_ensemble.sh   # Phase2t + two_part×tweedie ブレンド train+predict ※前処理不要・提出名 phase2_gbdt_temporal_blend
run-phase3.sh            # Phase3 CNN 一括: memmap前処理→検査→学習(holdout/full)→予測 ※提出名 phase3_cnn_{mode}
```

### Phase3 CNN のコード地図（追加）
```
src/preprocess_train.py --method cnn  # cnn.build_memmaps: 衛星別 入力テンソル memmap(N,51,41,41)f16 + y + fold
src/train.py            --method cnn  # cnn.train_cnn: holdout(fold0)/full(5fold OOF)+最終fit, phase3_models/{sat}.pt
src/predict.py          --method cnn  # predict_cnn: build_cnn_input→衛星別FCN forward(バッチ)→expm1→0クリップ→提出
conf/config.yaml cnn:                 # cv_mode/epochs/batch_size/lr/num_threads 等
outputs/phase3_cnn/{sat}_X.npy,_y.npy,_meta.npz  # memmap(f16, gitignore)
outputs/phase3_models/{sat}.pt        # 学習済み state_dict
outputs/phase3_{cv,selected}.json     # CV結果 / 採用情報
```

### 実行コマンド
```bash
# Phase1（物理ベースライン）  詳細: docs/phase1/RUNBOOK.md
uv run python -m src.build_phase1_suffstats
uv run python src/train.py   --method physical_ir
uv run python src/predict.py --method physical_ir --name phase1_physical_ir

# Phase2（GBDT）  詳細: docs/phase2/RUNBOOK.md
bash run-phase2.sh             # patch無し(63特徴)
bash run-phase2_patch.sh       # patchあり(83特徴) ※提出名 phase2_gbdt_patch
# 前処理済み(parquet健在)なら train+predict だけ再実行も可:
#   uv run python -u src/train.py   --method gbdt 2>&1 | tee train.log
#   uv run python -u src/predict.py --method gbdt --name phase2_gbdt 2>&1 | tee predict.log
```

### 主要な成果物（gitignore対象）
```
eda_cache/target_stats.parquet            # 全ターゲット集計
eda_cache/phase1_*_hist.parquet           # 物理 充足統計
outputs/phase1_model.json, phase1_train_cv.json
outputs/phase2_features_{sat}.parquet     # GBDT 画素特徴(83列=feat+y+fold)
outputs/phase2_cv.json                    # GBDT CV結果(two_part/tweedie)
outputs/phase2_selected.json              # 採用variant+特徴名
outputs/phase2_models/{sat}_{clf,reg,tweedie}.txt  # LightGBM Booster
outputs/phase2_feature_importance.csv
submissions/phase{1,2}*.zip               # 提出
```

---

## 3. ★環境の制約とハマりどころ（必読）

- **RAM 7.7GB（メモリ制約大）**。`free -h`/`swap` を常に意識。OOM は**エラー無しの突然死**（SIGKILL, ログ flush されず）として現れる。
- **OOM対策（実装済み）**:
  - 前処理: `build_gbdt_features` は「上限から逆算した実効サンプル率」で巨大配列を作らず省メモリ化。`conf gbdt.max_pixels`(既定2.5M)/`pixel_sample_frac`(0.15)。
  - 学習: `train_gbdt` は衛星を1つずつ遅延ロード。LightGBM は `force_col_wise: true` + `max_bin: 127`（conf）で省メモリ。
  - それでも OOM するなら `gbdt.max_pixels` を下げて前処理から再実行。
- **シェルスクリプトは `bash` で実行**（`sh` は dash で `set -o pipefail` 不可 → `bash run-phase2.sh`）。
- **ヒアドキュメントの位置**: `python -u - <<'PY' ... PY 2>&1 | tee log`（`<<` は python に結合。`| tee` の後ろに置くと python が stdin を読みハングする）。
- **出力バッファリング**: ログにリアルタイム表示するには `python -u` + `PYTHONUNBUFFERED=1` + `tee`。これが無いと長時間ジョブのログが空に見える。
- **`set -euo pipefail`**: 前処理が失敗したのに train へ進む事故を防ぐ（過去に goes/meteosat が smoke=300行のまま学習・提出され LB を毀損した）。
- **特徴スキーマを変えたら必ず前処理を再実行**（parquet列が変わるため。train は `features.feature_names()` 順で列選択）。

---

## 4. データ/モデリングの確定知見（再導出しないこと）

- ターゲット: 常に 1band 41×41 float32, NaN/負値なし, mm/hr。**ゼロ過剰: 正確な0が82.07%**（<0.1mm 85%）、全画素平均0.289、最大96.5。
- 入力: uint8 16band, **CRS/ジオトランスフォーム無し**。サイズ himawari81²/goes141²/meteosat144²。**入力とターゲットは同一ROI・異格子** → `cv2.INTER_AREA` で入力を41×41へ縮約して画素対応（中心合わせ・全域対応の仮定。未検証）。
- **train20地域 / eval18地域 は完全DISJOINT**（未知地域汎化が本質）。各地域は単一衛星に対応。
- 衛星別IR窓バンド(rasterio 1-based): himawari=13, goes=13, meteosat=14。split相手=15。**低DN=冷たい雲頂=強雨**（E[y|DN]単調減少, Spearman −0.4〜−0.5）。
- バンド相関: **IR/WV が一様に負相関、可視/近赤外はほぼ無相関**（夜間さらに弱い）→ IR/WV が主役、昼夜フラグ必須。詳細 docs/eda/sections/45。
- **GBDT feature importance: 局所MINが最重要**（ir38_min3, win_min3, win_min7…＝最も冷たい雲頂＝対流コア）。空間統計が gain の 77-85%。
- **強雨(≥5mm)が弱点**: cond≥5 の RMSE ≈ 8.08（全体の約7倍）。平均化でサブグリッドの冷たいコアが消えるため。→ パッチ統計(min保持)を追加した(Phase2+)。
- 欠測フレーム: train 0枚=235行/1枚=8/2枚=647/3枚=39796。eval にも 0枚=29 等あり → 推論器は 0〜3枚で動く設計、0枚は気候値0.2886フォールバック。

### 外部知見: LH24 ディスカッション（docs/discussion/）
- 公式CNN(可視) public **0.913**、本人の split-window 小型U-Net public **0.708**。「**バンド選択 > モデルサイズ**」。
- 推奨: 地域グループCV、log1p損失+expm1+clamp、split-window差、雲頂冷却率(BT_t0−BT_t2)。
- 詳細・我々のEDAとの突き合わせ: [docs/discussion/0708-lh24-summary.md](discussion/0708-lh24-summary.md)。

---

## 5. 次の一手（優先度順・GBDT中心）

1. ~~パッチ統計(83特徴)の効果確定~~ → ✅完了。CV/LB共に改善0.001前後・順位は後退。**空間統計は飽和**、これ以上の積み増しは見送り。
2. ~~時間特徴(直近3フレーム, 97特徴)~~ → ✅**完了・提出済（LB 0.70705, 24位, 新ベスト）**（2026-06-23, `bash run-phase2_temporal.sh`, `submissions/phase2_gbdt_temporal.zip`）。
   - CV 1.1757→1.1629（−0.0128, GBDT期最大）。cond≥5（強雨）8.067→7.96。goes が最も改善。LB 0.70845→0.70705 で**初の順位上昇(25→24)**。
   - 知見: 時間特徴の gain 割合は ~1%（上位15外）でも、二乗誤差を支配する希少な強雨画素を補正して効く。**3x3平滑した冷却率(`win_dt_s3`/`ir38_dt_s3`)が raw より上位**＝移流ノイズ除去が有効。
   - **拡張余地（後回し候補）**: tmax・冷却率の符号別カウント・フレーム数で正規化した冷却率(per-10min)・2フレーム目との差分など。ただし gain が小さいので大きな伸びは期待薄。先に下記3を優先。
3. ~~後処理較正(isotonic)~~ → ❌**棄却・LBでも悪化を確定**（2026-06-23, `run-phase2_calibrate.sh`, 提出 `phase2_gbdt_temporal_cal`）。
   - **結果: OOF RMSE 1.1629 → 1.1790（gain −0.0160, 悪化）, cond≥5 7.96 → 8.13, 全衛星で悪化**（goes 1.256→1.295 が最大）。**LB も 0.70705 → 0.709023（+0.00197 悪化）**。
   - **★fold-out が LB を正確に予測**: fold-out 悪化 +0.016 × 転移率0.11 ≈ +0.0018 ≒ 実測 +0.00197。**符号も大きさも一致** → honest fold-out CV は信頼できる LB 代理。今後 CV を順位付けに使ってよい強い裏付け。
   - **原因（重要・再導出不要）**: isotonic は in-sample では必ず RMSE を下げるが、**地域 GroupKFold の fold-out では悪化**。較正器が学ぶ「pred→y 関係」は地域ごとに異なり（気候/降水レジーム差）、学習地域の較正マップは**未知地域に転移しない**。生予測（地域非依存のベスト推定）を歪めて悪化させる。
   - **結論: post-hoc 較正はこのコンペ構造（disjoint 地域汎化）では行き止まり**。現行モデルは既に良較正(bias −0.023)。`conf gbdt.calibration: none`（既定）に固定、`phase2_calibrators.json` は削除済。コード（`calibrate.py` / run script）は将来の参照用に残置（既定 off で無害）。
4. ~~アンサンブル（two_part×tweedie ブレンド）~~ → ❌**棄却・fold-out で単一に劣り不採用**（2026-06-23, `run-phase2_ensemble.sh`）。
   - fold-out OOF: 単一 1.1629 → ブレンド 1.1661（gain −0.0032, 悪化）, cond≥5 7.96→8.13。weights={himawari0.606, goes0.709, meteosat1.0}。自己防衛で **variant=tweedie に自動フォールバック**＝提出物は temporal と同一(0.70705)なので**提出せず**（重複zip削除済）。
   - **較正と同じ地域非転移**（程度は軽い）。in-sample では凸結合は必ず単一以下だが、学習地域で最適化した重みが未知地域で外れて悪化。弱い two_part を混ぜた分が裏目。
   - コード（`ensemble.py` / run script）は残置（既定 `gbdt.ensemble: blend` だが自己防衛で安全。単一が勝てば自動で単一採用）。

### ★メタ結論: 学習データ依存の後処理・結合は本コンペで transfer しない（確定）
- **較正(isotonic) も ブレンド(blend) も、学習地域で最適化したパラメータが disjoint な未知地域へ転移せず逆効果**。共通原因＝train/eval 地域 DISJOINT。
- **transfer するのは「地域統計に依存しない操作」だけ**: ①入力特徴の改善（地域非依存な観測量。実際 time特徴が最大の利得）、②seed平均のような等重み平均（学習重みなし）、③モデル構造の改善（CNN）。
- **方針: GBDT の後処理チューニングは打ち止め**。次は「特徴の改善」か「CNN」へ。施策ごとに必ず「地域 fold-out で transfer するか」を先に問う。
5. **【次の二択】どちらも transfer する productive な軸**:
   - **(A) 追加の特徴エンジニアリング（低〜中リスク・実績あり）**: time特徴が最大利得だった事実から、特徴軸はまだ伸びうる。候補＝冷却率の per-10min 正規化 / tmax / 2フレーム目との差分 / マルチスケール時間窓 / 近傍×時間の交差（最冷セルの時間変化）。特徴は地域非依存なので transfer する。前処理から再生成が要る点だけ重い。
   - **(B) CNN(Phase3)** → ⚠️**実行済・小型FCNはGBDTに及ばず**（2026-06-24, holdout fold0）。
     - **公平比較（同一fold0）**: CNN overall 1.2090 vs GBDT tweedie 1.2090→**1.1755**。全衛星(himawari1.665/0.938/0.074 vs 1.637/0.883/0.074)・cond≥5(8.17 vs 7.48)・bias(−0.098 vs −0.019)すべてGBDT優位。**−0.034 負け**。
     - **解釈**: CPU制約下の小型FCN(142Kパラメータ, log1p MSE, 20ep)では空間文脈の優位が出ず、特に**強雨をより過小予測**（log1p損失＋小型が裾を平滑化）。「GPUなしの小型CNNでGBDT(良特徴)は超えにくい」＝LH24「band選択>モデルサイズ」とも整合。
     - **提出して LB 0.724842（temporal 0.70705 より +0.018 悪化）でCV判定を裏付け**。改善カード: 強雨重み損失/物理ベースchannel注入(§保留)/U-Net化/統合モデル(データ3倍)/seed平均。ただしCPU制約で ROI は低く、**idea-driven なGBDT改善(下記)の方が有望**。
     - 診断スクリプト: `scratch/gbdt_fold0_eval.py`（GBDTをfold0で評価=CNN holdoutと公平比較）。
   - **(C) idea-driven なGBDT改善**: ①~~地域不変な入力正規化(scene_z/scene_rank)~~ → ❌**棄却**（2026-06-24）。同一データのアブレーションで純効果 **+0.0002(無効)**、LBも 0.70705→**0.70790悪化**。原因: **入力分布が train/eval で既に揃っている**（EDA確定）ため、シーン正規化＝入力適応に伸びしろが無い。**この知見は「入力ベースの適応/類似度重み付け」路線全般が天井低い**ことを示す（disjointなのは入力でなくP(y|x)＝eval正解が無いと埋められない concept shift）。②~~物理特徴の深掘り（冷却分解 cool01/cool_accel + 雲相BTD 8.6µm−IR窓, 5特徴）~~ → ❌**棄却**（2026-06-25, 実験2 ablation）。同一データ・同一HPで純効果 **overall +0.0010 / cond5 +0.032(無効〜微悪化)**。cond≥5(強雨)を直撃する狙いだったが逆効果。空間統計・入力適応に続き**特徴の積み増し系は飽和**と確定。features.py から撤去。
     - ★OOM対策の副産物: preprocess に `del X,y,fold,out`（衛星間で~2GB解放）を追加。max_pixels=2.5M を保ったまま 105特徴でも通る基盤ができた（features.py からは scene を撤去済、現在 97特徴に復帰）。
   - **(D) HP チューニング** → ✅**大当たり・本命**（2026-06-25, HP探索R1/R2 + 本番実験3）。
     - **仮説検証成功**: 「CV(1.16)≫LB(0.707)の大乖離＝訓練地域への過適合」→ 正則化を強めると 5-fold OOF(地域汎化)が改善。確定HP=**reg_strong（num_leaves63→31, min_child200→500, reg_lambda5, reg_alpha1, colsample0.8→0.6, subsample0.8→0.7）＋ tweedie_variance_power 1.5→1.7**。
     - **本番(97特徴・2.5M)**: CV 1.1629→**1.1511(tweedie)/1.1497(blend)** ＝ **−0.0118〜−0.0132**（GBDT期で時間特徴に並ぶ最大級）。cond5 は 7.96→7.98 とほぼ不変（汎化向上が主因）。conf に反映済。提出 `phase2_gbdt_hp.zip` 生成済 → **要提出**（LB ~0.7056-0.7058 見込み）。
     - power は **1.7が頂点**（R2: 1.3悪化/1.7最良/1.9破綻）、正則化は reg_strong が適量（reg_Xstrong過剰で悪化）。**新HPではブレンドが僅かに採用される**（fold-out +0.0014, ただし cond5 悪化）= 提出は blend variant。
     - 副産物: preprocess のメモリ根治（衛星間 del + concat ピーク半減）で **2.5M が初めてOOMなしで通る**ようになった。
   - **(E) 残る候補**: 更なる時間特徴（移流補正/対ごと冷却を別実装で再挑戦）、HP微調整（learning_rate↓×n_estimators↑、num_leaves/正則化の再スキャン）、seed平均、（GPUあれば）本格CNN。ただし特徴積み増し系は飽和傾向。
     - **構成**: 衛星別 小型FCN（`cnn.py`, C=51=3frame×16band+presence3面, プーリング無で41×41維持・dilationで受容野拡大, 142Kパラメータ）。損失=log1p(y)のMSE→expm1→0クリップ。入力は f16 memmap でRAM節約。**CPU専用**（GPUなし環境）。
     - **2段運用**: ①`cv_mode=holdout`（既定, fold0検証で80%学習）でまず GBDT 0.70705 を超えられるか高速判定 → 提出 `phase3_cnn_holdout`。②有望なら `conf cnn.cv_mode=full` で 5fold OOF（正直なLB代理）+全データ最終fit → `phase3_cnn_full`。
     - **★compute 警告**: CPU 8コアで holdout 全データ ≈ 数時間, full ≈ その6倍（~17h）。長時間ジョブ前提。OOM時 `conf cnn.batch_size` を下げる。
     - **transfer 見込み**: モデル構造改善は地域統計に依存しないので較正/ブレンドと違い transfer するはず。判定は holdout(fold0=地域disjoint) RMSE を GBDT temporal CV1.1629 と比較（holdoutは単一foldで full OOF より高分散な点に留意）。
     - **次の改善余地（CNNが competitive なら）**: U-Net化, 時間特徴の明示注入, log1p以外の損失(強雨重み), seed平均, TTA。

### 確定知見: post-hoc 較正は本コンペでは無効（§5-3 詳細）
- **理由を一般化**: eval は train と disjoint な地域。**学習データ統計に依存する後処理（較正・地域別補正など）は未知地域へ転移せず逆効果**。生予測の平均（アンサンブル）のように地域非依存な操作のみが転移する。今後の施策はこの観点で「地域転移するか」を必ず問うこと。CV は必ず地域 GroupKFold の fold-out で測る。

詳細な全体戦略: [docs/survey/03_strategy.md](survey/03_strategy.md)。

---

## 6. ドキュメント地図

| 文書 | 内容 |
|---|---|
| [.claude/CLAUDE.md](../.claude/CLAUDE.md) | コンペ概要・タスク・規約・用語(LB/CV) |
| [docs/STATUS.md](STATUS.md) | ←本ファイル。進捗と引き継ぎの入口 |
| [docs/survey/03_strategy.md](survey/03_strategy.md) | 勝ち筋・ロードマップ(Phase0-5) |
| [docs/survey/01,02](survey/) | 入力特徴・推定手法のサーベイ |
| [docs/eda/README.md](eda/README.md) | EDA統合(10分布/20層別/30格子CV/40バンド/45バンド×降水/50ベースライン) |
| [docs/dataset/data-specification.md](dataset/data-specification.md) | 全バンドの意味・波長・用途 |
| [docs/phase1/RUNBOOK.md](phase1/RUNBOOK.md) | Phase1 実行手順 |
| [docs/phase2/RUNBOOK.md](phase2/RUNBOOK.md) | Phase2 実行手順 |
| [docs/discussion/0708-lh24-summary.md](discussion/0708-lh24-summary.md) | 参加者LH24の0.708解法要約と突き合わせ |

---

## 7. コンペ規約の要点（提出時）
- 外部データセット禁止。利用可能ライセンス: CC0/CC-BY/MIT/BSD/Apache 2.0 のみ（**timm事前学習重みは個別確認**）。
- 入賞コードは前処理/学習/予測の3モジュール構成（本リポジトリは準拠）。
- 提出 = zip{ evaluation_target.csv + test_files/(各eval行 41×41 float32 tif, 名前=gpm_imerg_filename) }。
- Private LB(全評価データ)が最終順位。Public は35%。
