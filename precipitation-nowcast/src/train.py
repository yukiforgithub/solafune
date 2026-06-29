"""学習 — Phase0 の定数ベースラインを学習し outputs/ に成果物を保存する。

役割（コンペ要件の3モジュール構成のうちの「学習」）:
    前処理済みデータを読み込みモデルを学習し、重み・統計を保存する。
    predict.py はここで保存した成果物だけを読んで推論する（学習と推論の分離）。

Phase0 で学習する「モデル」は定数ベースライン:
    - zero          : 全画素 0
    - global_mean   : 全画素 = 訓練全体平均
    - location_mean : 地域別平均（未知地域は global_mean フォールバック）

学習は EDA キャッシュ（eda_cache/target_stats.parquet）の per-tif 集計
（vsum=画素値合計, vsq=二乗和, npix=画素数）から閉形式で行う。各 tif を
個別に開かずに全体平均・地域別平均・CV RMSE を厳密に計算できる。

Phase1 物理 IR モデル（method=physical_ir）:
    充足統計キャッシュ（eda_cache/phase1_window_hist.parquet）から衛星別 256bin
    の窓 DN→RR lookup（window_lookup / per_satellite）を fit し、conf/folds.yaml の
    手設計 fold で地域 GroupKFold CV RMSE を閉形式で再現する。確定 lookup は
    outputs/phase1_model.json に保存（predict.py が引く）。

成果物（定数ベースライン）:
    outputs/baseline_model.json  : 手法・global_mean・地域別平均・メタ。
    outputs/cv_scores.json       : fold 別 / 全体の CV RMSE（zero / global_mean / location_mean）。
    outputs/folds.parquet        : unique_id ごとの fold 割当（再利用用）。

成果物（physical_ir）:
    outputs/phase1_model.json    : 衛星別 256bin lookup table（確定）。
    outputs/phase1_train_cv.json : CV RMSE（overall / fold 別 / 衛星別）と gain。

実行例:
    uv run python src/train.py --help
    uv run python src/train.py --method location_mean
    uv run python src/train.py --method physical_ir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from precip import config, cv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="学習: Phase0 定数ベースラインを学習し成果物を保存。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG_YAML, help="設定 YAML のパス。")
    p.add_argument(
        "--method", choices=(*config.BASELINE_METHODS, "constant", "physical_ir", "gbdt", "cnn"), default=None,
        help="手法。定数(zero/global_mean/location_mean/constant) / physical_ir / gbdt / cnn。"
             "未指定なら config の method → baseline.method。",
    )
    p.add_argument(
        "--cv-scheme", choices=("location_group", "random"), default=None,
        help="CV 分割方式（未指定なら config の cv.scheme）。",
    )
    p.add_argument("--n-splits", type=int, default=None, help="fold 数（未指定なら config）。")
    p.add_argument("--seed", type=int, default=None, help="乱数シード（未指定なら config）。")
    p.add_argument(
        "--stats", type=Path, default=config.TARGET_STATS_PARQUET,
        help="ターゲット統計キャッシュ parquet のパス。",
    )
    p.add_argument("--out-dir", type=Path, default=config.OUTPUTS_DIR, help="成果物の出力先。")
    return p


def _weighted_group_mean(stats: pd.DataFrame, by: str | None = None) -> float | pd.Series:
    """画素重み付き平均 = sum(vsum)/sum(npix)。by 指定でグループ別 Series。"""
    if by is None:
        return float(stats["vsum"].sum() / stats["npix"].sum())
    g = stats.groupby(by)[["vsum", "npix"]].sum()
    return g["vsum"] / g["npix"]


def _fold_rmse_for_constant(val: pd.DataFrame, const_map: pd.Series | float, global_mean: float) -> float:
    """検証 fold に定数（地域別 or 全体）を当てたときの画素レベル RMSE（閉形式）。

    RMSE^2 = sum( vsq - 2*c*vsum + c^2*npix ) / sum(npix)
    （c は行ごとに与えられる定数。地域別の場合は地域→平均、未知地域は global_mean。）
    """
    if isinstance(const_map, pd.Series):
        c = val["name_location"].map(const_map).fillna(global_mean).to_numpy(dtype=np.float64)
    else:
        c = np.full(len(val), float(const_map), dtype=np.float64)
    vsq = val["vsq"].to_numpy(dtype=np.float64)
    vsum = val["vsum"].to_numpy(dtype=np.float64)
    npix = val["npix"].to_numpy(dtype=np.float64)
    sse = np.sum(vsq - 2.0 * c * vsum + c * c * npix)
    return float(np.sqrt(sse / npix.sum()))


def compute_cv_scores(stats: pd.DataFrame, n_splits: int) -> dict:
    """fold 別・全体の CV RMSE を全手法（zero/global_mean/location_mean）で計算。

    各手法とも、検証 fold を除いた学習 fold から定数を推定し（リーク防止）、
    検証 fold に適用して RMSE を求める。
    """
    results: dict[str, dict] = {m: {"per_fold": [], "overall": None} for m in config.BASELINE_METHODS}

    # 全体（OOF 連結）の SSE 累積で overall RMSE を厳密化。
    oof_sse = {m: 0.0 for m in config.BASELINE_METHODS}
    oof_npix = 0.0

    for f in range(n_splits):
        trn = stats[stats["fold"] != f]
        val = stats[stats["fold"] == f]
        if len(val) == 0:
            for m in config.BASELINE_METHODS:
                results[m]["per_fold"].append(None)
            continue

        gmean = _weighted_group_mean(trn)               # 学習 fold の全体平均
        loc_mean = _weighted_group_mean(trn, "name_location")  # 学習 fold の地域別平均

        per_method_const = {
            "zero": 0.0,
            "global_mean": gmean,
            "location_mean": loc_mean,
        }
        val_npix = float(val["npix"].sum())
        oof_npix += val_npix
        for m, const in per_method_const.items():
            r = _fold_rmse_for_constant(val, const, gmean)
            results[m]["per_fold"].append(r)
            # overall 用に SSE を復元して累積。
            oof_sse[m] += (r * r) * val_npix

    for m in config.BASELINE_METHODS:
        results[m]["overall"] = float(np.sqrt(oof_sse[m] / oof_npix)) if oof_npix > 0 else None
    return results


def train_physical_ir(cfg: dict, out_dir: Path) -> dict:
    """Phase1 物理 IR モデルを充足統計キャッシュから fit / CV し成果物を保存する。

    手順:
      1. 窓 DN ヒストグラム（phase1_window_hist.parquet）を読む。
      2. conf/folds.yaml の手設計 fold で window_lookup/per_satellite の閉形式 CV RMSE。
      3. 全 TRAIN から衛星別 256bin lookup を fit（確定モデル）。
      4. outputs/phase1_model.json（既存と同型）と phase1_train_cv.json を書く。

    Returns:
        サマリ dict（cv_rmse, per_fold, per_satellite, model_path 等）。
    """
    from precip import cv as cvmod  # noqa: E402
    from precip import phase1_fit  # noqa: E402
    from precip.physical import PHASE1_MODEL_JSON  # noqa: E402
    from precip.phase1_suffstats import SPLIT_BAND_INDEX, WINDOW_BAND_INDEX  # noqa: E402

    pcfg = cfg.get("physical_ir", {}) or {}
    win_hist_path = config.REPO_ROOT / pcfg.get(
        "window_hist", "eda_cache/phase1_window_hist.parquet"
    )
    folds_yaml = config.REPO_ROOT / pcfg.get("folds_yaml", "conf/folds.yaml")
    isotonic = bool(pcfg.get("isotonic", True))
    fallback = float(pcfg.get("global_mean_fallback", phase1_fit.GLOBAL_MEAN_FALLBACK))
    model_json = config.REPO_ROOT / pcfg.get("model_json", "outputs/phase1_model.json")

    if not win_hist_path.exists():
        raise FileNotFoundError(
            f"窓 DN ヒストグラムキャッシュがありません: {win_hist_path}\n"
            f"先に `uv run python -m src.build_phase1_suffstats` を実行してください。"
        )
    win_hist = pd.read_parquet(win_hist_path)
    loc_to_fold = cvmod.load_handdesigned_folds(folds_yaml)
    n_splits = max(loc_to_fold.values()) + 1 if loc_to_fold else 5

    print(f"[train:physical_ir] 窓 DN ヒスト読込: {len(win_hist)} 行  "
          f"衛星={sorted(win_hist['satellite'].unique())}  fold 数={n_splits}")

    # 閉形式 CV RMSE（手設計 fold）。
    cv_res = phase1_fit.cv_rmse_window_lookup_per_satellite(
        win_hist, loc_to_fold, n_splits=n_splits, isotonic=isotonic, fallback=fallback
    )
    print(f"[train:physical_ir] CV RMSE overall={cv_res['overall']:.5f}  "
          f"per_fold={[round(x, 4) if x is not None else None for x in cv_res['per_fold']]}")
    print(f"[train:physical_ir] 衛星別 CV RMSE="
          f"{ {k: round(v, 5) for k, v in cv_res['per_satellite'].items()} }")

    # 全 TRAIN から確定 lookup を fit。
    tables = phase1_fit.fit_window_lookup_per_satellite(
        win_hist, isotonic=isotonic, fallback=fallback
    )
    sat_means = phase1_fit.satellite_means(win_hist)

    # gain 基準 = キャッシュ全体に「全体平均定数」を当てた RMSE（閉形式）。
    #   RMSE_globalmean^2 = Σsum_y2/N - (Σsum_y)^2/N^2  （c=Σy/N を当てた MSE）
    n_pix = float(win_hist["count"].sum())
    sum_y = float(win_hist["sum_y"].sum())
    sum_y2 = float(win_hist["sum_y2"].sum())
    gmean_value = sum_y / n_pix
    rmse_globalmean = float(np.sqrt(sum_y2 / n_pix - gmean_value * gmean_value))
    gain = rmse_globalmean - cv_res["overall"] if cv_res["overall"] is not None else None

    # outputs/phase1_model.json（既存と同型。推論器 PhysicalIRModel が読む）。
    model = {
        "model": "window_lookup",
        "feature": "window_dn",
        "scope": "per_satellite",
        "cv_rmse": cv_res["overall"],
        "gain_vs_globalmean": gain,
        "rmse_globalmean_ref": rmse_globalmean,
        "honest_groupkfold_globalmean": 1.4048,
        "window_band_index": dict(WINDOW_BAND_INDEX),
        "split_band_index": dict(SPLIT_BAND_INDEX),
        "global_mean_fallback": fallback,
        "per_satellite": {
            sat: {
                "kind": "window_lookup",
                "index_offset": 0,
                "satellite_mean": sat_means.get(sat),
                "table": [round(float(x), 6) for x in tbl],
            }
            for sat, tbl in tables.items()
        },
    }
    model_json.parent.mkdir(parents=True, exist_ok=True)
    with model_json.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    cv_path = out_dir / "phase1_train_cv.json"
    with cv_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "physical_ir",
                "model": "window_lookup",
                "scope": "per_satellite",
                "cv_scheme": "location_group_handdesigned",
                "n_splits": n_splits,
                "cv_rmse_overall": cv_res["overall"],
                "cv_rmse_per_fold": cv_res["per_fold"],
                "cv_rmse_per_satellite": cv_res["per_satellite"],
                "rmse_globalmean_ref": rmse_globalmean,
                "gain_vs_globalmean": gain,
                "n_pixels": cv_res["n_pixels"],
                "folds_yaml": str(folds_yaml),
                "window_hist": str(win_hist_path),
            },
            f, ensure_ascii=False, indent=2,
        )

    print(f"[train:physical_ir] 保存: {model_json}")
    print(f"[train:physical_ir] 保存: {cv_path}")
    print(f"[train:physical_ir] 全体平均原点 RMSE={rmse_globalmean:.5f}  gain={gain:.5f}")
    return {
        "model_path": str(model_json),
        "cv_path": str(cv_path),
        "cv_rmse": cv_res["overall"],
        "per_fold": cv_res["per_fold"],
        "per_satellite": cv_res["per_satellite"],
    }


def train_gbdt(cfg: dict, out_dir: Path) -> dict:
    """Phase2 GBDT を衛星別に CV 評価し、最良 variant の最終モデルを保存する。

    preprocess_train --method gbdt が作った衛星別特徴 parquet を読み、
    conf/folds.yaml の地域 GroupKFold で two_part / tweedie の CV RMSE を算出
    （overall / 衛星別 / 強雨条件付き y>=1,y>=5 / bias）。CV 最小の variant を選び、
    全データで最終モデルを fit して outputs/phase2_models/ に保存する。
    """
    import numpy as _np  # 局所 import（既存の np と同一だが明示）

    from precip import features, gbdt, metrics  # lightgbm はここで初めて読む

    gcfg = cfg.get("gbdt", {}) or {}
    variant_cfg = str(gcfg.get("variant", "two_part"))
    variants = ["two_part", "tweedie"] if variant_cfg == "both" else [variant_cfg]
    thr = float(gcfg.get("rain_threshold", 0.1))
    seed = int(gcfg.get("seed", 42))
    vpower = float(gcfg.get("tweedie_variance_power", 1.5))
    params_clf = dict(gcfg.get("lgb_classifier", {}) or {})
    params_reg = dict(gcfg.get("lgb_regressor", {}) or {})
    calibration = str(gcfg.get("calibration", "isotonic"))  # isotonic | none（後処理較正）
    ensemble_mode = str(gcfg.get("ensemble", "none"))  # blend | none（two_part×tweedie ブレンド）
    names = features.feature_names()

    import pyarrow.parquet as _pq

    from precip import cv as _cvmod

    # 存在する衛星キャッシュと行数（メタデータのみ。全読込せずメモリ節約）。
    present: dict[str, Path] = {}
    rowcounts: dict[str, int] = {}
    for sat in ("himawari", "goes", "meteosat"):
        p = out_dir / f"phase2_features_{sat}.parquet"
        if not p.exists():
            print(f"[train:gbdt] 警告: 特徴キャッシュなし、スキップ: {p}")
            continue
        present[sat] = p
        rowcounts[sat] = int(_pq.ParquetFile(str(p)).metadata.num_rows)
    if not present:
        raise FileNotFoundError(
            "GBDT 特徴キャッシュがありません。先に "
            "`uv run python src/preprocess_train.py --method gbdt` を実行してください。"
        )
    print(f"[train:gbdt] 行数: { {s: f'{n:,}' for s, n in rowcounts.items()} }")
    # 部分/smoke データ混入の検知（ある衛星だけ極端に少ない＝前処理が途中で死んだ等）。
    med = sorted(rowcounts.values())[len(rowcounts) // 2]
    for s, n in rowcounts.items():
        if n < 0.3 * med:
            print(f"[train:gbdt] ★警告: {s} の行数 {n:,} が他衛星(中央値 {med:,})より極端に少ない。"
                  f"前処理が途中で失敗した疑い → 再前処理を推奨。")
    loc_to_fold = _cvmod.load_handdesigned_folds()
    n_splits = (max(loc_to_fold.values()) + 1) if loc_to_fold else 5

    def _load(sat: str):
        dfp = pd.read_parquet(present[sat])
        X = dfp[names].to_numpy(dtype=_np.float32)
        y = dfp["y"].to_numpy(dtype=_np.float64)
        fold = dfp["fold"].to_numpy(dtype=int)
        return X, y, fold

    def _oof(variant: str, X, y, fold):
        oof = _np.full(len(y), _np.nan, dtype=_np.float64)
        for f in range(n_splits):
            tr = fold != f
            va = fold == f
            if va.sum() == 0 or tr.sum() == 0:
                continue
            if variant == "two_part":
                cb, rb = gbdt.fit_two_part(X[tr], y[tr], thr, params_clf, params_reg, seed)
                oof[va] = gbdt.predict_two_part(cb, rb, X[va])
            else:
                tb = gbdt.fit_tweedie(X[tr], y[tr], vpower, params_reg, seed)
                oof[va] = gbdt.predict_tweedie(tb, X[va])
        return oof

    # CV: 衛星を外側で1回だけ読み（メモリ節約）、各 variant の OOF を蓄積。
    # oof_by_sat は後処理較正（fold-out 評価＋最終LUT fit）のため (y,pred,fold) を衛星別に保持。
    acc: dict[str, dict] = {v: {"y": [], "p": [], "per_sat": {}, "oof_by_sat": {}} for v in variants}
    for sat in present:
        X, y, fold = _load(sat)
        print(f"[train:gbdt] {sat}: {len(y):,} 画素で CV...")
        for variant in variants:
            oof = _oof(variant, X, y, fold)
            mask = ~_np.isnan(oof)
            if not mask.any():
                print(f"[train:gbdt] 警告: {sat}/{variant} は検証画素なし（fold 不足）。スキップ。")
                continue
            acc[variant]["per_sat"][sat] = metrics.rmse(y[mask], oof[mask])
            acc[variant]["y"].append(y[mask])
            acc[variant]["p"].append(oof[mask])
            acc[variant]["oof_by_sat"][sat] = (y[mask], oof[mask], fold[mask])
        del X, y, fold

    cv_results: dict[str, dict] = {}
    for variant in variants:
        a = acc[variant]
        if not a["y"]:
            print(f"[train:gbdt] 警告: variant={variant} は評価画素なし。スキップ。")
            cv_results[variant] = {
                "overall": float("nan"), "per_satellite": a["per_sat"],
                "cond_rmse_ge1": None, "cond_rmse_ge5": None, "bias": None, "n_eval_pixels": 0,
            }
            continue
        ya = _np.concatenate(a["y"])
        pa = _np.concatenate(a["p"])
        cond1 = metrics.rmse(ya[ya >= 1.0], pa[ya >= 1.0]) if _np.any(ya >= 1.0) else None
        cond5 = metrics.rmse(ya[ya >= 5.0], pa[ya >= 5.0]) if _np.any(ya >= 5.0) else None
        cv_results[variant] = {
            "overall": metrics.rmse(ya, pa),
            "per_satellite": a["per_sat"],
            "cond_rmse_ge1": cond1,
            "cond_rmse_ge5": cond5,
            "bias": float(_np.mean(pa - ya)),
            "n_eval_pixels": int(len(ya)),
        }
        print(f"[train:gbdt] variant={variant} overall_RMSE={cv_results[variant]['overall']:.4f} "
              f"per_sat={ {k: round(v, 4) for k, v in a['per_sat'].items()} } "
              f"cond>=1={cond1 and round(cond1, 3)} cond>=5={cond5 and round(cond5, 3)} "
              f"bias={cv_results[variant]['bias']:+.4f}")

    # NaN（評価不能）variant を除いて最小 overall を採用。全 NaN なら先頭。
    finite = {v: r for v, r in cv_results.items() if r["overall"] == r["overall"]}
    selected = min(finite or cv_results, key=lambda v: cv_results[v]["overall"])
    print(f"[train:gbdt] 採用 variant={selected} (overall_RMSE={cv_results[selected]['overall']:.4f})")

    # --- 後処理較正（isotonic）: 採用 variant の OOF を fold-out 評価し最終 LUT を保存 ---
    cal_summary: dict | None = None
    if calibration == "isotonic" and acc[selected]["oof_by_sat"]:
        from precip import calibrate  # noqa: E402

        cal = calibrate.run(acc[selected]["oof_by_sat"], n_splits)
        b, a = cal["before"], cal["after"]
        cal_path = out_dir / "phase2_calibrators.json"
        with cal_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"method": "isotonic", "variant": selected, "per_satellite": cal["luts"]},
                f, ensure_ascii=False,
            )
        cal_summary = {
            "method": "isotonic",
            "variant": selected,
            "before": {k: v for k, v in b.items() if k != "luts"},
            "after": {k: v for k, v in a.items() if k != "luts"},
            "calibrators_path": str(cal_path),
        }
        gain = (b["overall"] - a["overall"]) if (b["overall"] and a["overall"]) else None
        print(f"[train:gbdt] 較正(isotonic) OOF RMSE: {b['overall']:.4f} -> {a['overall']:.4f} "
              f"(gain {gain:+.4f}) cond>=5 {b['cond_ge5'] and round(b['cond_ge5'],3)} -> "
              f"{a['cond_ge5'] and round(a['cond_ge5'],3)}  保存: {cal_path}")
    elif calibration == "isotonic":
        print("[train:gbdt] 較正スキップ: OOF が空。")

    # --- アンサンブル（two_part × tweedie ブレンド）: fold-out で単一を上回るか検証 ---
    # 較正の教訓: 採用可否は必ず地域 GroupKFold の fold-out OOF で判定（in-sample は不可）。
    use_blend = False
    blend_summary: dict | None = None
    other = "two_part" if selected == "tweedie" else "tweedie"
    both_present = (
        acc.get(selected, {}).get("oof_by_sat") and acc.get(other, {}).get("oof_by_sat")
    )
    if ensemble_mode == "blend" and both_present:
        from precip import ensemble  # noqa: E402

        bres = ensemble.evaluate(acc[selected]["oof_by_sat"], acc[other]["oof_by_sat"], n_splits)
        blend_overall = bres["blend"]["overall"]
        single_overall = cv_results[selected]["overall"]
        use_blend = blend_overall < single_overall  # fold-out で厳密に改善したときのみ採用
        blend_summary = {
            "primary": selected,
            "other": other,
            "weights": bres["weights"],
            "single_overall": single_overall,
            "blend_overall": blend_overall,
            "blend_cond_ge5": bres["blend"]["cond_ge5"],
            "single_cond_ge5": bres["primary"]["cond_ge5"],
            "use_blend": use_blend,
        }
        gain = single_overall - blend_overall
        print(f"[train:gbdt] ブレンド({selected}×{other}) fold-out OOF: 単一 {single_overall:.4f} -> "
              f"ブレンド {blend_overall:.4f} (gain {gain:+.4f}) "
              f"cond>=5 {bres['primary']['cond_ge5'] and round(bres['primary']['cond_ge5'],3)} -> "
              f"{bres['blend']['cond_ge5'] and round(bres['blend']['cond_ge5'],3)} "
              f"weights={ {k: round(v,3) for k,v in bres['weights'].items()} } "
              f"=> {'採用' if use_blend else '不採用(単一に劣る)'}")
    elif ensemble_mode == "blend":
        print("[train:gbdt] ブレンドスキップ: 両 variant の OOF が揃っていない（variant=both が必要）。")

    del acc  # OOF（両 variant 分）を解放してから最終 fit（RAM 7.7GB 対策）。

    # --- 最終モデル（採用 variant を全データで fit）と feature importance ---
    # ブレンド採用時は two_part / tweedie の両方を fit・保存（predict が per-sat 重みで結合）。
    models_dir = out_dir / "phase2_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    want_tp = (selected == "two_part") or use_blend
    want_tw = (selected == "tweedie") or use_blend
    importance_rows: list[dict] = []
    for sat in present:
        X, y, _fold = _load(sat)
        if want_tp:
            cb, rb = gbdt.fit_two_part(X, y, thr, params_clf, params_reg, seed)
            gbdt.save_booster(cb, models_dir / f"{sat}_clf.txt")
            for nm, g in zip(names, cb.feature_importance(importance_type="gain")):
                importance_rows.append({"satellite": sat, "submodel": "clf", "feature": nm, "gain": float(g)})
            if rb is not None:
                gbdt.save_booster(rb, models_dir / f"{sat}_reg.txt")
                for nm, g in zip(names, rb.feature_importance(importance_type="gain")):
                    importance_rows.append({"satellite": sat, "submodel": "reg", "feature": nm, "gain": float(g)})
        if want_tw:
            tb = gbdt.fit_tweedie(X, y, vpower, params_reg, seed)
            gbdt.save_booster(tb, models_dir / f"{sat}_tweedie.txt")
            for nm, g in zip(names, tb.feature_importance(importance_type="gain")):
                importance_rows.append({"satellite": sat, "submodel": "tweedie", "feature": nm, "gain": float(g)})
        del X, y, _fold

    cv_path = out_dir / "phase2_cv.json"
    with cv_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "gbdt",
                "variants": cv_results,
                "selected": selected,
                "n_splits": n_splits,
                "rain_threshold": thr,
                "calibration": cal_summary,
                "ensemble": blend_summary,
                "honest_groupkfold_globalmean": 1.4048,
                "phase1_physical_ir": 1.1644,
            },
            f, ensure_ascii=False, indent=2,
        )
    imp_path = out_dir / "phase2_feature_importance.csv"
    pd.DataFrame(importance_rows).to_csv(imp_path, index=False)
    # 採用 variant: ブレンド可なら "blend"（per-sat 重み付き two_part×tweedie）。
    sel_obj: dict = {
        "variant": "blend" if use_blend else selected,
        "cv_rmse_overall": (blend_summary["blend_overall"] if use_blend else cv_results[selected]["overall"]),
        "rain_threshold": thr,
        "models_dir": str(models_dir),
        "feature_names": names,
    }
    if use_blend:
        # blend = w·primary + (1-w)·other。predict は per-sat 重みで結合する。
        sel_obj["blend"] = {
            "primary": selected,
            "other": other,
            "weights": blend_summary["weights"],
        }
    sel_path = out_dir / "phase2_selected.json"
    with sel_path.open("w", encoding="utf-8") as f:
        json.dump(sel_obj, f, ensure_ascii=False, indent=2)
    print(f"[train:gbdt] 保存: {cv_path} / {imp_path} / {sel_path} / {models_dir}/")
    return {"selected": "blend" if use_blend else selected, "cv": cv_results}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load_yaml_config(args.config)
    cv_cfg = cfg.get("cv", {})
    base_cfg = cfg.get("baseline", {})

    # 手法解決: 明示 --method > config の top-level method > baseline.method。
    method = args.method or cfg.get("method") or base_cfg.get("method", config.DEFAULT_BASELINE_METHOD)

    config.ensure_output_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase2 GBDT 分岐 ---
    if method == "gbdt":
        train_gbdt(cfg, out_dir)
        return 0

    # --- Phase3 CNN 分岐 ---
    if method == "cnn":
        from precip import cnn
        cnn.train_cnn(cfg, out_dir)
        return 0

    # --- Phase1 物理 IR 分岐 ---
    if method == "physical_ir":
        train_physical_ir(cfg, out_dir)
        return 0

    # --- Phase0 定数ベースライン ---
    # "constant" は config の baseline.method を使う別名。
    if method == "constant":
        method = base_cfg.get("method", config.DEFAULT_BASELINE_METHOD)
    scheme = args.cv_scheme or cv_cfg.get("scheme", "location_group")
    n_splits = args.n_splits if args.n_splits is not None else int(cv_cfg.get("n_splits", config.DEFAULT_N_SPLITS))
    seed = args.seed if args.seed is not None else int(cv_cfg.get("seed", config.DEFAULT_SEED))

    stats_path = Path(args.stats)
    if not stats_path.exists():
        print(f"[train] エラー: 統計キャッシュが見つかりません: {stats_path}", file=sys.stderr)
        return 1
    stats = pd.read_parquet(stats_path)
    print(f"[train] 統計キャッシュ読込: {len(stats)} 行  手法={method} CV={scheme} "
          f"n_splits={n_splits} seed={seed}")

    # CV fold 付与。
    if scheme == "location_group":
        stats = cv.make_location_group_kfold(stats, n_splits=n_splits, seed=seed)
    else:
        stats = cv.random_kfold(stats, n_splits=n_splits, seed=seed)

    # 全データから最終モデル（定数）を推定 → predict.py が使う。
    global_mean = _weighted_group_mean(stats)
    location_mean = _weighted_group_mean(stats, "name_location")

    cv_scores = compute_cv_scores(stats, n_splits=n_splits)
    for m in config.BASELINE_METHODS:
        print(f"[train] CV RMSE  {m:14s} overall={cv_scores[m]['overall']:.4f}  "
              f"per_fold={[round(x, 4) if x is not None else None for x in cv_scores[m]['per_fold']]}")

    # 成果物保存。
    model = {
        "method": method,
        "global_mean": global_mean,
        "location_mean": {str(k): float(v) for k, v in location_mean.items()},
        "target_size": list(config.TARGET_SIZE),
        "meta": {
            "cv_scheme": scheme,
            "n_splits": n_splits,
            "seed": seed,
            "n_train_rows": int(len(stats)),
            "stats_source": str(stats_path),
        },
    }
    model_path = out_dir / "baseline_model.json"
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    scores_path = out_dir / "cv_scores.json"
    with scores_path.open("w", encoding="utf-8") as f:
        json.dump(cv_scores, f, ensure_ascii=False, indent=2)

    folds_path = out_dir / "folds.parquet"
    stats[["unique_id", "name_location", "satellite_target", "fold"]].to_parquet(folds_path, index=False)

    print(f"[train] 保存: {model_path}")
    print(f"[train] 保存: {scores_path}")
    print(f"[train] 保存: {folds_path}")
    print(f"[train] global_mean={global_mean:.4f}  地域数={len(location_mean)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
