"""前処理（学習用） — TRAIN データをモデル入力形式へ変換する。

役割（コンペ要件の3モジュール構成のうちの「前処理」）:
    提供された TRAIN データ（train_dataset.csv + 衛星 tif + GPM tif）を読み込み、
    各 Phase のモデルが直接食える形式へ変換してキャッシュ出力する。

将来の実装（Phase 別）:
    - Phase2 (GBDT): 画素特徴テーブル（16band + BTD + 近傍統計 + メタ）を parquet 出力。
    - Phase3 (CNN):  ROI を固定サイズパッチに揃えた npz/zarr を出力。
    - 共通: 無降水画素のダウンサンプル、log1p ターゲット、欠測マスクの付与。

Phase0 の責務:
    - TRAIN メタ CSV を読み込み妥当性を検証（衛星種別・フレーム数分布・ターゲット tif の存在）。
    - 最小の検証テーブル（地域×衛星×フレーム数の集計）を outputs/ に出力。

Phase1 物理 IR（method=physical_ir）の責務:
    - 物理特徴（窓 DN / split 差）の充足統計キャッシュ
      eda_cache/phase1_window_hist.parquet の存在を確認する（無ければ
      build_phase1_suffstats の実行を促す）。学習・推論はこのキャッシュ起点。
    - フレーム選択規約（最新フレーム・0 フレーム行のフォールバック対象）の集計を出力。

実行例:
    uv run python src/preprocess_train.py --help
    uv run python src/preprocess_train.py --limit 1000
    uv run python src/preprocess_train.py --method physical_ir
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# src/ をパスに追加し、cwd に依存せず precip を import 可能にする。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from precip import config, cv as cvmod, dataio, features  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="前処理（学習用）: TRAIN データの検証と最小テーブル出力（Phase0 スタブ）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=config.DEFAULT_CONFIG_YAML, help="設定 YAML のパス。"
    )
    p.add_argument(
        "--out-dir", type=Path, default=config.OUTPUTS_DIR, help="出力先ディレクトリ。"
    )
    p.add_argument(
        "--method", choices=("constant", "physical_ir", "gbdt", "cnn"), default=None,
        help="前処理対象の手法（未指定なら config の method）。physical_ir=物理特徴キャッシュ確認 / "
             "gbdt=衛星別の画素特徴テーブルを構築 / cnn=衛星別の入力テンソル memmap を構築。",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="先頭 N 行だけ処理（疎通確認用）。"
    )
    return p


def validate_train_df(df: pd.DataFrame) -> dict:
    """TRAIN メタ CSV の基本的な妥当性を検証し、サマリ dict を返す。"""
    issues: list[str] = []

    unknown_sat = set(df["satellite_target"].unique()) - set(config.SATELLITE_DIRNAMES)
    if unknown_sat:
        issues.append(f"未知の衛星種別: {sorted(unknown_sat)}")

    if df["unique_id"].duplicated().any():
        issues.append("unique_id に重複があります。")

    summary = {
        "n_rows": int(len(df)),
        "n_locations": int(df["name_location"].nunique()),
        "satellites": sorted(df["satellite_target"].unique().tolist()),
        "n_frames_dist": {int(k): int(v) for k, v in df["n_frames"].value_counts().sort_index().items()},
        "issues": issues,
    }
    return summary


def build_gbdt_features(df_full: pd.DataFrame, cfg: dict, out_dir: Path, *, limit: int | None = None) -> dict:
    """Phase2 GBDT 用の衛星別画素特徴テーブルを構築し parquet 保存する。

    各行の直近最大3フレームから features.extract_features_for_row で画素特徴(1681,97)を作り、
    base rate を保つ一様サンプリング（pixel_sample_frac）で間引いて y / fold とともに
    outputs/phase2_features_{satellite}.parquet へ保存する（衛星別 GBDT 用）。

    Args:
        df_full: load_train_df() の全行（frame_list / n_frames 付き）。
        cfg: 設定 dict（gbdt セクションを参照）。
        out_dir: 出力先。
        limit: 各衛星の先頭 N 行だけ処理（疎通確認用、None で全行）。

    Returns:
        衛星別の保存行数サマリ dict。
    """
    gcfg = cfg.get("gbdt", {}) or {}
    frac = float(gcfg.get("pixel_sample_frac", 0.25))
    max_pixels = int(gcfg.get("max_pixels", 6_000_000))
    seed = int(gcfg.get("seed", 42))
    n_workers = int(gcfg.get("n_workers", 8))

    loc_to_fold = cvmod.load_handdesigned_folds()
    dt = pd.to_datetime(df_full["datetime"])
    work = df_full.assign(
        hour_=dt.dt.hour.astype(int),
        month_=dt.dt.month.astype(int),
        fold_=df_full["name_location"].map(loc_to_fold),
    )
    # 未知地域（fold 未割当）は学習に使わない（TRAIN 20 地域は全て fold を持つ）。
    work = work[work["fold_"].notna()].copy()
    work["fold_"] = work["fold_"].astype(int)

    names = features.feature_names()
    (out_dir / "phase2_feature_names.json").write_text(
        json.dumps(names, ensure_ascii=False), encoding="utf-8"
    )

    npix_per_row = config.TARGET_SIZE[0] * config.TARGET_SIZE[1]

    def _worker(task):
        idx, satellite, frame_list, hour, month, fold, gpm_name, eff_frac = task
        X = features.extract_features_for_row(satellite, list(frame_list), hour, month, train=True)
        if X is None:
            return None
        try:
            y = dataio.read_target(config.TRAIN_TARGET_DIR / gpm_name).ravel()
        except Exception:
            return None
        if y.shape[0] != X.shape[0]:
            return None
        rng = np.random.default_rng(seed + int(idx))
        k = max(1, int(round(X.shape[0] * eff_frac)))
        sel = rng.choice(X.shape[0], size=k, replace=False)
        return X[sel], y[sel].astype(np.float32), np.full(k, int(fold), np.int16)

    summary: dict[str, int] = {}
    for sat in ("himawari", "goes", "meteosat"):
        sub = work[work["satellite_target"] == sat]
        if limit is not None:
            # 疎通確認用: 地域（fold）が偏らないよう head ではなくランダム抽出。
            sub = sub.sample(n=min(limit, len(sub)), random_state=seed)
        # 巨大配列を作ってから間引くと OOM するため、上限から逆算した実効サンプル率で
        # 最初から少なく取る（concat ピークを max_pixels 相当に抑える）。
        est_total = max(len(sub) * npix_per_row, 1)
        eff_frac = min(frac, max_pixels / est_total)
        tasks = [
            (i, sat, r.frame_list, int(r.hour_), int(r.month_), int(r.fold_), r.gpm_imerg_filename, eff_frac)
            for i, r in enumerate(sub.itertuples(index=False))
        ]
        if not tasks:
            print(f"[preprocess_train:gbdt] {sat}: 対象行なし、スキップ。")
            continue
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        fs: list[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for res in ex.map(_worker, tasks, chunksize=16):
                if res is None:
                    continue
                Xs.append(res[0])
                ys.append(res[1])
                fs.append(res[2])
        if not Xs:
            print(f"[preprocess_train:gbdt] {sat}: 有効画素なし、スキップ。")
            continue
        # メモリピーク半減（RAM 7.7GB対策）: np.concatenate は出力とXsが同時に生きて2倍に
        # なるため、事前確保した配列へチャンクを詰めつつ Xs を逐次解放（ピークを~1倍に）。
        n_total = sum(len(a) for a in Xs)
        n_col = Xs[0].shape[1]
        X = np.empty((n_total, n_col), dtype=np.float32)
        y = np.empty(n_total, dtype=np.float32)
        fold = np.empty(n_total, dtype=np.int16)
        pos = 0
        while Xs:
            xc = Xs.pop(0); yc = ys.pop(0); fc = fs.pop(0)
            m = len(xc)
            X[pos:pos + m] = xc
            y[pos:pos + m] = yc
            fold[pos:pos + m] = fc
            pos += m
            del xc, yc, fc
        del Xs, ys, fs
        if len(y) > max_pixels:
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(y), size=max_pixels, replace=False)
            X, y, fold = X[sel], y[sel], fold[sel]
        out = pd.DataFrame(X, columns=names)
        out["y"] = y
        out["fold"] = fold
        path = out_dir / f"phase2_features_{sat}.parquet"
        out.to_parquet(path, index=False)
        summary[sat] = int(len(y))
        rain_pct = float((y >= float(gcfg.get("rain_threshold", 0.1))).mean() * 100.0)
        print(f"[preprocess_train:gbdt] {sat}: {len(y):,} 画素 (有雨{rain_pct:.1f}%) -> {path}")
        # ★衛星の巨大配列(X/out ~2GB)を次の衛星の構築前に解放（RAM 7.7GB でのOOM対策）。
        #   これが無いと himawari の 2GB が goes の構築ピークと重なり OOM する。
        del X, y, fold, out
    print(f"[preprocess_train:gbdt] 特徴数={len(names)} 保存サマリ={summary}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load_yaml_config(args.config)  # 将来の前処理パラメータ用に読み込んでおく
    config.ensure_output_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess_train] TRAIN CSV: {config.TRAIN_CSV}")
    df_full = dataio.load_train_df()
    df = df_full.head(args.limit) if args.limit is not None else df_full

    summary = validate_train_df(df)
    print(f"[preprocess_train] 行数={summary['n_rows']} 地域数={summary['n_locations']} "
          f"衛星={summary['satellites']}")
    print(f"[preprocess_train] フレーム数分布={summary['n_frames_dist']}")
    if summary["issues"]:
        print(f"[preprocess_train] 警告: {summary['issues']}")

    # 最小の検証テーブル: 地域×衛星×フレーム数の集計。
    agg = (
        df.groupby(["name_location", "satellite_target", "n_frames"])
        .size()
        .reset_index(name="count")
        .sort_values(["name_location", "n_frames"])
    )
    out_path = out_dir / "preprocess_train_summary.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"[preprocess_train] 検証テーブルを書き出し: {out_path}  (cfg keys={list(cfg)})")

    # --- Phase2 GBDT: 衛星別の画素特徴テーブルを構築 ---
    method = args.method or cfg.get("method", "constant")
    if method == "gbdt":
        build_gbdt_features(df_full, cfg, out_dir, limit=args.limit)
        return 0

    # --- Phase3 CNN: 衛星別の入力テンソル memmap を構築 ---
    if method == "cnn":
        from precip import cnn
        cnn.build_memmaps(df_full, cfg, out_dir, limit=args.limit)
        return 0

    # --- Phase1 物理 IR: 物理特徴の充足統計キャッシュの存在確認 ---
    if method == "physical_ir":
        pcfg = cfg.get("physical_ir", {}) or {}
        win_hist = config.REPO_ROOT / pcfg.get(
            "window_hist", "eda_cache/phase1_window_hist.parquet"
        )
        if win_hist.exists():
            wh = pd.read_parquet(win_hist)
            print(f"[preprocess_train] 物理特徴キャッシュ OK: {win_hist} "
                  f"({len(wh)} 行, 衛星={sorted(wh['satellite'].unique())})")
        else:
            print(f"[preprocess_train] 警告: 物理特徴キャッシュがありません: {win_hist}\n"
                  f"  `uv run python -m src.build_phase1_suffstats` で構築してください "
                  f"（最新フレーム・INTER_AREA・衛星別 IR 窓バンドで窓 DN/split 差を集計）。")
        n_zero = summary["n_frames_dist"].get(0, 0)
        print(f"[preprocess_train] 0 フレーム行（気候値フォールバック対象）= {n_zero}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
