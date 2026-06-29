"""予測 — 学習成果物と EVAL CSV を読み、提出 zip を生成する。

役割（コンペ要件の3モジュール構成のうちの「予測」）:
    train.py が outputs/ に保存したモデルと EVAL メタ CSV を読み込み、
    EVAL の各行につき 41x41 float32 1band tif を1枚生成して提出 zip を作る。

提出フォーマット:
    submissions/<name>.zip
    ├── evaluation_target.csv         (提供 EVAL CSV のコピー)
    └── test_files/
        └── {gpm_imerg_filename}      (各行 1 枚、ファイル名は行の値と完全一致)

Phase0 の予測は定数ベースライン:
    - zero          : 全画素 0
    - global_mean   : 全画素 = 訓練全体平均
    - location_mean : 学習済み地域別平均。未知地域（EVAL は TRAIN と DISJOINT）は
                      global_mean にフォールバックする。

Phase1 物理 IR（method=physical_ir）:
    EVAL 各行の最新フレームから IR 窓バンドを読み、cv2.INTER_AREA で 41x41 へ縮約し、
    衛星別 256bin lookup（outputs/phase1_model.json）を引いて RR 41x41 を得る。
    負値 0 クリップ。0 フレーム行・読み込み失敗は気候値フォールバック。
    EVAL 入力読込（約 29,090 行）は ThreadPoolExecutor で並列化する。

注意: EVAL ディレクトリ内の test_files/ には sample_submission の placeholder tif が
入っているが、これは正解ではない。予測は必ず本スクリプトが生成する。

実行例:
    uv run python src/predict.py --help
    uv run python src/predict.py --name baseline_location_mean --limit 50
    uv run python src/predict.py --method physical_ir --name phase1_physical_ir
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from precip import config, dataio  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="予測: 成果物と EVAL CSV から提出 zip を生成。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG_YAML, help="設定 YAML のパス。")
    p.add_argument(
        "--model", type=Path, default=None,
        help="モデル JSON。未指定なら手法に応じ baseline_model.json / phase1_model.json。",
    )
    p.add_argument(
        "--method", choices=(*config.BASELINE_METHODS, "constant", "physical_ir", "gbdt", "cnn"), default=None,
        help="予測手法（未指定なら config の method、なければモデル JSON の method）。",
    )
    p.add_argument("--name", type=str, default="baseline", help="提出名（出力 zip / 中間ディレクトリ名）。")
    p.add_argument("--out-dir", type=Path, default=config.SUBMISSIONS_DIR, help="提出物の出力先。")
    p.add_argument("--limit", type=int, default=None, help="先頭 N 行だけ生成（疎通確認用）。")
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="EVAL 入力読込のスレッド数（physical_ir のみ。未指定なら config）。",
    )
    p.add_argument("--no-zip", action="store_true", help="zip 化せず test_files/ のみ生成。")
    return p


def _predict_constant(method: str, location: str, global_mean: float, loc_mean: dict[str, float]) -> float:
    """1 行ぶんの定数予測値を返す。"""
    if method == "zero":
        return 0.0
    if method == "global_mean":
        return float(global_mean)
    if method == "location_mean":
        return float(loc_mean.get(location, global_mean))  # 未知地域は global_mean
    raise ValueError(f"未知の手法: {method}")


def _prepare_work_dir(out_dir: Path, name: str) -> tuple[Path, Path]:
    """提出作業ディレクトリ（work_dir / test_files）を作り直して返す。"""
    work_dir = out_dir / name
    test_files_dir = work_dir / "test_files"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    test_files_dir.mkdir(parents=True, exist_ok=True)
    return work_dir, test_files_dir


def _finalize_submission(out_dir: Path, work_dir: Path, test_files_dir: Path, name: str, no_zip: bool) -> int:
    """evaluation_target.csv を同梱し、必要なら zip 化する（提出形式を確定）。"""
    csv_dst = work_dir / "evaluation_target.csv"
    shutil.copyfile(config.EVAL_CSV, csv_dst)
    n_tif = sum(1 for _ in test_files_dir.iterdir())
    print(f"[predict] tif {n_tif} 枚を書き出し: {test_files_dir}")
    print(f"[predict] evaluation_target.csv をコピー: {csv_dst}")

    if no_zip:
        print("[predict] --no-zip 指定のため zip 化はスキップ。")
        return 0

    zip_path = out_dir / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_dst, "evaluation_target.csv")
        for tif in sorted(test_files_dir.iterdir()):
            zf.write(tif, f"test_files/{tif.name}")
    print(f"[predict] 提出 zip を生成: {zip_path}")
    return 0


def predict_constant(args, model: dict) -> int:
    """Phase0 定数ベースラインで提出を生成する。"""
    method = args.method or model.get("method", config.DEFAULT_BASELINE_METHOD)
    if method == "constant":
        method = model.get("method", config.DEFAULT_BASELINE_METHOD)
    global_mean = float(model["global_mean"])
    loc_mean = {str(k): float(v) for k, v in model.get("location_mean", {}).items()}
    h, w = config.TARGET_SIZE

    work_dir, test_files_dir = _prepare_work_dir(Path(args.out_dir), args.name)
    df = dataio.load_eval_df()
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"[predict] 手法={method} 行数={len(df)} global_mean={global_mean:.4f} 地域数={len(loc_mean)}")

    base_tile = np.zeros((h, w), dtype=np.float32)
    for row in df.itertuples(index=False):
        val = max(_predict_constant(method, row.name_location, global_mean, loc_mean), 0.0)
        tile = base_tile if val == 0.0 else np.full((h, w), val, dtype=np.float32)
        dataio.write_prediction_tif(test_files_dir / row.gpm_imerg_filename, tile)

    return _finalize_submission(Path(args.out_dir), work_dir, test_files_dir, args.name, args.no_zip)


def predict_physical_ir(args, cfg: dict) -> int:
    """Phase1 物理 IR モデル（window_lookup / 衛星別）で提出を生成する。

    EVAL 各行の最新フレームから IR 窓バンドを読み 41x41 へ INTER_AREA 縮約 → 衛星別
    lookup を引いて RR 41x41（負値 0 クリップ）。0 フレーム / 読込失敗は気候値
    フォールバック。EVAL 入力読込は ThreadPoolExecutor で並列化する。
    """
    from concurrent.futures import ThreadPoolExecutor

    from precip import physical  # noqa: E402

    pcfg = cfg.get("physical_ir", {}) or {}
    model_path = Path(args.model) if args.model else (
        config.REPO_ROOT / pcfg.get("model_json", "outputs/phase1_model.json")
    )
    if not model_path.exists():
        print(f"[predict] エラー: 物理モデルが見つかりません: {model_path}  先に "
              f"`uv run python src/train.py --method physical_ir` を実行してください。",
              file=sys.stderr)
        return 1
    model = physical.PhysicalIRModel.from_json(model_path)
    n_workers = args.n_workers if args.n_workers is not None else int(pcfg.get("n_workers", 8))

    work_dir, test_files_dir = _prepare_work_dir(Path(args.out_dir), args.name)
    df = dataio.load_eval_df()
    if args.limit is not None:
        df = df.head(args.limit)

    rows = [
        (r.satellite_target, r.frame_list, r.gpm_imerg_filename)
        for r in df.itertuples(index=False)
    ]
    n_zero = sum(1 for _, fl, _ in rows if not fl)
    print(f"[predict] 手法=physical_ir 行数={len(rows)} workers={n_workers} "
          f"0フレーム={n_zero}（気候値フォールバック {model.global_mean_fallback}） model={model_path}")

    def _one(task):
        sat, frames, out_name = task
        tile = physical.predict_row(model, sat, list(frames), train=False)
        dataio.write_prediction_tif(test_files_dir / out_name, tile)
        return 1

    # EVAL 入力読込（rasterio I/O は GIL を解放）を ThreadPool で並列化。
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for _ in ex.map(_one, rows, chunksize=64):
            done += 1
            if done % 5000 == 0:
                print(f"[predict] 進捗 {done}/{len(rows)}", flush=True)

    return _finalize_submission(Path(args.out_dir), work_dir, test_files_dir, args.name, args.no_zip)


def predict_gbdt(args, cfg: dict) -> int:
    """Phase2 GBDT（衛星別・選定 variant）で提出を生成する。

    EVAL 各行の最新フレーム→features.extract_features_for_row→衛星別モデルで
    画素予測→41x41 へ reshape（負値 0 クリップ）。0 フレーム/読込失敗/未学習衛星は
    気候値フォールバック。特徴抽出をスレッド並列（I/O 律速）、予測はメインスレッドで
    安全に実行（Booster の並列 predict を避ける）。チャンク処理でメモリを抑える。
    """
    from concurrent.futures import ThreadPoolExecutor

    from precip import calibrate, ensemble, features, gbdt  # lightgbm はここで初めて読む

    gcfg = cfg.get("gbdt", {}) or {}
    sel_path = config.OUTPUTS_DIR / "phase2_selected.json"
    if not sel_path.exists():
        print(f"[predict] エラー: {sel_path} がありません。先に "
              f"`uv run python src/train.py --method gbdt` を実行してください。", file=sys.stderr)
        return 1
    with sel_path.open("r", encoding="utf-8") as f:
        sel = json.load(f)
    variant = sel["variant"]
    blend_cfg = sel.get("blend") or {}  # variant=="blend" のとき {primary, other, weights}
    models_dir = Path(sel["models_dir"])
    fallback = float(gcfg.get("global_mean_fallback", 0.2886))
    n_workers = args.n_workers if args.n_workers is not None else int(gcfg.get("n_workers", 8))
    target = config.TARGET_SIZE

    # 衛星別モデル読込。
    models: dict[str, tuple] = {}
    for sat in ("himawari", "goes", "meteosat"):
        if variant == "two_part":
            clf_p = models_dir / f"{sat}_clf.txt"
            reg_p = models_dir / f"{sat}_reg.txt"
            if not clf_p.exists():
                continue
            cb = gbdt.load_booster(clf_p)
            rb = gbdt.load_booster(reg_p) if reg_p.exists() else None
            models[sat] = ("two_part", cb, rb)
        elif variant == "blend":
            # two_part(clf,reg) と tweedie の両方＋per-sat 重みで結合。
            clf_p = models_dir / f"{sat}_clf.txt"
            tw_p = models_dir / f"{sat}_tweedie.txt"
            if not (clf_p.exists() and tw_p.exists()):
                continue
            reg_p = models_dir / f"{sat}_reg.txt"
            cb = gbdt.load_booster(clf_p)
            rb = gbdt.load_booster(reg_p) if reg_p.exists() else None
            tb = gbdt.load_booster(tw_p)
            w = float(blend_cfg.get("weights", {}).get(sat, 1.0))
            models[sat] = ("blend", (cb, rb, tb, w, blend_cfg.get("primary", "tweedie")), None)
        else:
            tw_p = models_dir / f"{sat}_tweedie.txt"
            if not tw_p.exists():
                continue
            models[sat] = ("tweedie", gbdt.load_booster(tw_p), None)
    if not models:
        print(f"[predict] エラー: モデルが見つかりません: {models_dir}", file=sys.stderr)
        return 1

    # 後処理較正 LUT（任意）。phase2_calibrators.json があり config で無効化されていなければ適用。
    calibrators: dict[str, tuple] = {}
    if str(gcfg.get("calibration", "isotonic")) != "none":
        cal_path = config.OUTPUTS_DIR / "phase2_calibrators.json"
        if cal_path.exists():
            with cal_path.open("r", encoding="utf-8") as f:
                cal = json.load(f)
            for s, lut in cal.get("per_satellite", {}).items():
                calibrators[s] = (np.asarray(lut["x"], np.float64), np.asarray(lut["y"], np.float64))
            print(f"[predict] 後処理較正 isotonic 適用: {sorted(calibrators)} ({cal_path})")

    work_dir, test_files_dir = _prepare_work_dir(Path(args.out_dir), args.name)
    df = dataio.load_eval_df()
    if args.limit is not None:
        df = df.head(args.limit)
    dt = pd.to_datetime(df["datetime"])
    df = df.assign(hour_=dt.dt.hour.astype(int), month_=dt.dt.month.astype(int))
    rows = [
        (r.satellite_target, r.frame_list, int(r.hour_), int(r.month_), r.gpm_imerg_filename)
        for r in df.itertuples(index=False)
    ]
    n_zero = sum(1 for r in rows if not r[1])
    print(f"[predict] 手法=gbdt variant={variant} 行数={len(rows)} workers={n_workers} "
          f"0フレーム={n_zero}（気候値 {fallback}） models={models_dir}")

    def _extract(row):
        sat, frames, hour, month, name = row
        X = features.extract_features_for_row(sat, list(frames), hour, month, train=False)
        return name, sat, X

    chunk = 512
    done = 0
    for start in range(0, len(rows), chunk):
        batch = rows[start:start + chunk]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            extracted = list(ex.map(_extract, batch, chunksize=16))
        for name, sat, X in extracted:
            m = models.get(sat)
            if X is None or m is None:
                tile = np.full(target, max(fallback, 0.0), dtype=np.float32)
            else:
                kind, b1, b2 = m
                if kind == "two_part":
                    pred = gbdt.predict_two_part(b1, b2, X)
                elif kind == "blend":
                    cb, rb, tb, w, primary = b1
                    p_tp = gbdt.predict_two_part(cb, rb, X)
                    p_tw = gbdt.predict_tweedie(tb, X)
                    p_primary, p_other = (p_tw, p_tp) if primary == "tweedie" else (p_tp, p_tw)
                    pred = ensemble.blend(p_primary, p_other, w)
                else:
                    pred = gbdt.predict_tweedie(b1, X)
                cx = calibrators.get(sat)
                if cx is not None:
                    pred = calibrate.apply_lut(pred, cx[0], cx[1])
                tile = np.clip(pred.reshape(target), 0.0, None).astype(np.float32)
            dataio.write_prediction_tif(test_files_dir / name, tile)
            done += 1
        if done % 5000 < chunk:
            print(f"[predict] 進捗 {done}/{len(rows)}", flush=True)

    return _finalize_submission(Path(args.out_dir), work_dir, test_files_dir, args.name, args.no_zip)


def predict_cnn(args, cfg: dict) -> int:
    """Phase3 CNN（衛星別 FCN）で提出を生成する。

    EVAL 各行の直近最大3フレーム→cnn.build_cnn_input→衛星別 FCN で 41x41 を予測
    （log1p 出力→expm1→0クリップ）。0フレーム/読込失敗/未学習衛星は気候値フォールバック。
    特徴抽出をスレッド並列（I/O 律速）、forward は衛星別にバッチ化して CPU で実行。
    """
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from precip import cnn

    ccfg = cfg.get("cnn", {}) or {}
    sel_path = config.OUTPUTS_DIR / "phase3_selected.json"
    if not sel_path.exists():
        print(f"[predict] エラー: {sel_path} がありません。先に "
              f"`uv run python src/train.py --method cnn` を実行してください。", file=sys.stderr)
        return 1
    with sel_path.open("r", encoding="utf-8") as f:
        sel = json.load(f)
    models_dir = Path(sel["models_dir"])
    fallback = float(ccfg.get("global_mean_fallback", 0.2886))
    n_workers = args.n_workers if args.n_workers is not None else int(ccfg.get("n_workers", 8))
    batch = int(ccfg.get("batch_size", 64))
    torch.set_num_threads(int(ccfg.get("num_threads", 8)))
    target = config.TARGET_SIZE

    models = {}
    for sat in ("himawari", "goes", "meteosat"):
        mp = models_dir / f"{sat}.pt"
        if mp.exists():
            models[sat] = cnn.load_model(mp)
    if not models:
        print(f"[predict] エラー: モデルが見つかりません: {models_dir}", file=sys.stderr)
        return 1

    work_dir, test_files_dir = _prepare_work_dir(Path(args.out_dir), args.name)
    df = dataio.load_eval_df()
    if args.limit is not None:
        df = df.head(args.limit)
    rows = [(r.satellite_target, r.frame_list, r.gpm_imerg_filename) for r in df.itertuples(index=False)]
    n_zero = sum(1 for r in rows if not r[1])
    print(f"[predict] 手法=cnn 行数={len(rows)} workers={n_workers} models={sorted(models)} "
          f"0フレーム={n_zero}（気候値 {fallback}）")

    def _extract(row):
        sat, frames, name = row
        return name, sat, cnn.build_cnn_input(sat, list(frames), train=False)

    chunk = 256
    done = 0
    for start in range(0, len(rows), chunk):
        batch_rows = rows[start:start + chunk]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            extracted = list(ex.map(_extract, batch_rows, chunksize=16))
        results: dict = {}
        by_sat: dict = {}
        for name, sat, xi in extracted:
            if xi is None or sat not in models:
                results[name] = np.full(target, max(fallback, 0.0), dtype=np.float32)
            else:
                by_sat.setdefault(sat, ([], []))
                by_sat[sat][0].append(name)
                by_sat[sat][1].append(xi)
        for sat, (names_s, xis) in by_sat.items():
            model = models[sat]
            arr = np.stack(xis).astype(np.float32)
            with torch.no_grad():
                for s in range(0, len(arr), batch):
                    xb = torch.from_numpy(arr[s:s + batch])
                    pl = model(xb).numpy()[:, 0]
                    pr = np.clip(np.expm1(pl), 0.0, None).astype(np.float32)
                    for j in range(pr.shape[0]):
                        results[names_s[s + j]] = pr[j]
        for name, _sat, _xi in extracted:
            dataio.write_prediction_tif(test_files_dir / name, results[name])
            done += 1
        if done % 5000 < chunk:
            print(f"[predict] 進捗 {done}/{len(rows)}", flush=True)

    return _finalize_submission(Path(args.out_dir), work_dir, test_files_dir, args.name, args.no_zip)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load_yaml_config(args.config)

    config.ensure_output_dirs()

    # 手法解決: 明示 --method > config の top-level method > モデル JSON の method。
    method = args.method or cfg.get("method")

    # --- Phase2 GBDT 分岐 ---
    if method == "gbdt":
        return predict_gbdt(args, cfg)

    # --- Phase3 CNN 分岐 ---
    if method == "cnn":
        return predict_cnn(args, cfg)

    # --- Phase1 物理 IR 分岐 ---
    if method == "physical_ir":
        return predict_physical_ir(args, cfg)

    # --- Phase0 定数ベースライン ---
    model_path = Path(args.model) if args.model else (config.OUTPUTS_DIR / "baseline_model.json")
    if not model_path.exists():
        print(f"[predict] エラー: モデルが見つかりません: {model_path}  先に train.py を実行してください。",
              file=sys.stderr)
        return 1
    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    return predict_constant(args, model)


if __name__ == "__main__":
    raise SystemExit(main())
