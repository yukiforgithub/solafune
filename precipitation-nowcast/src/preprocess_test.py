"""前処理（評価用） — EVAL データをモデル入力形式へ変換する。

役割（コンペ要件の3モジュール構成のうちの「前処理」の評価側）:
    提供された EVAL データ（evaluation_target.csv + 衛星 tif、ターゲットは無し）を
    読み込み、predict.py が食える形式へ変換してキャッシュ出力する。
    train 側と同一の特徴生成ロジックを使うこと（学習・推論で前処理を厳密一致させる）。

将来の実装（Phase 別）:
    - Phase2 (GBDT): EVAL 各行の画素特徴テーブルを parquet 出力。
    - Phase3 (CNN):  EVAL 各行のパッチを npz/zarr 出力。
    - 共通: 欠測フレーム時のフォールバック（1〜2 フレームでも動く設計）。

Phase0 の責務:
    - EVAL メタ CSV を読み込み妥当性を検証（衛星種別・フレーム数分布・gpm_imerg_filename の一意性）。
    - 提出に必要な行数とファイル名の整合を確認し、最小テーブルを outputs/ に出力。

Phase1 物理 IR（method=physical_ir）の責務:
    - EVAL 各行の「最新フレーム」basename と n_frames を解決した推論マニフェスト
      outputs/preprocess_test_manifest.parquet を出力する（predict.py が同一規約で
      引けるよう、フレーム選択を前処理で固定）。0 フレーム行は気候値フォールバック対象。
    - train 側（充足統計）と同一のフレーム選択・バンド規約を使う（学習/推論一致）。

実行例:
    uv run python src/preprocess_test.py --help
    uv run python src/preprocess_test.py --limit 1000
    uv run python src/preprocess_test.py --method physical_ir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from precip import config, dataio  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="前処理（評価用）: EVAL データの検証と最小テーブル出力（Phase0 スタブ）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=config.DEFAULT_CONFIG_YAML, help="設定 YAML のパス。"
    )
    p.add_argument(
        "--out-dir", type=Path, default=config.OUTPUTS_DIR, help="出力先ディレクトリ。"
    )
    p.add_argument(
        "--method", choices=("constant", "physical_ir", "gbdt"), default=None,
        help="前処理対象の手法（未指定なら config の method）。physical_ir=推論マニフェスト出力 / "
             "gbdt=検証のみ（特徴は predict.py が実行時に抽出）。",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="先頭 N 行だけ処理（疎通確認用）。"
    )
    return p


def validate_eval_df(df: pd.DataFrame) -> dict:
    """EVAL メタ CSV の基本的な妥当性を検証し、サマリ dict を返す。"""
    issues: list[str] = []

    unknown_sat = set(df["satellite_target"].unique()) - set(config.SATELLITE_DIRNAMES)
    if unknown_sat:
        issues.append(f"未知の衛星種別: {sorted(unknown_sat)}")

    # 提出ファイル名はターゲット列で一意に決まる必要がある。
    if df["gpm_imerg_filename"].duplicated().any():
        issues.append("gpm_imerg_filename に重複があります（提出ファイル名が衝突します）。")

    summary = {
        "n_rows": int(len(df)),
        "n_locations": int(df["name_location"].nunique()),
        "satellites": sorted(df["satellite_target"].unique().tolist()),
        "n_frames_dist": {int(k): int(v) for k, v in df["n_frames"].value_counts().sort_index().items()},
        "issues": issues,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load_yaml_config(args.config)
    config.ensure_output_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess_test] EVAL CSV: {config.EVAL_CSV}")
    df = dataio.load_eval_df()
    if args.limit is not None:
        df = df.head(args.limit)

    summary = validate_eval_df(df)
    print(f"[preprocess_test] 行数={summary['n_rows']} 地域数={summary['n_locations']} "
          f"衛星={summary['satellites']}")
    print(f"[preprocess_test] フレーム数分布={summary['n_frames_dist']}")
    if summary["issues"]:
        print(f"[preprocess_test] 警告: {summary['issues']}")

    agg = (
        df.groupby(["name_location", "satellite_target", "n_frames"])
        .size()
        .reset_index(name="count")
        .sort_values(["name_location", "n_frames"])
    )
    out_path = out_dir / "preprocess_test_summary.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"[preprocess_test] 検証テーブルを書き出し: {out_path}  (cfg keys={list(cfg)})")

    # --- Phase2 GBDT: 特徴は predict.py が最新フレームから実行時抽出するため検証のみ ---
    method = args.method or cfg.get("method", "constant")
    if method == "gbdt":
        n_zero = summary["n_frames_dist"].get(0, 0)
        print(f"[preprocess_test] gbdt: 特徴は predict.py が実行時抽出。0フレーム行={n_zero}"
              f"（気候値フォールバック対象）。検証のみで完了。")
        return 0

    # --- Phase1 物理 IR: 推論マニフェスト（最新フレーム basename 解決）を出力 ---
    if method == "physical_ir":
        # フレーム選択規約「最新（リスト最後）」を前処理で固定。0 フレームは空文字。
        latest = df["frame_list"].map(lambda fl: fl[-1] if fl else "")
        manifest = pd.DataFrame(
            {
                "unique_id": df["unique_id"].to_numpy(),
                "name_location": df["name_location"].to_numpy(),
                "satellite_target": df["satellite_target"].to_numpy(),
                "gpm_imerg_filename": df["gpm_imerg_filename"].to_numpy(),
                "n_frames": df["n_frames"].to_numpy(),
                "latest_frame": latest.to_numpy(),
            }
        )
        man_path = out_dir / "preprocess_test_manifest.parquet"
        manifest.to_parquet(man_path, index=False)
        n_zero = int((manifest["n_frames"] == 0).sum())
        print(f"[preprocess_test] 推論マニフェストを書き出し: {man_path} "
              f"({len(manifest)} 行, 0 フレーム={n_zero} → 気候値フォールバック)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
