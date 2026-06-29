"""パス定数・衛星仕様・成果物ディレクトリの一元管理。

リポジトリルートを基準に全パスを絶対パスで解決する。データセットの
ハッシュ付きディレクトリ名はここだけで管理し、他モジュールは本モジュールの
定数を参照する（将来データ版が変わってもここ1箇所の修正で済む）。
"""

from __future__ import annotations

from pathlib import Path

# --- リポジトリルート（このファイルは src/precip/config.py） ---
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# --- データ置き場（gitignore 済み） ---
DATA_DIR: Path = REPO_ROOT / "data"

# データセット本体（ハッシュ付きディレクトリ名はここで吸収する）
TRAIN_DIR: Path = DATA_DIR / "train_dataset_b1c74968f2f24eaeb2852b47b80a581e"
EVAL_DIR: Path = DATA_DIR / "evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d"
SAMPLE_SUB_DIR: Path = DATA_DIR / "sample_submission_95c3b1e094034f5fbba421f5e5310f8a"

# --- CSV（表形式メタデータ） ---
# train は train_dataset.csv、eval/sample は evaluation_target.csv とファイル名が異なる点に注意。
TRAIN_CSV: Path = TRAIN_DIR / "train_dataset.csv"
EVAL_CSV: Path = EVAL_DIR / "evaluation_target.csv"
SAMPLE_SUB_CSV: Path = SAMPLE_SUB_DIR / "evaluation_target.csv"

# --- 入力衛星画像ディレクトリ（衛星名 = サブディレクトリ名） ---
# train / eval ともに同じ3衛星ぶんのサブディレクトリを持つ。
SATELLITE_DIRNAMES: dict[str, str] = {
    "himawari": "himawari",
    "goes": "goes",
    "meteosat": "meteosat",
}

# ターゲット GPM-IMERG GeoTIFF は train のみ提供される（eval は予測対象なので無い）。
TRAIN_TARGET_DIR: Path = TRAIN_DIR / "gpm_imerg"

# --- 成果物ディレクトリ（gitignore 済み） ---
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"       # 学習成果物（モデル重み・統計・fold 表など）
SUBMISSIONS_DIR: Path = REPO_ROOT / "submissions"  # 提出 zip と中間 test_files/
EDA_CACHE_DIR: Path = REPO_ROOT / "eda_cache"   # EDA キャッシュ（target_stats.parquet 等）

# EDA で構築済みのターゲット統計キャッシュ（TRAIN の各ターゲット tif 1 行）。
TARGET_STATS_PARQUET: Path = EDA_CACHE_DIR / "target_stats.parquet"

# --- ターゲット仕様 ---
# GPM-IMERG ターゲットは常に 1band / 41x41 / float32 / NaN無し / 負値無し / 単位 mm/hr。
TARGET_SIZE: tuple[int, int] = (41, 41)  # (H, W)
N_TARGET_BANDS: int = 1

# --- 衛星バンド仕様（16band, uint8, CRS/ジオトランスフォーム無し） ---
# 衛星別の入力画像サイズ（H, W）。入力とターゲットは同一地理 ROI を異なる画素格子で表す。
SATELLITE_INPUT_SIZE: dict[str, tuple[int, int]] = {
    "himawari": (81, 81),
    "goes": (141, 141),
    "meteosat": (144, 144),
}

N_INPUT_BANDS: int = 16

# 衛星別バンド名（読み込み後の band 軸 0..15 に対応する論理名）。
SATELLITE_BANDS: dict[str, list[str]] = {
    "himawari": [f"B{i:02d}" for i in range(1, 17)],   # B01..B16
    "goes": [f"C{i:02d}" for i in range(1, 17)],        # C01..C16
    "meteosat": [
        "vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
        "nir_13", "nir_16", "nir_22",
        "ir_38", "wv_63", "wv_73", "ir_87", "ir_97",
        "ir_105", "ir_123", "ir_133",
    ],
}

# 想定入力フレーム数（直近30分・10分間隔・最大3枚）。train には 0〜3 枚の行が混在する。
MAX_INPUT_FRAMES: int = 3

# --- 既定のハイパーパラメータ（conf/config.yaml で上書き可能） ---
DEFAULT_N_SPLITS: int = 5
DEFAULT_SEED: int = 42

# Phase0 の定数ベースライン手法。
#   "zero"          : 全画素 0
#   "global_mean"   : 全画素 = 訓練全体平均
#   "location_mean" : 地域別平均（未知地域は global_mean にフォールバック）
BASELINE_METHODS: tuple[str, ...] = ("zero", "global_mean", "location_mean")
DEFAULT_BASELINE_METHOD: str = "location_mean"


def ensure_output_dirs() -> None:
    """成果物ディレクトリ（outputs/ submissions/ eda_cache/）を作成する。"""
    for d in (OUTPUTS_DIR, SUBMISSIONS_DIR, EDA_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# 既定の設定ファイル（conf/config.yaml）。
DEFAULT_CONFIG_YAML: Path = REPO_ROOT / "conf" / "config.yaml"


def load_yaml_config(path: Path | str | None = None) -> dict:
    """YAML 設定を読み込む。存在しなければ空 dict を返す。

    Args:
        path: 設定ファイルパス。None なら DEFAULT_CONFIG_YAML。

    Returns:
        パース済みの dict（ファイルが無い場合は空）。
    """
    import yaml  # 遅延 import（設定不要なコードパスで依存を増やさない）

    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_YAML
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
