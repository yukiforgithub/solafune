import pandas as pd

TARGET = 'construction_cost_per_m2_usd'

DROP_COLS = [
    'data_id', 'quarter_label',
    'sentinel2_tiff_file_name', 'viirs_tiff_file_name',
]

CAT_COLS = [
    'country', 'geolocation_name', 'developed_country', 'landlocked',
    'region_economic_classification', 'access_to_airport',
    'access_to_port', 'access_to_highway', 'access_to_railway',
    'seismic_hazard_zone', 'flood_risk_class',
    'tropical_cyclone_wind_risk', 'tornadoes_wind_risk',
    'koppen_climate_zone',
]


def prepare_features(
    df: pd.DataFrame,
    drop_cols: list[str] = DROP_COLS,
    cat_cols: list[str] = CAT_COLS,
    extra_drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    特徴量 DataFrame を返す。

    Parameters
    ----------
    df : 元の DataFrame（TARGET 列があっても無くても可）
    drop_cols : 除外する列（デフォルトは DROP_COLS）
    cat_cols : カテゴリカル列（文字列化 + 欠損補完を行う）
    extra_drop_cols : 追加で除外する列（実験ごとに指定する場合）
    """
    df = df.copy()
    df['quarter_num'] = df['quarter_label'].str.extract(r'Q(\d)').astype(int)

    exclude = set(drop_cols + [TARGET])
    if extra_drop_cols:
        exclude |= set(extra_drop_cols)

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna('missing').astype(str)

    return X
