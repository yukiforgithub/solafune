from __future__ import annotations

from pathlib import Path
import yaml


def load_config(case_name: str, conf_dir: str | Path = '../conf') -> dict:
    """
    conf/parameter_{case_name}.yml を読み込んで辞書で返す。

    Parameters
    ----------
    case_name : YAML ファイル名の {case_name} 部分
    conf_dir  : conf/ ディレクトリへのパス（ノートブックからは '../conf'）
    """
    path = Path(conf_dir) / f'parameter_{case_name}.yml'
    with open(path) as f:
        return yaml.safe_load(f)
