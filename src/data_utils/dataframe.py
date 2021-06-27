import logging
import os

import pandas as pd

logging.getLogger(__name__)


def load_dataframe(fpath: str) -> pd.DataFrame:
    """Prepare dataframe"""
    assert os.path.isfile(fpath)
    return pd.read_csv(fpath, sep=',', encoding='utf-8')
