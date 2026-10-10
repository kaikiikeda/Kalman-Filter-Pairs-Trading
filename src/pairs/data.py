"""Price data and the train/test split used throughout the study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "prices.csv"

TICKERS = ["KO", "PEP", "V", "MA", "XOM", "CVX", "HD", "LOW"]
CANDIDATE_PAIRS = [("KO", "PEP"), ("V", "MA"), ("XOM", "CVX"), ("HD", "LOW")]

START, END = "2016-01-01", "2025-12-31"
TRAIN = ("2016-01-01", "2021-12-31")  # 1,511 trading days: model fitting + pair selection
TEST = ("2022-01-01", "2025-12-31")  # 1,003 trading days: untouched until the final backtest


def load_prices() -> pd.DataFrame:
    """Split- and dividend-adjusted daily closes (Yahoo Finance), one column per ticker."""
    if not DATA.exists():
        raise FileNotFoundError("data/prices.csv not found; run `python scripts/download_data.py`")
    return pd.read_csv(DATA, index_col=0, parse_dates=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.loc[TRAIN[0]:TRAIN[1]], df.loc[TEST[0]:TEST[1]]


def log_prices(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df)
