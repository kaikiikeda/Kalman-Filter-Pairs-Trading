"""Download adjusted daily closes for the 8 candidate tickers into data/prices.csv.

Usage:  python scripts/download_data.py
Requires `yfinance`. Prices are split- and dividend-adjusted (auto_adjust=True), the
same series the original notebooks used. Yahoo revises adjusted prices when new
dividends are paid, which rescales history by a constant factor per ticker; that moves
the regression intercept but not the hedge ratio, spreads or returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs.data import DATA, TICKERS  # noqa: E402


def main():
    import yfinance as yf

    px = yf.download(TICKERS, start="2016-01-01", end="2026-01-01", interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna()[sorted(TICKERS)]
    px.index.name = "Date"
    DATA.parent.mkdir(exist_ok=True)
    px.round(6).to_csv(DATA)
    print(f"wrote {DATA} with {len(px)} rows, {px.index[0].date()} .. {px.index[-1].date()}")
    assert len(px) == 2514, "expected 2,514 trading days for 2016-2025"


if __name__ == "__main__":
    main()
