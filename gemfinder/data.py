"""Price download with a local cache so re-runs don't re-hit Yahoo."""

import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache")
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def get_ohlcv(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Daily adjusted OHLCV for one ticker, cached on disk.

    Returns None when Yahoo has no data for the ticker (delisted/renamed),
    so callers can count and report coverage honestly.
    """
    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"ohlcv_{ticker}_{start}_{end}.csv"

    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return None if df.empty else df

    import yfinance as yf

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,  # adjusted for splits/dividends
                progress=False,
            )
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))

    if raw is None or raw.empty:
        # cache the miss too, so re-runs skip dead tickers instantly
        pd.DataFrame(columns=OHLCV_COLS).to_csv(path)
        return None

    if isinstance(raw.columns, pd.MultiIndex):  # yfinance multi-level columns
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[OHLCV_COLS].dropna(subset=["close"])
    df.to_csv(path)
    return df


def get_close_series(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
) -> pd.Series | None:
    df = get_ohlcv(ticker, start, end, cache_dir=cache_dir)
    if df is None:
        return None
    return df["close"].rename("close")
