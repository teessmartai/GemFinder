"""Price download with a local cache so re-runs don't re-hit Yahoo."""

import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache")


def _cache_path(ticker: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{ticker}_{start}_{end}.csv"


def get_close_series(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
    max_retries: int = 3,
) -> pd.Series | None:
    """Daily split/dividend-adjusted closes for one ticker, cached on disk.

    Returns None when Yahoo has no data for the ticker (delisted/renamed),
    so callers can count and report coverage honestly.
    """
    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker}_{start}_{end}.csv"

    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        return df["close"]

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
        pd.DataFrame(columns=["close"]).to_csv(path)
        return None

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance may return multi-level cols
        close = close.iloc[:, 0]
    close = close.dropna().rename("close")
    close.to_frame().to_csv(path)
    return close
