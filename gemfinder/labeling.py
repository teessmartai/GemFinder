"""Forward-return labeling: find days where a stock subsequently rallied.

Given a daily adjusted-close series, for every day t we look at the next
`horizon` trading days (t+1 .. t+horizon) and record:

  - max_fwd_return: best gain achievable, max(close[t+1..t+horizon]) / close[t] - 1
  - horizon_return: gain if held to the end,   close[t+horizon] / close[t] - 1
  - days_to_peak:   trading days from t to the day of the window maximum

A "winner" day is one where max_fwd_return >= threshold. Because windows
overlap, a single rally produces a long run of consecutive winner days;
merge_episodes() collapses those runs into one event per rally so the
output is one row per opportunity, not hundreds.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ~90 calendar days of US market time
TRADING_DAYS_90 = 63


@dataclass
class Episode:
    ticker: str
    entry_date: pd.Timestamp     # first day the forward window clears the threshold
    entry_price: float
    peak_date: pd.Timestamp      # day of the window maximum after entry_date
    peak_price: float
    max_return: float            # peak_price / entry_price - 1
    days_to_peak: int            # trading days from entry to peak
    horizon_return: float        # return if held exactly `horizon` days
    signal_days: int             # how many consecutive days qualified (rally "width")


def forward_returns(close: pd.Series, horizon: int = TRADING_DAYS_90) -> pd.DataFrame:
    """Per-day forward stats over the next `horizon` trading days.

    The last `horizon` rows have incomplete windows and are returned as NaN
    so partial future data never creates false labels.
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    max_fwd = np.full(n, np.nan)
    days_to_peak = np.full(n, np.nan)
    horizon_ret = np.full(n, np.nan)

    for i in range(n - horizon):
        window = c[i + 1 : i + 1 + horizon]
        j = int(np.argmax(window))
        max_fwd[i] = window[j] / c[i] - 1.0
        days_to_peak[i] = j + 1
        horizon_ret[i] = c[i + horizon] / c[i] - 1.0

    return pd.DataFrame(
        {
            "close": close,
            "max_fwd_return": max_fwd,
            "days_to_peak": days_to_peak,
            "horizon_return": horizon_ret,
        },
        index=close.index,
    )


def merge_episodes(
    ticker: str,
    fwd: pd.DataFrame,
    threshold: float = 0.50,
    horizon: int = TRADING_DAYS_90,
) -> list[Episode]:
    """Collapse consecutive qualifying days into one Episode per rally.

    Qualifying days less than `horizon` trading days apart belong to the
    same rally (their forward windows overlap); a larger gap starts a new
    episode. Entry is the *first* qualifying day — the earliest moment the
    move was still fully ahead of you.
    """
    hits = fwd.index[fwd["max_fwd_return"] >= threshold]
    if len(hits) == 0:
        return []

    positions = fwd.index.get_indexer(hits)
    episodes: list[Episode] = []
    start = 0
    for k in range(1, len(positions) + 1):
        if k == len(positions) or positions[k] - positions[k - 1] >= horizon:
            entry_pos = positions[start]
            entry_date = fwd.index[entry_pos]
            row = fwd.iloc[entry_pos]
            peak_pos = entry_pos + int(row["days_to_peak"])
            episodes.append(
                Episode(
                    ticker=ticker,
                    entry_date=entry_date,
                    entry_price=float(row["close"]),
                    peak_date=fwd.index[peak_pos],
                    peak_price=float(fwd["close"].iloc[peak_pos]),
                    max_return=float(row["max_fwd_return"]),
                    days_to_peak=int(row["days_to_peak"]),
                    horizon_return=float(row["horizon_return"]),
                    signal_days=k - start,
                )
            )
            start = k
    return episodes


def find_winners(
    ticker: str,
    close: pd.Series,
    threshold: float = 0.50,
    horizon: int = TRADING_DAYS_90,
) -> list[Episode]:
    """Full pipeline for one ticker: forward stats -> merged episodes."""
    close = close.dropna()
    if len(close) <= horizon:
        return []
    fwd = forward_returns(close, horizon=horizon)
    return merge_episodes(ticker, fwd, threshold=threshold, horizon=horizon)
