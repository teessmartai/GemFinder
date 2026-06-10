"""Matched control sampling.

A feature common among winners means nothing unless it is rarer among
stocks that did NOT rally. For each winner episode we sample K control
(ticker, date) pairs on the SAME calendar date — so a control group from
March 2009 is compared with winners from March 2009, keeping market-regime
effects from masquerading as stock-picking signals.

A valid control must have a complete forward window (no peeking at partial
futures) whose max forward return stays below the winner threshold.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from gemfinder.labeling import TRADING_DAYS_90, forward_returns


def sample_controls(
    winners: pd.DataFrame,
    universe: list[str],
    load_close: Callable[[str], pd.Series | None],
    threshold: float = 0.50,
    horizon: int = TRADING_DAYS_90,
    per_winner: int = 3,
    seed: int = 42,
    max_tries_per_control: int = 40,
) -> pd.DataFrame:
    """Sample non-winner (ticker, date) pairs matched to winners by date.

    `load_close` maps ticker -> adjusted close series (or None); forward
    returns per ticker are computed once and reused across winners.
    """
    rng = np.random.default_rng(seed)
    universe = sorted(set(universe))
    fwd_cache: dict[str, pd.DataFrame | None] = {}

    def fwd(ticker: str) -> pd.DataFrame | None:
        if ticker not in fwd_cache:
            close = load_close(ticker)
            if close is None or len(close.dropna()) <= horizon:
                fwd_cache[ticker] = None
            else:
                fwd_cache[ticker] = forward_returns(close.dropna(), horizon=horizon)
        return fwd_cache[ticker]

    rows = []
    used: set[tuple[str, pd.Timestamp]] = set()
    for _, w in winners.iterrows():
        date = pd.Timestamp(w["entry_date"])
        found, tries = 0, 0
        while found < per_winner and tries < max_tries_per_control * per_winner:
            tries += 1
            ticker = universe[int(rng.integers(len(universe)))]
            if ticker == w["ticker"] or (ticker, date) in used:
                continue
            f = fwd(ticker)
            if f is None or date not in f.index:
                continue
            row = f.loc[date]
            if np.isnan(row["max_fwd_return"]) or row["max_fwd_return"] >= threshold:
                continue
            used.add((ticker, date))
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "matched_winner": w["ticker"],
                    "max_fwd_return": float(row["max_fwd_return"]),
                    "horizon_return": float(row["horizon_return"]),
                }
            )
            found += 1

    return pd.DataFrame(rows)
