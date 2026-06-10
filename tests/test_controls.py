"""Tests for matched control sampling."""

import numpy as np
import pandas as pd

from gemfinder.controls import sample_controls

HORIZON = 63


def make_series(prices):
    idx = pd.bdate_range("2010-01-04", periods=len(prices))
    return pd.Series(prices, index=idx, name="close", dtype=float)


def build_universe():
    """WIN rallies +60% mid-series; FLAT1/FLAT2 stay flat; RALLY2 also rallies."""
    n = 400
    flat = [100.0] * n
    win = [100.0] * 200 + list(np.linspace(100, 160, 30)) + [160.0] * (n - 230)
    return {
        "WIN": make_series(win),
        "FLAT1": make_series(flat),
        "FLAT2": make_series(flat),
        "RALLY2": make_series(win),
    }


# first index whose 63-day forward window contains a +50% gain: the rally
# (indices 200-229) crosses 150 at index 225, so entries from 225-63=162 qualify
ENTRY_IDX = 162


def winners_df(series):
    return pd.DataFrame(
        [{"ticker": "WIN", "entry_date": series["WIN"].index[ENTRY_IDX]}]
    )


def test_controls_are_same_date_non_winners():
    series = build_universe()
    winners = winners_df(series)
    controls = sample_controls(
        winners, list(series), lambda t: series[t],
        threshold=0.50, horizon=HORIZON, per_winner=2, seed=1,
    )
    assert len(controls) == 2
    assert set(controls["ticker"]) <= {"FLAT1", "FLAT2"}  # never WIN or RALLY2
    assert (controls["date"] == winners["entry_date"].iloc[0]).all()
    assert (controls["max_fwd_return"] < 0.50).all()


def test_rallying_ticker_excluded_from_controls():
    series = build_universe()
    winners = winners_df(series)
    # RALLY2 qualifies as a winner on the same date, so with only rallying
    # candidates available there is nothing valid to sample
    only_rallies = {"WIN": series["WIN"], "RALLY2": series["RALLY2"]}
    controls = sample_controls(
        winners, list(only_rallies), lambda t: only_rallies[t],
        threshold=0.50, horizon=HORIZON, per_winner=2, seed=1,
    )
    assert len(controls) == 0


def test_no_duplicate_ticker_dates():
    series = build_universe()
    # two winners on the same date -> controls must not collide
    winners = pd.DataFrame(
        [
            {"ticker": "WIN", "entry_date": series["WIN"].index[ENTRY_IDX]},
            {"ticker": "RALLY2", "entry_date": series["WIN"].index[ENTRY_IDX]},
        ]
    )
    controls = sample_controls(
        winners, list(series), lambda t: series[t],
        threshold=0.50, horizon=HORIZON, per_winner=1, seed=1,
    )
    keys = list(zip(controls["ticker"], controls["date"]))
    assert len(keys) == len(set(keys))


def test_incomplete_forward_window_rejected():
    series = build_universe()
    # winner entry in the last `horizon` days: controls there would have
    # incomplete forward windows and must be rejected
    winners = pd.DataFrame(
        [{"ticker": "WIN", "entry_date": series["WIN"].index[-10]}]
    )
    controls = sample_controls(
        winners, list(series), lambda t: series[t],
        threshold=0.50, horizon=HORIZON, per_winner=2, seed=1,
    )
    assert len(controls) == 0


def test_deterministic_with_seed():
    series = build_universe()
    winners = winners_df(series)
    a = sample_controls(winners, list(series), lambda t: series[t], per_winner=2, seed=7)
    b = sample_controls(winners, list(series), lambda t: series[t], per_winner=2, seed=7)
    pd.testing.assert_frame_equal(a, b)
