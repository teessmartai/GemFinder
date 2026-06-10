"""Labeling tests on synthetic prices (no network needed)."""

import numpy as np
import pandas as pd
import pytest

from gemfinder.labeling import find_winners, forward_returns, merge_episodes


def make_series(prices):
    idx = pd.bdate_range("2010-01-04", periods=len(prices))
    return pd.Series(prices, index=idx, name="close", dtype=float)


def flat(n, level=100.0):
    return [level] * n


def brute_force_first_entry(prices, threshold, horizon):
    """Independent reference: first i whose next-`horizon` window gains >= threshold."""
    for i in range(len(prices) - horizon):
        window = prices[i + 1 : i + 1 + horizon]
        if max(window) / prices[i] - 1 >= threshold:
            return i
    return None


def test_detects_60pct_rally_within_window():
    # flat at 100, then a straight run to 160 over 30 days, then flat
    prices = flat(100) + list(np.linspace(100, 160, 30)) + flat(100, 160)
    s = make_series(prices)
    eps = find_winners("TEST", s, threshold=0.50, horizon=63)
    assert len(eps) == 1
    e = eps[0]
    # entry is the FIRST day from which +50% was achievable, so its window
    # return clears the threshold but need not capture the full +60% rally
    expected_entry = brute_force_first_entry(prices, 0.50, 63)
    assert e.entry_date == s.index[expected_entry]
    assert e.max_return >= 0.50
    assert e.peak_price <= 160.0
    assert 1 <= e.days_to_peak <= 63
    assert e.peak_price == pytest.approx(prices[expected_entry + e.days_to_peak])


def test_ignores_rally_below_threshold():
    prices = flat(100) + list(np.linspace(100, 140, 30)) + flat(100, 140)  # +40%
    eps = find_winners("TEST", make_series(prices), threshold=0.50, horizon=63)
    assert eps == []


def test_ignores_rally_slower_than_horizon():
    # +60% but spread over 200 trading days: no 63-day window gains 50%
    prices = flat(20) + list(np.linspace(100, 160, 200)) + flat(20, 160)
    eps = find_winners("TEST", make_series(prices), threshold=0.50, horizon=63)
    assert eps == []


def test_two_separate_rallies_become_two_episodes():
    rally = list(np.linspace(100, 160, 20))
    crash = list(np.linspace(160, 100, 20))
    prices = flat(80) + rally + crash + flat(150) + rally + flat(80, 160)
    eps = find_winners("TEST", make_series(prices), threshold=0.50, horizon=63)
    assert len(eps) == 2
    assert eps[0].entry_date < eps[1].entry_date


def test_one_rally_is_one_episode_despite_many_signal_days():
    prices = flat(100) + list(np.linspace(100, 160, 30)) + flat(100, 160)
    eps = find_winners("TEST", make_series(prices), threshold=0.50, horizon=63)
    assert len(eps) == 1
    assert eps[0].signal_days > 1  # many overlapping days merged into one event


def test_last_horizon_days_are_never_labeled():
    # the final `horizon` rows have incomplete forward windows: they must be
    # NaN, and no episode may start there (no peeking at partial futures)
    prices = flat(100) + list(np.linspace(100, 160, 30)) + flat(10, 160)
    s = make_series(prices)
    fwd = forward_returns(s, horizon=63)
    assert fwd["max_fwd_return"].iloc[-63:].isna().all()
    assert fwd["max_fwd_return"].iloc[: -63].notna().all()
    for e in merge_episodes("TEST", fwd, threshold=0.50, horizon=63):
        assert e.entry_date <= s.index[len(s) - 64]


def test_forward_window_excludes_entry_day():
    # spike on day 0 itself must not count as its own forward gain
    prices = [160.0] + flat(70)  # 100s after a one-day 160
    fwd = forward_returns(make_series(prices), horizon=63)
    assert fwd["max_fwd_return"].iloc[0] == pytest.approx(100 / 160 - 1)


def test_short_series_returns_empty():
    assert find_winners("TEST", make_series(flat(30)), horizon=63) == []
