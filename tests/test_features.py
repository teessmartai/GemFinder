"""Price-feature tests; the look-ahead test is the load-bearing one."""

import math

import numpy as np
import pandas as pd
import pytest

from gemfinder.features import price_features


def make_ohlcv(closes, volumes=None):
    idx = pd.bdate_range("2010-01-04", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    v = pd.Series(volumes if volumes is not None else [1e6] * len(c), index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": v})


def assert_same_features(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and math.isnan(a[k]):
            assert math.isnan(b[k]), k
        else:
            assert a[k] == pytest.approx(b[k]), k


def test_no_lookahead():
    # features at the cutoff must be identical whether the future is calm
    # or a 10x moonshot with a volume explosion
    rng = np.random.default_rng(0)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 400)))
    df = make_ohlcv(closes)
    cutoff = df.index[299]

    before = price_features(df, cutoff)

    tampered = df.copy()
    tampered.loc[tampered.index[300]:, "close"] *= 10
    tampered.loc[tampered.index[300]:, "volume"] *= 100
    after = price_features(tampered, cutoff)

    assert_same_features(before, after)


def test_momentum_values():
    closes = [100.0] * 300 + [150.0]  # one final jump
    feats = price_features(make_ohlcv(closes), make_ohlcv(closes).index[-1])
    assert feats["ret_21d"] == pytest.approx(0.50)
    assert feats["ret_252d"] == pytest.approx(0.50)
    assert feats["dist_52w_high"] == pytest.approx(0.0)  # at its 52w high
    assert feats["dist_52w_low"] == pytest.approx(0.50)
    assert feats["close"] == 150.0


def test_insufficient_history_gives_nan():
    closes = [100.0] * 60  # under a quarter of data
    feats = price_features(make_ohlcv(closes), make_ohlcv(closes).index[-1])
    assert math.isnan(feats["ret_252d"])
    assert math.isnan(feats["dist_52w_high"])
    assert math.isnan(feats["sma200_ratio"])
    assert feats["ret_21d"] == pytest.approx(0.0)  # enough history for this one
    assert feats["history_days"] == 60


def test_volume_ratio_detects_surge():
    volumes = [1e6] * 280 + [3e6] * 21  # last month triple the prior norm
    closes = [100.0] * 301
    df = make_ohlcv(closes, volumes)
    feats = price_features(df, df.index[-1])
    assert feats["volume_ratio_21v63"] == pytest.approx(3.0)


def test_drawdown():
    closes = [100.0] * 100 + [50.0] * 200  # halved during the trailing year
    df = make_ohlcv(closes)
    feats = price_features(df, df.index[-1])
    assert feats["max_drawdown_252d"] == pytest.approx(-0.50)
