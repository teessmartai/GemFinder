"""Point-in-time price/volume features.

Every feature is computed from OHLCV rows dated on or before the cutoff
date — the entry_date from winners.csv. Data after the cutoff must never
influence a feature (enforced by tests/test_features.py).

Features needing more history than exists come back NaN rather than being
computed on a short window under a misleading name.
"""

import numpy as np
import pandas as pd

YEAR = 252  # trading days


def _trailing_return(close: pd.Series, k: int) -> float:
    if len(close) <= k:
        return np.nan
    return close.iloc[-1] / close.iloc[-1 - k] - 1.0


def _annualized_vol(returns: pd.Series, k: int) -> float:
    if len(returns) < k:
        return np.nan
    return float(returns.iloc[-k:].std() * np.sqrt(YEAR))


def _rsi(close: pd.Series, k: int = 14) -> float:
    if len(close) <= k:
        return np.nan
    delta = close.diff().iloc[-k:]
    gain = delta.clip(lower=0).mean()
    loss = (-delta.clip(upper=0)).mean()
    if gain + loss == 0:
        return 50.0
    return float(100.0 * gain / (gain + loss))


def price_features(ohlcv: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Feature dict as of `cutoff` (inclusive — its close is known at entry)."""
    df = ohlcv.loc[:cutoff]
    close, volume = df["close"], df["volume"]
    daily = close.pct_change().dropna()
    year = close.iloc[-YEAR:]

    feats = {
        # momentum
        "ret_21d": _trailing_return(close, 21),
        "ret_63d": _trailing_return(close, 63),
        "ret_126d": _trailing_return(close, 126),
        "ret_252d": _trailing_return(close, 252),
        # risk
        "vol_21d": _annualized_vol(daily, 21),
        "vol_63d": _annualized_vol(daily, 63),
        "rsi_14": _rsi(close),
    }

    # position within the trailing year
    if len(close) >= YEAR:
        feats["dist_52w_high"] = float(close.iloc[-1] / year.max() - 1.0)
        feats["dist_52w_low"] = float(close.iloc[-1] / year.min() - 1.0)
        feats["max_drawdown_252d"] = float((year / year.cummax() - 1.0).min())
    else:
        feats["dist_52w_high"] = feats["dist_52w_low"] = np.nan
        feats["max_drawdown_252d"] = np.nan

    # trend
    feats["sma50_ratio"] = (
        float(close.iloc[-1] / close.iloc[-50:].mean() - 1.0) if len(close) >= 50 else np.nan
    )
    feats["sma200_ratio"] = (
        float(close.iloc[-1] / close.iloc[-200:].mean() - 1.0) if len(close) >= 200 else np.nan
    )

    # volume: recent month vs the prior quarter
    if len(volume) >= 84 and volume.iloc[-84:-21].mean() > 0:
        feats["volume_ratio_21v63"] = float(volume.iloc[-21:].mean() / volume.iloc[-84:-21].mean())
    else:
        feats["volume_ratio_21v63"] = np.nan

    # liquidity / size proxy
    feats["dollar_vol_21d"] = (
        float((close * volume).iloc[-21:].median()) if len(close) >= 21 else np.nan
    )

    feats["close"] = float(close.iloc[-1])
    feats["history_days"] = len(close)
    return feats
