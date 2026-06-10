#!/usr/bin/env python3
"""Scan historical prices for stocks that gained >= 50% within 90 days.

This is step 1 of the GemFinder research plan: build the labeled set of
past winners (ticker + entry date). Step 2 gathers point-in-time data as
of each entry_date — never anything later — to look for common signals.

Usage:
    python find_winners.py                          # current S&P 500, last 20y
    python find_winners.py --universe tickers.txt   # your own ticker list
    python find_winners.py --threshold 1.0 --horizon 126 --years 10

Output: winners.csv, one row per rally episode:
    ticker, entry_date, entry_price, peak_date, peak_price,
    max_return_pct, days_to_peak, horizon_return_pct, signal_days
entry_date is the cutoff for step-2 feature collection.
"""

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from gemfinder.data import get_close_series
from gemfinder.labeling import TRADING_DAYS_90, find_winners
from gemfinder.universe import load_universe, sp500_tickers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--universe", help="text file of tickers, one per line "
                    "(default: scrape current S&P 500 from Wikipedia)")
    ap.add_argument("--years", type=int, default=20, help="lookback years (default 20)")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="minimum gain within the window, 0.50 = +50%% (default)")
    ap.add_argument("--horizon", type=int, default=TRADING_DAYS_90,
                    help="forward window in trading days, 63 ~= 90 calendar days (default)")
    ap.add_argument("--out", default="winners.csv", help="output CSV (default winners.csv)")
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=args.years * 365)

    if args.universe:
        tickers = load_universe(args.universe)
    else:
        print("No --universe given; using current S&P 500 from Wikipedia.")
        print("NOTE: this list has survivorship bias — see README.")
        tickers = sp500_tickers()
    print(f"Scanning {len(tickers)} tickers, {start} .. {end}, "
          f"threshold +{args.threshold:.0%} within {args.horizon} trading days\n")

    episodes = []
    missing = []
    for i, ticker in enumerate(tickers, 1):
        try:
            close = get_close_series(ticker, str(start), str(end))
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {ticker}: download failed ({e})")
            missing.append(ticker)
            continue
        if close is None or len(close) <= args.horizon:
            missing.append(ticker)
            continue
        eps = find_winners(ticker, close, threshold=args.threshold, horizon=args.horizon)
        episodes.extend(eps)
        if i % 25 == 0 or eps:
            print(f"[{i}/{len(tickers)}] {ticker}: {len(eps)} episode(s), "
                  f"{len(episodes)} total")

    if not episodes:
        print("\nNo qualifying episodes found.")
        return 1

    df = pd.DataFrame(
        {
            "ticker": e.ticker,
            "entry_date": e.entry_date.date(),
            "entry_price": round(e.entry_price, 4),
            "peak_date": e.peak_date.date(),
            "peak_price": round(e.peak_price, 4),
            "max_return_pct": round(e.max_return * 100, 2),
            "days_to_peak": e.days_to_peak,
            "horizon_return_pct": round(e.horizon_return * 100, 2),
            "signal_days": e.signal_days,
        }
        for e in episodes
    ).sort_values(["entry_date", "ticker"])
    df.to_csv(args.out, index=False)

    by_year = df["entry_date"].map(lambda d: d.year).value_counts().sort_index()
    print(f"\n{len(df)} episodes across {df['ticker'].nunique()} tickers "
          f"-> {args.out}")
    print(f"{len(missing)} tickers had no usable data: "
          f"{', '.join(missing[:15])}{'...' if len(missing) > 15 else ''}")
    print("\nEpisodes per entry year (watch for crisis-rebound clusters):")
    print(by_year.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
