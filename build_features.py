#!/usr/bin/env python3
"""Step 2: point-in-time features for winners and matched controls.

Reads winners.csv from step 1 (find_winners.py), samples K non-winner
control (ticker, date) pairs per winner on the same calendar dates, and
computes for every row only data that existed on or before its date:

  - price/volume features from Yahoo Finance daily history
  - (optional) fundamentals from SEC EDGAR's XBRL API, selected by the
    date each filing was FILED — never by period covered

Usage:
    python build_features.py --sec-email you@example.com
    python build_features.py --universe tickers.txt --controls-per-winner 5
    python build_features.py --no-fundamentals          # prices only

Output: features.csv with label=1 (winner) / 0 (control) plus the forward
outcome columns, ready for step-3 analysis.
"""

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from gemfinder.controls import sample_controls
from gemfinder.data import get_close_series, get_ohlcv
from gemfinder.features import price_features
from gemfinder.fundamentals import fundamental_features, get_companyfacts, ticker_to_cik
from gemfinder.labeling import TRADING_DAYS_90
from gemfinder.universe import load_universe, sp500_tickers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--winners", default="winners.csv", help="step-1 output (default winners.csv)")
    ap.add_argument("--universe", help="ticker file for the control pool "
                    "(default: current S&P 500 from Wikipedia; same caveats as step 1)")
    ap.add_argument("--years", type=int, default=20,
                    help="price history lookback, should match step 1 (default 20)")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="winner threshold used in step 1 (default 0.50)")
    ap.add_argument("--horizon", type=int, default=TRADING_DAYS_90,
                    help="forward window in trading days used in step 1 (default 63)")
    ap.add_argument("--controls-per-winner", type=int, default=3)
    ap.add_argument("--no-fundamentals", action="store_true",
                    help="skip SEC EDGAR fundamentals (price features only)")
    ap.add_argument("--sec-email", help="your email for the SEC API User-Agent "
                    "(required unless --no-fundamentals)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="features.csv")
    args = ap.parse_args()

    if not args.no_fundamentals and not args.sec_email:
        ap.error("--sec-email is required for SEC EDGAR access "
                 "(or pass --no-fundamentals)")

    end = date.today()
    start = end - timedelta(days=args.years * 365)
    winners = pd.read_csv(args.winners, parse_dates=["entry_date"])
    print(f"{len(winners)} winner episodes from {args.winners}")

    universe = load_universe(args.universe) if args.universe else sp500_tickers()
    print(f"Control pool: {len(universe)} tickers")

    rows = []

    # winners: label=1, outcomes carried over from step 1
    for i, w in winners.iterrows():
        ohlcv = get_ohlcv(w["ticker"], str(start), str(end))
        if ohlcv is None:
            print(f"  {w['ticker']}: no price data, skipped")
            continue
        feats = price_features(ohlcv, w["entry_date"])
        rows.append({
            "ticker": w["ticker"], "date": w["entry_date"].date(), "label": 1,
            "max_fwd_return": w["max_return_pct"] / 100.0,
            "horizon_return": w["horizon_return_pct"] / 100.0,
            **feats,
        })

    # controls: label=0, same dates, did NOT rally
    print(f"Sampling {args.controls_per_winner} controls per winner...")
    controls = sample_controls(
        winners, universe,
        load_close=lambda t: get_close_series(t, str(start), str(end)),
        threshold=args.threshold, horizon=args.horizon,
        per_winner=args.controls_per_winner, seed=args.seed,
    )
    for _, c in controls.iterrows():
        ohlcv = get_ohlcv(c["ticker"], str(start), str(end))
        feats = price_features(ohlcv, c["date"])
        rows.append({
            "ticker": c["ticker"], "date": c["date"].date(), "label": 0,
            "max_fwd_return": c["max_fwd_return"],
            "horizon_return": c["horizon_return"],
            **feats,
        })
    print(f"{len(controls)} controls sampled "
          f"(target {len(winners) * args.controls_per_winner})")

    df = pd.DataFrame(rows)

    if not args.no_fundamentals:
        ua = f"GemFinder research {args.sec_email}"
        print("Fetching SEC EDGAR fundamentals (cached in data/sec/)...")
        cik_map = ticker_to_cik(ua)
        facts_cache: dict[str, dict | None] = {}
        fund_rows = []
        for _, r in df.iterrows():
            ticker = r["ticker"]
            if ticker not in facts_cache:
                cik = cik_map.get(ticker.replace("-", ""))  # SEC map has no '-'
                facts_cache[ticker] = get_companyfacts(cik, ua) if cik else None
            fund = fundamental_features(facts_cache[ticker], str(r["date"]))
            if fund.get("shares_outstanding") and pd.notna(r["close"]):
                fund["market_cap"] = fund["shares_outstanding"] * r["close"]
            fund_rows.append(fund)
        df = pd.concat([df, pd.DataFrame(fund_rows, index=df.index)], axis=1)
        n_fund = df["revenue"].notna().sum()
        print(f"Fundamentals found for {n_fund}/{len(df)} rows "
              f"(pre-2009 dates and non-US filers come back empty)")

    df = df.sort_values(["date", "ticker"])
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows ({int(df['label'].sum())} winners, "
          f"{int((df['label'] == 0).sum())} controls) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
