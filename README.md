# GemFinder

Research pipeline for finding signals that precede large stock rallies.

**The plan:**
1. **Label the winners (this repo, step 1 — done).** Scan ~20 years of daily
   prices and find every episode where a stock gained ≥50% within the next
   90 days. Output: `winners.csv` with one row per rally — ticker,
   `entry_date`, entry price, peak, return.
2. **Collect point-in-time features (step 2 — next).** For each
   `(ticker, entry_date)`, gather only data that existed *on or before*
   `entry_date`: fundamentals, price/volume history, etc.
3. **Find what winners have in common — versus a control group.** Compare the
   winners' features against matched non-winners. Anything "common to
   winners" is meaningless unless it is *rarer among stocks that didn't
   rally*.

## Quick start

```bash
pip install -r requirements.txt

# Current S&P 500, last 20 years, +50% within ~90 calendar days:
python find_winners.py

# Your own ticker list, custom parameters:
python find_winners.py --universe my_tickers.txt --threshold 0.50 --horizon 63 --years 20
```

`--horizon` is in trading days (63 ≈ 90 calendar days). Downloads are cached
in `data/cache/`, so re-runs are fast. Output goes to `winners.csv`.

## How the labeling works

For each day *t* of each stock we look at adjusted closes over the next 63
trading days (*t+1 … t+63*, entry day excluded) and mark *t* a "winner day"
if the window's maximum is ≥50% above the close at *t*. Because windows
overlap, one rally produces a long run of winner days; these are merged into
a single **episode** whose `entry_date` is the *first* day the move was still
fully ahead of you. Days near the end of the data with incomplete forward
windows are never labeled. The labeler is unit-tested against brute-force
references (`tests/test_labeling.py`).

`entry_date` is the cutoff for step 2: features for an episode may use
nothing dated after it.

## Honest limitations to keep in mind

- **Survivorship bias (the big one).** The default universe is the *current*
  S&P 500 — companies that survived. Yahoo Finance also lacks most delisted
  tickers. Stocks that rallied 50% and later went to zero are
  underrepresented, which inflates how attractive the signals will look. For
  serious results, use point-in-time constituent lists or a survivorship-free
  dataset (Sharadar, Norgate, CRSP).
- **Base rates depend on the universe.** +50%/90d is rare in large caps and
  routine in micro-caps and biotech. If the universe mixes both, "what
  winners have in common" will mostly rediscover "being a volatile small
  cap." Compare against controls drawn from the same universe.
- **Events cluster in regimes.** Expect spikes of episodes in 2009 and 2020
  (the per-year count is printed for this reason). A signal that only says
  "it was March 2009" won't help you next year — consider regime-aware
  controls or excluding/downweighting those windows.
- **Multiple-hypothesis risk in step 3.** Test enough features on a few
  thousand episodes and some will look predictive by chance. Hold out the
  most recent years entirely and validate any candidate signal there once.
- **Prices are split/dividend-adjusted** (`auto_adjust=True`), so the
  returns are total-price moves, not artifacts of splits.
