# GemFinder

Research pipeline for finding signals that precede large stock rallies.

**The plan:**
1. **Label the winners (step 1 — done).** Scan ~20 years of daily prices and
   find every episode where a stock gained ≥50% within the next 90 days.
   Output: `winners.csv` with one row per rally — ticker, `entry_date`,
   entry price, peak, return.
2. **Collect point-in-time features (step 2 — done).** For each
   `(ticker, entry_date)` — and for matched control stocks that did *not*
   rally — gather only data that existed *on or before* that date.
   Output: `features.csv`.
3. **Find what winners have in common — versus the controls (step 3 — next).**
   Anything "common to winners" is meaningless unless it is *rarer among
   stocks that didn't rally*; that's what the control rows are for.

## Quick start

```bash
pip install -r requirements.txt

# Step 1: current S&P 500, last 20 years, +50% within ~90 calendar days
python find_winners.py

# Step 2: features for winners + 3 same-date controls each
python build_features.py --sec-email you@example.com
# or skip SEC fundamentals (price/volume features only, works pre-2009 too):
python build_features.py --no-fundamentals
```

`--horizon` is in trading days (63 ≈ 90 calendar days). Both steps accept
`--universe my_tickers.txt`, and everything downloaded is cached under
`data/` so re-runs are fast.

## Data sources

| Data | Source | Cost | Point-in-time safety |
|------|--------|------|----------------------|
| Daily OHLCV prices | Yahoo Finance (`yfinance`) | free | features computed strictly from rows ≤ entry date |
| Fundamentals | SEC EDGAR XBRL API (`data.sec.gov`) | free | facts filtered by the date the filing was **filed**, not the period covered |
| Ticker→CIK map | SEC `company_tickers.json` | free | — |
| Universe | Wikipedia S&P 500 (or your own file) | free | **not** point-in-time — see limitations |

The filed-date rule is what makes EDGAR usable here: a 10-K covering
December but filed in February is invisible to a January entry date.
Limits: XBRL is only mandatory since ~2009 and covers US filers only, so
older episodes get price features but empty fundamentals. The SEC asks for
an identifying User-Agent — hence `--sec-email` — and ≤10 requests/second
(the client sleeps between calls and caches every response in `data/sec/`).

## What's in features.csv

One row per (ticker, date) with `label` 1 = winner episode, 0 = matched
control, plus the forward outcome (`max_fwd_return`, `horizon_return`) and:

- **Momentum:** `ret_21d/63d/126d/252d`
- **Risk:** `vol_21d`, `vol_63d` (annualized), `rsi_14`, `max_drawdown_252d`
- **Position:** `dist_52w_high`, `dist_52w_low`, `sma50_ratio`, `sma200_ratio`
- **Volume/size:** `volume_ratio_21v63` (recent month vs prior quarter),
  `dollar_vol_21d`
- **Fundamentals** (when available): `revenue`, `net_income`, `assets`,
  `equity`, `cash`, `eps_diluted`, `revenue_yoy`, `net_income_yoy`,
  `shares_outstanding`, `market_cap` — each with `*_period_end` and
  `*_filed` audit columns so you can verify nothing post-dates the entry.

Controls are sampled on the **same calendar date** as the winner they match
(default 3 per winner, `--controls-per-winner`), so March-2009 winners are
compared against March-2009 non-winners — market regime can't masquerade as
a stock-picking signal. A control must have a complete forward window whose
max return stays below the threshold.

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
