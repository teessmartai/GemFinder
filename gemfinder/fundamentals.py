"""Point-in-time fundamentals from SEC EDGAR's free XBRL API.

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
Every reported fact carries the date the filing was *filed* with the SEC.
The point-in-time rule here is strict: a fact is visible at a cutoff date
only if `filed` <= cutoff. A 10-K covering December but filed in February
is correctly invisible during January.

Coverage caveats: XBRL is mandatory only since ~2009, US filers only, and
companies tag the same concept under different names — we try a list of
synonyms per concept. Episodes before ~2009 will simply get NaNs.

SEC fair-access policy: identify yourself via User-Agent and stay under
10 requests/second (we sleep between calls and cache every response).
"""

import json
import time
from pathlib import Path

SEC_CACHE = Path("data/sec")
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# concept -> us-gaap tag synonyms, most preferred first
CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}


def _fetch_json(url: str, cache_name: str, user_agent: str) -> dict | None:
    SEC_CACHE.mkdir(parents=True, exist_ok=True)
    path = SEC_CACHE / cache_name
    if path.exists():
        return json.loads(path.read_text())

    import requests

    time.sleep(0.15)  # stay well under SEC's 10 req/s limit
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    if resp.status_code == 404:
        path.write_text("null")  # cache the miss
        return None
    resp.raise_for_status()
    path.write_text(resp.text)
    return resp.json()


def ticker_to_cik(user_agent: str) -> dict[str, int]:
    """Map of TICKER -> CIK from the SEC's official list."""
    data = _fetch_json(TICKER_MAP_URL, "company_tickers.json", user_agent)
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def get_companyfacts(cik: int, user_agent: str) -> dict | None:
    return _fetch_json(
        COMPANYFACTS_URL.format(cik=cik), f"companyfacts_{cik:010d}.json", user_agent
    )


def _visible_items(facts: dict, taxonomy: str, tag: str, cutoff: str) -> list[dict]:
    """All facts for a tag that were FILED on or before the cutoff date."""
    units = facts.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {})
    out = []
    for items in units.values():
        for it in items:
            if it.get("filed") and it.get("end") and it.get("val") is not None:
                if it["filed"] <= cutoff:  # ISO dates compare as strings
                    out.append(it)
    return out


def point_in_time_fact(facts: dict, taxonomy: str, tag: str, cutoff: str) -> dict | None:
    """Most recent fact (latest period end, then latest filed) visible at cutoff."""
    items = _visible_items(facts, taxonomy, tag, cutoff)
    if not items:
        return None
    return max(items, key=lambda it: (it["end"], it["filed"]))


def _year_ago_value(items: list[dict], current: dict) -> float | None:
    """Same-concept fact for the comparable period ~1 year before `current`."""
    import pandas as pd

    cur_end = pd.Timestamp(current["end"])
    cur_dur = None
    if current.get("start"):
        cur_dur = (cur_end - pd.Timestamp(current["start"])).days
    best = None
    for it in items:
        days_back = (cur_end - pd.Timestamp(it["end"])).days
        if not 340 <= days_back <= 390:
            continue
        if cur_dur is not None and it.get("start"):
            dur = (pd.Timestamp(it["end"]) - pd.Timestamp(it["start"])).days
            if abs(dur - cur_dur) > 21:  # quarter vs full-year mismatch
                continue
        if best is None or (it["end"], it["filed"]) > (best["end"], best["filed"]):
            best = it
    return float(best["val"]) if best else None


def fundamental_features(facts: dict | None, cutoff: str) -> dict:
    """Point-in-time fundamental dict as of `cutoff` (ISO date string).

    For each concept, returns the value plus the period-end and filed dates
    so results can be audited for look-ahead.
    """
    out: dict = {}
    if facts is None:
        facts = {}

    for name, tags in CONCEPTS.items():
        best, best_items = None, []
        for tag in tags:
            fact = point_in_time_fact(facts, "us-gaap", tag, cutoff)
            if fact and (best is None or (fact["end"], fact["filed"]) > (best["end"], best["filed"])):
                best = fact
                best_items = _visible_items(facts, "us-gaap", tag, cutoff)
        if best:
            out[name] = float(best["val"])
            out[f"{name}_period_end"] = best["end"]
            out[f"{name}_filed"] = best["filed"]
            if name in ("revenue", "net_income"):
                prior = _year_ago_value(best_items, best)
                if prior not in (None, 0):
                    out[f"{name}_yoy"] = float(best["val"]) / prior - 1.0
        else:
            out[name] = None

    shares = point_in_time_fact(facts, "dei", "EntityCommonStockSharesOutstanding", cutoff)
    out["shares_outstanding"] = float(shares["val"]) if shares else None
    return out
