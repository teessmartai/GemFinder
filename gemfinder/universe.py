"""Ticker universes to scan.

IMPORTANT — survivorship bias: any "current constituents" list excludes
companies that were delisted, acquired, or went bankrupt during the last
20 years. A scan over today's S&P 500 will overstate how often stocks
rally 50% because the failures are missing from the sample. Use it to get
started, but for real research feed --universe a point-in-time list
(e.g. historical index constituents from a paid provider, or the full
Nasdaq/NYSE symbol files including delistings).
"""

import pandas as pd

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituents scraped from Wikipedia (needs internet)."""
    tables = pd.read_html(SP500_WIKI_URL)
    symbols = tables[0]["Symbol"].astype(str).str.strip()
    # Yahoo uses '-' where the listing uses '.' (e.g. BRK.B -> BRK-B)
    return sorted(symbols.str.replace(".", "-", regex=False))


def load_universe(path: str) -> list[str]:
    """Read tickers from a text file: one per line, '#' comments allowed."""
    tickers = []
    with open(path) as f:
        for line in f:
            t = line.split("#", 1)[0].strip().upper()
            if t:
                tickers.append(t)
    return sorted(set(tickers))
