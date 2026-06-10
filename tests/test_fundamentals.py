"""Tests for the SEC point-in-time rule: visibility is by FILED date."""

import pytest

from gemfinder.fundamentals import fundamental_features, point_in_time_fact


def fact(val, start, end, filed):
    return {"val": val, "start": start, "end": end, "filed": filed}


def facts_json(tag_items: dict, taxonomy: str = "us-gaap", unit: str = "USD"):
    return {
        "facts": {
            taxonomy: {
                tag: {"units": {unit: items}} for tag, items in tag_items.items()
            }
        }
    }


Q3_2019 = fact(900, "2019-07-01", "2019-09-30", "2019-11-05")
FY_2019 = fact(4000, "2019-01-01", "2019-12-31", "2020-02-20")
Q3_2018 = fact(750, "2018-07-01", "2018-09-30", "2018-11-06")
FY_2018 = fact(3200, "2018-01-01", "2018-12-31", "2019-02-21")


def test_filing_not_yet_filed_is_invisible():
    # In January 2020 the FY2019 10-K (filed Feb 20) must NOT be visible,
    # even though its period already ended — this is the look-ahead trap.
    facts = facts_json({"Revenues": [Q3_2019, FY_2019]})
    best = point_in_time_fact(facts, "us-gaap", "Revenues", "2020-01-15")
    assert best["val"] == 900  # still the Q3 10-Q

    # one day after filing, the 10-K becomes visible
    best = point_in_time_fact(facts, "us-gaap", "Revenues", "2020-02-21")
    assert best["val"] == 4000


def test_latest_period_end_wins_among_visible():
    facts = facts_json({"Revenues": [Q3_2018, FY_2018, Q3_2019]})
    best = point_in_time_fact(facts, "us-gaap", "Revenues", "2019-12-01")
    assert best["end"] == "2019-09-30"


def test_no_visible_facts_returns_none():
    facts = facts_json({"Revenues": [Q3_2019]})
    assert point_in_time_fact(facts, "us-gaap", "Revenues", "2019-01-01") is None


def test_yoy_matches_comparable_period_not_annual():
    # current = Q3 2019 (3-month duration); YoY must compare against
    # Q3 2018, never against the FY2018 annual figure
    facts = facts_json({"Revenues": [Q3_2018, FY_2018, Q3_2019]})
    out = fundamental_features(facts, "2019-12-01")
    assert out["revenue"] == 900
    assert out["revenue_yoy"] == pytest.approx(900 / 750 - 1)


def test_tag_synonyms_are_tried():
    # company tags revenue under the ASC 606 name instead of "Revenues"
    facts = facts_json({"RevenueFromContractWithCustomerExcludingAssessedTax": [Q3_2019]})
    out = fundamental_features(facts, "2019-12-01")
    assert out["revenue"] == 900


def test_audit_columns_present():
    facts = facts_json({"Revenues": [Q3_2019]})
    out = fundamental_features(facts, "2019-12-01")
    assert out["revenue_period_end"] == "2019-09-30"
    assert out["revenue_filed"] == "2019-11-05"


def test_missing_company_gives_nones():
    out = fundamental_features(None, "2019-12-01")
    assert out["revenue"] is None
    assert out["shares_outstanding"] is None


def test_shares_outstanding_from_dei():
    facts = {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [fact(5_000_000, None, "2019-09-30", "2019-11-05")]}
                }
            }
        }
    }
    out = fundamental_features(facts, "2019-12-01")
    assert out["shares_outstanding"] == 5_000_000
    # and invisible before it was filed
    out = fundamental_features(facts, "2019-10-01")
    assert out["shares_outstanding"] is None
