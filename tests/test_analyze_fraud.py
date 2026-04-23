import pandas as pd
import pytest
from analyze_fraud import score_transactions, summarize_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transactions(rows):
    """Build a transactions DataFrame from a list of dicts."""
    defaults = {
        "merchant_category": "grocery",
        "channel": "web",
        "ip_country": "US",
        "chargeback_within_60d": 0,
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


def _make_accounts(rows):
    defaults = {
        "customer_name": "Test User",
        "country": "US",
        "signup_date": "2022-01-01",
        "kyc_level": "full",
        "account_age_days": 365,
        "is_vip": "N",
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


def _make_chargebacks(*tx_ids):
    return pd.DataFrame({"transaction_id": list(tx_ids)})


# Realistic low-risk transaction (scores low)
_LOW_TX = {
    "transaction_id": 1, "account_id": 101,
    "amount_usd": 45.0, "device_risk_score": 5,
    "is_international": 0, "velocity_24h": 1, "failed_logins_24h": 0,
}

# Realistic high-risk transaction (scores high after fixes)
_HIGH_TX = {
    "transaction_id": 2, "account_id": 102,
    "amount_usd": 1400.0, "device_risk_score": 85,
    "is_international": 1, "velocity_24h": 8, "failed_logins_24h": 7,
}

_LOW_ACCT = {"account_id": 101, "prior_chargebacks": 0}
_HIGH_ACCT = {"account_id": 102, "prior_chargebacks": 1}


# ---------------------------------------------------------------------------
# score_transactions — output structure
# ---------------------------------------------------------------------------

def test_score_transactions_adds_risk_score_column():
    txns = _make_transactions([_LOW_TX])
    accts = _make_accounts([_LOW_ACCT])
    result = score_transactions(txns, accts)
    assert "risk_score" in result.columns


def test_score_transactions_adds_risk_label_column():
    txns = _make_transactions([_LOW_TX])
    accts = _make_accounts([_LOW_ACCT])
    result = score_transactions(txns, accts)
    assert "risk_label" in result.columns


def test_score_transactions_preserves_all_rows():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    result = score_transactions(txns, accts)
    assert len(result) == 2


def test_score_transactions_risk_score_in_valid_range():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    result = score_transactions(txns, accts)
    assert result["risk_score"].between(0, 100).all()


def test_score_transactions_risk_label_values_valid():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    result = score_transactions(txns, accts)
    assert set(result["risk_label"]).issubset({"low", "medium", "high"})


def test_score_transactions_label_consistent_with_score():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    result = score_transactions(txns, accts)
    for _, row in result.iterrows():
        score = row["risk_score"]
        label = row["risk_label"]
        if score >= 60:
            assert label == "high", f"score {score} should be 'high', got '{label}'"
        elif score >= 30:
            assert label == "medium", f"score {score} should be 'medium', got '{label}'"
        else:
            assert label == "low", f"score {score} should be 'low', got '{label}'"


# ---------------------------------------------------------------------------
# score_transactions — scoring correctness
# ---------------------------------------------------------------------------

def test_clean_transaction_scores_low():
    txns = _make_transactions([_LOW_TX])
    accts = _make_accounts([_LOW_ACCT])
    result = score_transactions(txns, accts)
    assert result["risk_label"].iloc[0] == "low"


def test_high_risk_transaction_scores_high():
    txns = _make_transactions([_HIGH_TX])
    accts = _make_accounts([_HIGH_ACCT])
    result = score_transactions(txns, accts)
    assert result["risk_label"].iloc[0] == "high"


def test_high_risk_scores_higher_than_low_risk():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    result = score_transactions(txns, accts).set_index("transaction_id")
    assert result.loc[2, "risk_score"] > result.loc[1, "risk_score"]


def test_confirmed_chargebacks_score_non_low():
    """
    All 8 confirmed chargebacks from the dataset must score medium or high.
    TX 50008 (device score 68, just below the 70 threshold) legitimately scores
    medium (50); the remaining 7 score high. None should score low.
    """
    fraud_txns = [
        # tx_id, account_id, amount, device, intl, vel, logins
        (50003, 1003, 1250.0, 81, 1, 6, 5),
        (50006, 1006, 399.99, 77, 1, 7, 6),
        (50008, 1008, 620.0, 68, 1, 5, 3),   # device 68 → medium (score 50)
        (50011, 1011, 1400.0, 85, 1, 8, 7),
        (50013, 1003, 150.0, 79, 1, 7, 5),
        (50014, 1006, 49.99, 72, 1, 9, 7),
        (50015, 1008, 910.0, 71, 1, 6, 4),
        (50019, 1011, 75.0, 83, 1, 10, 8),
    ]
    rows = [
        {"transaction_id": tid, "account_id": aid, "amount_usd": amt,
         "device_risk_score": dev, "is_international": intl,
         "velocity_24h": vel, "failed_logins_24h": logins}
        for tid, aid, amt, dev, intl, vel, logins in fraud_txns
    ]
    acct_rows = [
        {"account_id": 1003, "prior_chargebacks": 0},
        {"account_id": 1006, "prior_chargebacks": 3},
        {"account_id": 1008, "prior_chargebacks": 0},
        {"account_id": 1011, "prior_chargebacks": 1},
    ]
    txns = _make_transactions(rows)
    accts = _make_accounts(acct_rows)
    result = score_transactions(txns, accts)

    scored_low = result[result["risk_label"] == "low"][["transaction_id", "risk_score", "risk_label"]]
    assert scored_low.empty, f"No chargeback tx should score 'low':\n{scored_low}"

    scored_high = result[result["risk_label"] == "high"]
    assert len(scored_high) == 7, f"Expected 7 of 8 to score 'high', got {len(scored_high)}"

    tx_50008 = result[result["transaction_id"] == 50008].iloc[0]
    assert tx_50008["risk_label"] == "medium"
    assert tx_50008["risk_score"] == 50


# ---------------------------------------------------------------------------
# summarize_results — output structure
# ---------------------------------------------------------------------------

def test_summarize_results_returns_dataframe():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks())
    assert isinstance(result, pd.DataFrame)


def test_summarize_results_has_required_columns():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks())
    for col in ("risk_label", "transactions", "total_amount_usd", "avg_amount_usd",
                "chargebacks", "chargeback_rate"):
        assert col in result.columns, f"Missing column: {col}"


def test_summarize_results_one_row_per_risk_label():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks())
    assert result["risk_label"].nunique() == len(result)


# ---------------------------------------------------------------------------
# summarize_results — chargeback counting
# ---------------------------------------------------------------------------

def test_summarize_results_no_chargebacks_rate_is_zero():
    txns = _make_transactions([_LOW_TX])
    accts = _make_accounts([_LOW_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks())
    assert (result["chargebacks"] == 0).all()
    assert (result["chargeback_rate"] == 0).all()


def test_summarize_results_all_chargebacks_rate_is_one():
    txns = _make_transactions([_HIGH_TX])
    accts = _make_accounts([_HIGH_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks(2))
    assert result["chargeback_rate"].iloc[0] == 1.0


def test_summarize_results_chargeback_rate_partial():
    two_high = [
        {"transaction_id": 10, "account_id": 102, "amount_usd": 1400.0,
         "device_risk_score": 85, "is_international": 1, "velocity_24h": 8,
         "failed_logins_24h": 7},
        {"transaction_id": 11, "account_id": 102, "amount_usd": 1200.0,
         "device_risk_score": 80, "is_international": 1, "velocity_24h": 7,
         "failed_logins_24h": 6},
    ]
    txns = _make_transactions(two_high)
    accts = _make_accounts([{"account_id": 102, "prior_chargebacks": 1}])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks(10))  # only tx 10 charged back
    high_row = result[result["risk_label"] == "high"].iloc[0]
    assert high_row["chargebacks"] == 1
    assert high_row["chargeback_rate"] == pytest.approx(0.5)


def test_summarize_results_transaction_counts_match():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks())
    assert result["transactions"].sum() == len(txns)


def test_summarize_results_total_amount_matches():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks())
    expected_total = txns["amount_usd"].sum()
    assert result["total_amount_usd"].sum() == pytest.approx(expected_total)


def test_summarize_results_chargeback_not_in_scored_is_ignored():
    """A chargeback transaction_id that doesn't appear in scored should not inflate counts."""
    txns = _make_transactions([_LOW_TX])
    accts = _make_accounts([_LOW_ACCT])
    scored = score_transactions(txns, accts)
    result = summarize_results(scored, _make_chargebacks(9999))  # unknown tx_id
    assert result["chargebacks"].sum() == 0


# ---------------------------------------------------------------------------
# Integration: full pipeline on dataset-like data
# ---------------------------------------------------------------------------

def test_full_pipeline_high_risk_bucket_has_highest_chargeback_rate():
    txns = _make_transactions([_LOW_TX, _HIGH_TX])
    accts = _make_accounts([_LOW_ACCT, _HIGH_ACCT])
    scored = score_transactions(txns, accts)
    # tx 2 (high risk) is the chargeback
    result = summarize_results(scored, _make_chargebacks(2))

    high_rate = result.loc[result["risk_label"] == "high", "chargeback_rate"].iloc[0]
    low_rate = result.loc[result["risk_label"] == "low", "chargeback_rate"].iloc[0]
    assert high_rate > low_rate
