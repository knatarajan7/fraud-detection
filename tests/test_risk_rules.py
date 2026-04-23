import pytest
from risk_rules import label_risk, score_transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_tx():
    """Minimal low-risk transaction used to isolate individual signals."""
    return {
        "device_risk_score": 5,
        "is_international": 0,
        "amount_usd": 10,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }


# ---------------------------------------------------------------------------
# Existing tests (must stay passing)
# ---------------------------------------------------------------------------

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    tx = {**_clean_tx(), "amount_usd": 1200}
    assert score_transaction(tx) >= 25


# ---------------------------------------------------------------------------
# Bug 1 — device risk score
# ---------------------------------------------------------------------------

def test_high_risk_device_raises_score():
    baseline = score_transaction(_clean_tx())
    tx = {**_clean_tx(), "device_risk_score": 85}
    assert score_transaction(tx) > baseline


def test_medium_risk_device_raises_score_less_than_high():
    tx_medium = {**_clean_tx(), "device_risk_score": 50}
    tx_high = {**_clean_tx(), "device_risk_score": 85}
    assert score_transaction(tx_medium) < score_transaction(tx_high)


def test_safe_device_does_not_add_risk():
    tx_safe = {**_clean_tx(), "device_risk_score": 5}
    tx_risky = {**_clean_tx(), "device_risk_score": 85}
    assert score_transaction(tx_safe) < score_transaction(tx_risky)


# ---------------------------------------------------------------------------
# Bug 2 — international flag
# ---------------------------------------------------------------------------

def test_international_transaction_raises_score():
    baseline = score_transaction(_clean_tx())
    tx = {**_clean_tx(), "is_international": 1}
    assert score_transaction(tx) > baseline


# ---------------------------------------------------------------------------
# Bug 3 — velocity in 24h
# ---------------------------------------------------------------------------

def test_high_velocity_raises_score():
    baseline = score_transaction(_clean_tx())
    tx = {**_clean_tx(), "velocity_24h": 8}
    assert score_transaction(tx) > baseline


def test_moderate_velocity_raises_score_less_than_high():
    tx_moderate = {**_clean_tx(), "velocity_24h": 4}
    tx_high = {**_clean_tx(), "velocity_24h": 8}
    assert score_transaction(tx_moderate) < score_transaction(tx_high)


# ---------------------------------------------------------------------------
# Bug 4 — prior chargebacks
# ---------------------------------------------------------------------------

def test_one_prior_chargeback_raises_score():
    baseline = score_transaction(_clean_tx())
    tx = {**_clean_tx(), "prior_chargebacks": 1}
    assert score_transaction(tx) > baseline


def test_multiple_prior_chargebacks_raises_score_more():
    tx_one = {**_clean_tx(), "prior_chargebacks": 1}
    tx_two = {**_clean_tx(), "prior_chargebacks": 2}
    assert score_transaction(tx_two) > score_transaction(tx_one)


# ---------------------------------------------------------------------------
# End-to-end: known fraud transactions from transactions.csv
# ---------------------------------------------------------------------------

def test_known_fraud_tx_50011_scores_high():
    # $1,400 crypto, device 85, international, velocity 8, 7 failed logins, 1 prior CB
    tx = {
        "device_risk_score": 85,
        "is_international": 1,
        "amount_usd": 1400,
        "velocity_24h": 8,
        "failed_logins_24h": 7,
        "prior_chargebacks": 1,
    }
    assert score_transaction(tx) >= 60
    assert label_risk(score_transaction(tx)) == "high"


def test_known_fraud_tx_50006_scores_high():
    # $399, device 77, international, velocity 7, 6 failed logins, 3 prior CBs
    tx = {
        "device_risk_score": 77,
        "is_international": 1,
        "amount_usd": 399.99,
        "velocity_24h": 7,
        "failed_logins_24h": 6,
        "prior_chargebacks": 3,
    }
    assert score_transaction(tx) >= 60
    assert label_risk(score_transaction(tx)) == "high"


def test_clean_transaction_scores_low():
    # Low-risk domestic grocery purchase
    tx = {
        "device_risk_score": 8,
        "is_international": 0,
        "amount_usd": 45.20,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    assert label_risk(score_transaction(tx)) == "low"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_field_raises():
    tx = _clean_tx()
    del tx["device_risk_score"]
    with pytest.raises(ValueError, match="Missing required fields"):
        score_transaction(tx)


def test_multiple_missing_fields_raises():
    tx = _clean_tx()
    del tx["device_risk_score"]
    del tx["amount_usd"]
    with pytest.raises(ValueError, match="Missing required fields"):
        score_transaction(tx)


def test_device_risk_score_above_100_raises():
    tx = {**_clean_tx(), "device_risk_score": 150}
    with pytest.raises(ValueError, match="device_risk_score"):
        score_transaction(tx)


def test_device_risk_score_negative_raises():
    tx = {**_clean_tx(), "device_risk_score": -1}
    with pytest.raises(ValueError, match="device_risk_score"):
        score_transaction(tx)


def test_invalid_is_international_raises():
    tx = {**_clean_tx(), "is_international": 2}
    with pytest.raises(ValueError, match="is_international"):
        score_transaction(tx)


def test_negative_amount_raises():
    tx = {**_clean_tx(), "amount_usd": -0.01}
    with pytest.raises(ValueError, match="amount_usd"):
        score_transaction(tx)


def test_negative_velocity_raises():
    tx = {**_clean_tx(), "velocity_24h": -1}
    with pytest.raises(ValueError, match="velocity_24h"):
        score_transaction(tx)


def test_negative_failed_logins_raises():
    tx = {**_clean_tx(), "failed_logins_24h": -1}
    with pytest.raises(ValueError, match="failed_logins_24h"):
        score_transaction(tx)


def test_negative_prior_chargebacks_raises():
    tx = {**_clean_tx(), "prior_chargebacks": -1}
    with pytest.raises(ValueError, match="prior_chargebacks"):
        score_transaction(tx)
