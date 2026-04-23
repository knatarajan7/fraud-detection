import pandas as pd
import pytest
from features import build_model_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _txns(**overrides):
    base = {
        "transaction_id": [1],
        "account_id": [101],
        "amount_usd": [500.0],
        "device_risk_score": [20],
        "is_international": [0],
        "velocity_24h": [1],
        "failed_logins_24h": [0],
        "merchant_category": ["grocery"],
        "channel": ["web"],
        "ip_country": ["US"],
        "chargeback_within_60d": [0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _accts(**overrides):
    base = {
        "account_id": [101],
        "customer_name": ["Alice"],
        "country": ["US"],
        "signup_date": ["2022-01-01"],
        "kyc_level": ["full"],
        "account_age_days": [720],
        "prior_chargebacks": [0],
        "is_vip": ["Y"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Output shape and columns
# ---------------------------------------------------------------------------

def test_output_contains_all_transaction_rows():
    txns = pd.DataFrame(
        {"transaction_id": [1, 2, 3], "account_id": [101, 101, 101],
         "amount_usd": [10, 20, 30], "device_risk_score": [5, 5, 5],
         "is_international": [0, 0, 0], "velocity_24h": [1, 1, 1],
         "failed_logins_24h": [0, 0, 0], "merchant_category": ["g", "g", "g"],
         "channel": ["web", "web", "web"], "ip_country": ["US", "US", "US"],
         "chargeback_within_60d": [0, 0, 0]}
    )
    result = build_model_frame(txns, _accts())
    assert len(result) == 3


def test_output_contains_derived_columns():
    result = build_model_frame(_txns(), _accts())
    assert "is_large_amount" in result.columns
    assert "login_pressure" in result.columns


def test_output_contains_account_columns():
    result = build_model_frame(_txns(), _accts())
    assert "prior_chargebacks" in result.columns
    assert "kyc_level" in result.columns


# ---------------------------------------------------------------------------
# is_large_amount
# ---------------------------------------------------------------------------

def test_is_large_amount_at_threshold():
    result = build_model_frame(_txns(amount_usd=[1000.0]), _accts())
    assert result["is_large_amount"].iloc[0] == 1


def test_is_large_amount_above_threshold():
    result = build_model_frame(_txns(amount_usd=[2500.0]), _accts())
    assert result["is_large_amount"].iloc[0] == 1


def test_is_large_amount_just_below_threshold():
    result = build_model_frame(_txns(amount_usd=[999.99]), _accts())
    assert result["is_large_amount"].iloc[0] == 0


def test_is_large_amount_small_purchase():
    result = build_model_frame(_txns(amount_usd=[14.99]), _accts())
    assert result["is_large_amount"].iloc[0] == 0


def test_is_large_amount_is_integer():
    result = build_model_frame(_txns(amount_usd=[1500.0]), _accts())
    assert result["is_large_amount"].iloc[0] in (0, 1)


# ---------------------------------------------------------------------------
# login_pressure
# ---------------------------------------------------------------------------

def test_login_pressure_none_at_zero():
    result = build_model_frame(_txns(failed_logins_24h=[0]), _accts())
    assert result["login_pressure"].iloc[0] == "none"


def test_login_pressure_low_at_one():
    result = build_model_frame(_txns(failed_logins_24h=[1]), _accts())
    assert result["login_pressure"].iloc[0] == "low"


def test_login_pressure_low_at_two():
    result = build_model_frame(_txns(failed_logins_24h=[2]), _accts())
    assert result["login_pressure"].iloc[0] == "low"


def test_login_pressure_high_at_three():
    result = build_model_frame(_txns(failed_logins_24h=[3]), _accts())
    assert result["login_pressure"].iloc[0] == "high"


def test_login_pressure_high_at_max():
    result = build_model_frame(_txns(failed_logins_24h=[8]), _accts())
    assert result["login_pressure"].iloc[0] == "high"


def test_login_pressure_categories_ordered():
    txns = pd.DataFrame(
        {"transaction_id": [1, 2, 3], "account_id": [101, 101, 101],
         "amount_usd": [10, 10, 10], "device_risk_score": [5, 5, 5],
         "is_international": [0, 0, 0], "velocity_24h": [1, 1, 1],
         "failed_logins_24h": [0, 2, 5], "merchant_category": ["g", "g", "g"],
         "channel": ["web", "web", "web"], "ip_country": ["US", "US", "US"],
         "chargeback_within_60d": [0, 0, 0]}
    )
    result = build_model_frame(txns, _accts())
    assert list(result["login_pressure"]) == ["none", "low", "high"]


# ---------------------------------------------------------------------------
# Join behaviour
# ---------------------------------------------------------------------------

def test_account_data_merged_by_account_id():
    accts = _accts(prior_chargebacks=[3])
    result = build_model_frame(_txns(), accts)
    assert result["prior_chargebacks"].iloc[0] == 3


def test_multiple_transactions_same_account_all_get_account_data():
    txns = pd.DataFrame(
        {"transaction_id": [1, 2], "account_id": [101, 101],
         "amount_usd": [10, 20], "device_risk_score": [5, 5],
         "is_international": [0, 0], "velocity_24h": [1, 1],
         "failed_logins_24h": [0, 0], "merchant_category": ["g", "g"],
         "channel": ["web", "web"], "ip_country": ["US", "US"],
         "chargeback_within_60d": [0, 0]}
    )
    accts = _accts(prior_chargebacks=[2])
    result = build_model_frame(txns, accts)
    assert list(result["prior_chargebacks"]) == [2, 2]


def test_unmatched_account_id_produces_nan():
    txns = _txns(account_id=[999])  # no matching account
    result = build_model_frame(txns, _accts())
    assert result["prior_chargebacks"].isna().iloc[0]


def test_left_join_keeps_all_transactions():
    txns = pd.DataFrame(
        {"transaction_id": [1, 2], "account_id": [101, 999],
         "amount_usd": [10, 20], "device_risk_score": [5, 5],
         "is_international": [0, 0], "velocity_24h": [1, 1],
         "failed_logins_24h": [0, 0], "merchant_category": ["g", "g"],
         "channel": ["web", "web"], "ip_country": ["US", "US"],
         "chargeback_within_60d": [0, 0]}
    )
    result = build_model_frame(txns, _accts())
    assert len(result) == 2
