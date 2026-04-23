from __future__ import annotations

from typing import Dict

_REQUIRED_FIELDS = {
    "device_risk_score",
    "is_international",
    "amount_usd",
    "velocity_24h",
    "failed_logins_24h",
    "prior_chargebacks",
}


def _validate(tx: Dict) -> None:
    missing = _REQUIRED_FIELDS - tx.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if not (0 <= tx["device_risk_score"] <= 100):
        raise ValueError("device_risk_score must be between 0 and 100")
    if tx["is_international"] not in (0, 1):
        raise ValueError("is_international must be 0 or 1")
    if tx["amount_usd"] < 0:
        raise ValueError("amount_usd must be non-negative")
    if tx["velocity_24h"] < 0:
        raise ValueError("velocity_24h must be non-negative")
    if tx["failed_logins_24h"] < 0:
        raise ValueError("failed_logins_24h must be non-negative")
    if tx["prior_chargebacks"] < 0:
        raise ValueError("prior_chargebacks must be non-negative")


def score_transaction(tx: Dict) -> int:
    """Return a simple fraud risk score from 0 to 100."""
    _validate(tx)
    score = 0

    if tx["device_risk_score"] >= 70:
        score += 25
    elif tx["device_risk_score"] >= 40:
        score += 10

    if tx["is_international"] == 1:
        score += 15

    if tx["amount_usd"] >= 1000:
        score += 25
    elif tx["amount_usd"] >= 500:
        score += 10

    if tx["velocity_24h"] >= 6:
        score += 20
    elif tx["velocity_24h"] >= 3:
        score += 5

    # Prior login failures can signal account takeover.
    if tx["failed_logins_24h"] >= 5:
        score += 20
    elif tx["failed_logins_24h"] >= 2:
        score += 10

    if tx["prior_chargebacks"] >= 2:
        score += 20
    elif tx["prior_chargebacks"] == 1:
        score += 5

    return max(0, min(score, 100))


def label_risk(score: int) -> str:
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"
