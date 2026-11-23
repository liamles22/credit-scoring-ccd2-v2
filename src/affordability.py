"""Illustrative affordability scenarios; not a credit decision or compliance tool."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    net_income_fraction: float
    living_cost_fraction: float
    income_shock: float = 0.0
    expense_shock: float = 0.0

    def __post_init__(self):
        values = (self.net_income_fraction, self.living_cost_fraction, self.income_shock, self.expense_shock)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Scenario parameters must be finite")
        if not 0 < self.net_income_fraction <= 1 or not 0 <= self.living_cost_fraction <= 1:
            raise ValueError("Income and expense fractions are outside their valid ranges")
        if not 0 <= self.income_shock < 1 or self.expense_shock < 0:
            raise ValueError("Invalid shock")


def evaluate_affordability(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Calculate a scenario balance from application-time proxy inputs.

    annual_inc is gross annual income, dti is existing monthly debt / gross monthly
    income in percent, and proposed_payment is the offered monthly payment.
    Missing or invalid inputs remain unknown, never automatically approved.
    """
    required = ["annual_inc", "dti", "proposed_payment"]
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values).all(axis=1) & values["annual_inc"].gt(0)
    valid &= values["dti"].ge(0) & values["proposed_payment"].ge(0)
    gross = values["annual_inc"] / 12
    net = gross * scenario.net_income_fraction
    existing_debt = gross * values["dti"] / 100
    living_cost = net * scenario.living_cost_fraction * (1 + scenario.expense_shock)
    balance = net * (1 - scenario.income_shock) - existing_debt - values["proposed_payment"] - living_cost
    return pd.DataFrame({"balance": balance.where(valid), "valid": valid}, index=frame.index)


def policy_masks(frame: pd.DataFrame, scenario: Scenario, max_pd: float) -> dict:
    """Return mutually comparable masks and the common complete-case population."""
    if not np.isfinite(max_pd) or not 0 <= max_pd <= 1:
        raise ValueError("max_pd must be between zero and one")
    affordability = evaluate_affordability(frame, scenario)
    pd_estimate = pd.to_numeric(frame["pd"], errors="coerce")
    valid = affordability["valid"] & np.isfinite(pd_estimate) & pd_estimate.between(0, 1)
    risk = valid & pd_estimate.le(max_pd)
    affordable = valid & affordability["balance"].ge(0)
    return {"eligible": valid, "pd_only": risk, "affordability_only": affordable,
            "combined": risk & affordable}


def summarize(frame: pd.DataFrame, scenario: Scenario, max_pd: float) -> pd.DataFrame:
    masks = policy_masks(frame, scenario, max_pd)
    valid = masks["eligible"]
    target = pd.to_numeric(frame["default_12m"], errors="coerce")
    if (frame["default_12m"].notna() & target.isna()).any() or not target.dropna().isin([0, 1]).all():
        raise ValueError("default_12m must be 0, 1, or missing")
    rows = []
    for name in ("pd_only", "affordability_only", "combined"):
        approved = masks[name]
        observed = approved & target.notna()
        rows.append({"scenario": scenario.name, "policy": name, "rows": len(frame),
                     "eligible": int(valid.sum()), "excluded": int((~valid).sum()),
                     "approved": int(approved.sum()),
                     "approval_rate": approved.sum() / valid.sum() if valid.sum() else np.nan,
                     "observed_outcomes": int(observed.sum()),
                     "default_rate": target[observed].mean()})
    return pd.DataFrame(rows)
