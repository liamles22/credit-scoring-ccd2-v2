import numpy as np
import pandas as pd
import pytest

from src.affordability import Scenario, evaluate_affordability, policy_masks, summarize


def test_balance_and_stress():
    frame = pd.DataFrame({"annual_inc": [12000], "dti": [10], "proposed_payment": [100]})
    base = evaluate_affordability(frame, Scenario("base", 0.8, 0.5))
    stressed = evaluate_affordability(frame, Scenario("stress", 0.8, 0.5, 0.25))
    assert base.loc[0, "balance"] == pytest.approx(200)
    assert stressed.loc[0, "balance"] == pytest.approx(0)


def test_invalid_inputs_are_excluded_and_combined_is_intersection():
    frame = pd.DataFrame({"annual_inc": [12000, 12000, 0, 12000], "dti": [10, 10, 10, np.nan],
                          "proposed_payment": [100, 1000, 100, 100], "pd": [0.01, 0.02, 0.01, 0.01],
                          "default_12m": [0, 1, 0, 0]})
    masks = policy_masks(frame, Scenario("base", 0.8, 0.5), 0.05)
    assert masks["eligible"].tolist() == [True, True, False, False]
    assert masks["combined"].tolist() == [True, False, False, False]
    result = summarize(frame, Scenario("base", 0.8, 0.5), 0.05).set_index("policy")
    assert result.loc["combined", "approval_rate"] == 0.5
    assert result.loc["pd_only", "default_rate"] == 0.5


def test_invalid_scenario_and_threshold():
    with pytest.raises(ValueError):
        Scenario("bad", 1.5, 0.5)
    frame = pd.DataFrame({"annual_inc": [12000], "dti": [0], "proposed_payment": [0], "pd": [0.1]})
    with pytest.raises(ValueError):
        policy_masks(frame, Scenario("base", 0.8, 0.5), 2)


def test_invalid_outcome_is_not_silently_treated_as_missing():
    frame = pd.DataFrame({"annual_inc": [12000], "dti": [0], "proposed_payment": [0],
                          "pd": [0.1], "default_12m": ["unknown label"]})
    with pytest.raises(ValueError, match="default_12m"):
        summarize(frame, Scenario("base", 0.8, 0.5), 0.2)


def test_outcome_coverage_and_empty_approval():
    frame = pd.DataFrame({"annual_inc": [12000, 12000], "dti": [0, 0],
                          "proposed_payment": [0, 0], "pd": [0.1, 0.1],
                          "default_12m": [1, np.nan]})
    scenario = Scenario("base", 0.8, 0.5)
    result = summarize(frame, scenario, 0.2).set_index("policy")
    assert result.loc["pd_only", "observed_defaults"] == 1
    assert result.loc["pd_only", "missing_outcomes"] == 1
    assert result.loc["pd_only", "outcome_coverage"] == 0.5
    assert result.loc["pd_only", "default_rate"] == 1
    empty = summarize(frame, scenario, 0.05).set_index("policy")
    assert np.isnan(empty.loc["pd_only", "outcome_coverage"])
    assert np.isnan(empty.loc["pd_only", "default_rate"])
