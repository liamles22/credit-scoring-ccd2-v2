"""Sample scope and time-based split.

The out-of-time year is fixed in advance (2017). The training window is chosen in
notebook 02 on a separate validation slice (H2 2016), never on the out-of-time year,
so that the out-of-time Gini stays an honest estimate.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import BUFFER_MONTHS, HORIZON_MONTHS, PROJECT_ROOT, TARGET, labelled_loans, observed_long_enough

OOT = ("2017-01", "2017-12")
WINDOW_VALIDATION = ("2016-07", "2016-12")  # used only to choose the training window
CANDIDATE_WINDOWS = {
    "H1 2016": ("2016-01", "2016-06"),
    "2015 - H1 2016": ("2015-01", "2016-06"),
    "2013 - H1 2016": ("2013-01", "2016-06"),
}
TRAIN_END = "2016-12"  # the chosen window is shifted to end here for the final split
TEST_SHARE = 0.2
SEED = 42

# Result of the experiment in notebook 02: highest validation Gini, consistent in every month.
CHOSEN_WINDOW = "2013 - H1 2016"
CHOSEN_FEATURE_SET = "full"  # base features plus the bureau fields that appear in 2016

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "samples.parquet"


def select_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Individual applications only: joint applications appear late 2015 with their own fields."""
    return df[df["application_type"] == "Individual"].reset_index(drop=True)


def issued_between(df: pd.DataFrame, window) -> pd.Series:
    """True for loans issued in the inclusive month range `window` = ("YYYY-MM", "YYYY-MM")."""
    month = df["issue_d"].dt.to_period("M")
    return month.between(pd.Period(window[0], "M"), pd.Period(window[1], "M"))


def shift_to_end(window, end: str = TRAIN_END):
    """Same number of months as `window`, ending at `end`."""
    start, stop = pd.Period(window[0], "M"), pd.Period(window[1], "M")
    new_end = pd.Period(end, "M")
    return (str(new_end - (stop - start).n), str(new_end))


def final_split(df: pd.DataFrame, train_window, test_share: float = TEST_SHARE,
                seed: int = SEED) -> pd.DataFrame:
    """Label loans as train / test (random, stratified, inside the window) or oot.

    Drop unresolved outcomes and loans outside the development and OOT windows.
    """
    df = labelled_loans(df)
    in_window = df[issued_between(df, train_window)]
    train_idx, test_idx = train_test_split(
        in_window.index, test_size=test_share, stratify=in_window[TARGET], random_state=seed
    )
    sample = pd.Series(pd.NA, index=df.index, dtype="string")
    sample[train_idx] = "train"
    sample[test_idx] = "test"
    sample[issued_between(df, OOT)] = "oot"
    out = df.assign(sample=sample)
    return out[out["sample"].notna()].reset_index(drop=True)


def development_masks(df: pd.DataFrame, last_issue_month: pd.Timestamp) -> dict[str, pd.Series]:
    """Assign the last three issue months to calibration and the preceding three to early stopping."""
    issued = df["issue_d"].dt.to_period("M")
    last = pd.Period(last_issue_month, freq="M")
    calibration_start = last - 2
    stop_start = calibration_start - 3
    return {
        "fit": issued.lt(stop_start),
        "early_stop": issued.ge(stop_start) & issued.lt(calibration_start),
        "calibration": issued.between(calibration_start, last),
    }


def as_of_masks(df: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, pd.Series]:
    """Partition a backtest at a month start using only mature development labels."""
    as_of = pd.Timestamp(as_of)
    latest_mature_issue = as_of - pd.DateOffset(months=HORIZON_MONTHS + BUFFER_MONTHS)
    issued = df["issue_d"]
    mature = observed_long_enough(df, as_of)
    development = development_masks(df, latest_mature_issue)
    return {
        **{name: mask & mature for name, mask in development.items()},
        "evaluation": issued.ge(as_of) & issued.lt(as_of + pd.DateOffset(months=3)),
    }
