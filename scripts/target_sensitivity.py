"""Audit unresolved labels and compare three treatments of Late (31-120 days)."""

import argparse
from pathlib import Path

import pandas as pd

from scripts.as_of_backtest import run_fold
from src.data import (
    INTERIM_PATH, LATE_POLICIES, PROJECT_ROOT, TARGET, TARGET_UNKNOWN_REASON,
    add_target_12m, labelled_loans, load_interim, make_modelling_base,
)
from src.features_config import assert_no_leakage, model_features
from src.split import (
    CANDIDATE_WINDOWS, CHOSEN_WINDOW, OOT, as_of_masks, issued_between, select_scope, shift_to_end,
)

TABLES = PROJECT_ROOT / "reports" / "tables"
PAYMENT_COLS = ["loan_status", "installment", "total_rec_prncp", "total_rec_int"]


def load_cohort(path: Path, features: list) -> pd.DataFrame:
    columns = list(dict.fromkeys([
        *(feature for feature in features if feature != "credit_hist_months"),
        *PAYMENT_COLS, "issue_d", "issue_year", "last_pymnt_d", "earliest_cr_line",
        "application_type", "sub_grade",
    ]))
    base = select_scope(make_modelling_base(load_interim(path, columns=columns)))
    start, _ = shift_to_end(CANDIDATE_WINDOWS[CHOSEN_WINDOW])
    base = base.loc[issued_between(base, (start, OOT[1]))].copy()
    issue, earliest = base["issue_d"].dt, base["earliest_cr_line"].dt
    base["credit_hist_months"] = (issue.year - earliest.year) * 12 + issue.month - earliest.month
    return base


def audit_by_month(base: pd.DataFrame) -> pd.DataFrame:
    counts = base.groupby(base["issue_d"].dt.to_period("M").rename("issue_month")).agg(
        loans=(TARGET, "size"),
        labelled_loans=(TARGET, "count"),
        defaults=(TARGET, "sum"),
        unknown_status=(TARGET_UNKNOWN_REASON, lambda s: s.eq("unknown_status").sum()),
        invalid_payment_data=(TARGET_UNKNOWN_REASON, lambda s: s.eq("invalid_payment_data").sum()),
        ambiguous_late_status=(TARGET_UNKNOWN_REASON, lambda s: s.eq("ambiguous_late_status").sum()),
    )
    counts["unresolved"] = counts["loans"] - counts["labelled_loans"]
    counts["unresolved_share"] = counts["unresolved"] / counts["loans"]
    return counts.reset_index()


def sensitivity(base: pd.DataFrame, features: list) -> pd.DataFrame:
    rows = []
    dates = pd.date_range(*OOT, freq="QS")
    totals = {date: int(as_of_masks(base, date)["evaluation"].sum()) for date in dates}
    payments = base[PAYMENT_COLS]
    data = base[["issue_d", "sub_grade", *features]]
    for policy in LATE_POLICIES:
        targets = add_target_12m(payments.copy(), late_policy=policy)
        known = labelled_loans(data.assign(**{TARGET: targets[TARGET]}))
        for date in dates:
            print(f"{policy}: backtesting {date.date()}", flush=True)
            row = run_fold(known, date, features)
            row["late_policy"] = policy
            row["evaluation_excluded"] = totals[date] - row["evaluation_loans"]
            rows.append(row)
    return pd.DataFrame(rows).round(6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTERIM_PATH, help="Interim loans from notebook 01")
    parser.add_argument("--output-dir", type=Path, default=TABLES)
    args = parser.parse_args()

    features = model_features()
    assert_no_leakage(features)
    base = load_cohort(args.input, features)
    monthly = audit_by_month(base)
    results = sensitivity(base, features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly["unresolved_share"] = monthly["unresolved_share"].round(8)
    monthly.to_csv(args.output_dir / "target_label_coverage.csv", index=False)
    results.to_csv(args.output_dir / "target_late_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
