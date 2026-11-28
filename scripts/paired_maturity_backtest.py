"""Compare retrospective and maturity-gated LightGBM on the same OOT loans."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.as_of_backtest import binary_target, fit_model, predict_model
from src.data import TARGET
from src.features_config import assert_no_leakage, model_features
from src.metrics import gini
from src.split import OOT, PROCESSED_PATH, SEED, TRAIN_END, as_of_masks, development_masks

OUT = PROCESSED_PATH.with_name("paired_maturity_backtest_2017.csv")
N_BOOT = 1000


def paired_intervals(y, as_of_score, retrospective_score, grade, n_boot: int = N_BOOT) -> np.ndarray:
    """Resample defaults and nondefaults separately, sharing draws across all scores."""
    y = np.asarray(y)
    scores = np.stack([as_of_score, retrospective_score, grade])
    nondefault = np.flatnonzero(y == 0)
    default = np.flatnonzero(y == 1)
    rng = np.random.default_rng(SEED)
    differences = np.empty((n_boot, 3))
    for draw in range(n_boot):
        idx = np.concatenate([rng.choice(nondefault, len(nondefault)), rng.choice(default, len(default))])
        as_of_gini, retrospective_gini, grade_gini = (gini(y[idx], score[idx]) for score in scores)
        differences[draw] = (as_of_gini - retrospective_gini,
                             as_of_gini - grade_gini,
                             retrospective_gini - grade_gini)
    return np.quantile(differences, [0.025, 0.975], axis=0).T


def run(data: pd.DataFrame, features: list) -> pd.DataFrame:
    retrospective_masks = development_masks(data, pd.Timestamp(TRAIN_END))
    print(f"Fitting retrospective model through {TRAIN_END}", flush=True)
    retrospective = fit_model(data, retrospective_masks, pd.Timestamp(OOT[0]), features)

    rows = []
    for date in pd.date_range(*OOT, freq="QS"):
        print(f"Fitting and evaluating {date.date()}", flush=True)
        masks = as_of_masks(data, date)
        fitted = fit_model(data, masks, date, features)
        evaluation = data.loc[masks["evaluation"], [*features, TARGET, "sub_grade"]]

        y = binary_target(evaluation, "evaluation", date).to_numpy()
        grade = evaluation["sub_grade"].rank(method="dense").to_numpy()
        as_of_score, as_of_pd = predict_model(fitted, evaluation, features)
        ret_score, ret_pd = predict_model(retrospective, evaluation, features)
        as_of_gini, ret_gini, grade_gini = (gini(y, score) for score in (as_of_score, ret_score, grade))
        ci = paired_intervals(y, as_of_score, ret_score, grade)

        rows.append({
            "as_of_date": date.date(),
            "evaluation_loans": len(evaluation),
            "evaluation_defaults": int(y.sum()),
            "as_of_fit_loans": int(masks["fit"].sum()),
            "retrospective_fit_loans": int(retrospective_masks["fit"].sum()),
            "as_of_last_development_month": data.loc[masks["calibration"], "issue_d"].max().strftime("%Y-%m"),
            "retrospective_last_development_month": TRAIN_END,
            "as_of_gini": as_of_gini,
            "retrospective_gini": ret_gini,
            "benchmark_gini": grade_gini,
            "as_of_minus_retrospective": as_of_gini - ret_gini,
            "as_of_minus_retrospective_ci_low": ci[0, 0],
            "as_of_minus_retrospective_ci_high": ci[0, 1],
            "as_of_minus_benchmark": as_of_gini - grade_gini,
            "as_of_minus_benchmark_ci_low": ci[1, 0],
            "as_of_minus_benchmark_ci_high": ci[1, 1],
            "retrospective_minus_benchmark": ret_gini - grade_gini,
            "retrospective_minus_benchmark_ci_low": ci[2, 0],
            "retrospective_minus_benchmark_ci_high": ci[2, 1],
            "as_of_calibrated_obs_pred": y.mean() / as_of_pd.mean(),
            "retrospective_calibrated_obs_pred": y.mean() / ret_pd.mean(),
        })
    return pd.DataFrame(rows).round(6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH, help="Labelled loans from notebook 02")
    parser.add_argument("--output", type=Path, default=OUT, help="Destination CSV")
    args = parser.parse_args()

    features = model_features()
    assert_no_leakage(features)
    data = pd.read_parquet(args.input, columns=["issue_d", TARGET, "sub_grade", *features])
    results = run(data, features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
