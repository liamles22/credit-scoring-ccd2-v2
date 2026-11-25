"""Backtest 2017 quarterly deployment dates with maturity-gated development labels."""

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.calibration import InterceptShift
from src.data import PROJECT_ROOT, TARGET
from src.features_config import CATEGORICAL, assert_no_leakage, model_features
from src.lightgbm_features import fit_category_levels, lightgbm_frame
from src.metrics import gini
from src.split import OOT, PROCESSED_PATH, SEED, as_of_masks

OUT = PROJECT_ROOT / "reports" / "tables" / "as_of_backtest_2017.csv"


def binary_target(part: pd.DataFrame, name: str, as_of: pd.Timestamp) -> pd.Series:
    y = part[TARGET]
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError(f"{as_of.date()} {name} needs both outcome classes and no invalid labels")
    return y


def run_fold(data: pd.DataFrame, as_of: pd.Timestamp, features: list) -> dict:
    masks = as_of_masks(data, as_of)
    fit = data.loc[masks["fit"], [*features, TARGET]]
    stop = data.loc[masks["early_stop"], [*features, TARGET]]
    cal = data.loc[masks["calibration"], [*features, TARGET]]
    y_fit = binary_target(fit, "fit", as_of)
    y_stop = binary_target(stop, "early stopping", as_of)
    y_cal = binary_target(cal, "calibration", as_of)

    levels = fit_category_levels(fit, CATEGORICAL)
    model = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.05, num_leaves=31, min_child_samples=100,
        metric="auc", random_state=SEED, n_jobs=8, verbosity=-1,
    )
    model.fit(
        lightgbm_frame(fit, features, levels), y_fit,
        eval_X=(lightgbm_frame(stop, features, levels),), eval_y=(y_stop,),
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    raw_cal = model.predict_proba(lightgbm_frame(cal, features, levels))[:, 1]
    calibrator = InterceptShift().fit(raw_cal, y_cal)

    evaluation = data.loc[masks["evaluation"], [*features, TARGET, "sub_grade"]]
    raw = model.predict_proba(lightgbm_frame(evaluation, features, levels))[:, 1]
    calibrated = calibrator.transform(raw)
    y = binary_target(evaluation, "evaluation", as_of)
    benchmark = evaluation["sub_grade"].rank(method="dense")
    raw_gini, benchmark_gini = gini(y, raw), gini(y, benchmark)

    return {
        "as_of_date": as_of.date(),
        "fit_loans": len(fit),
        "fit_last_issue_month": data.loc[masks["fit"], "issue_d"].max().strftime("%Y-%m"),
        "early_stop_loans": len(stop),
        "early_stop_last_issue_month": data.loc[masks["early_stop"], "issue_d"].max().strftime("%Y-%m"),
        "calibration_loans": len(cal),
        "calibration_last_issue_month": data.loc[masks["calibration"], "issue_d"].max().strftime("%Y-%m"),
        "evaluation_loans": len(evaluation),
        "evaluation_defaults": int(y.sum()),
        "gini": raw_gini,
        "benchmark_gini": benchmark_gini,
        "gini_difference": raw_gini - benchmark_gini,
        "observed_over_predicted": y.mean() / raw.mean(),
        "observed_over_predicted_calibrated": y.mean() / calibrated.mean(),
        "brier": brier_score_loss(y, raw),
        "brier_calibrated": brier_score_loss(y, calibrated),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH, help="Processed loans from notebook 02")
    parser.add_argument("--output", type=Path, default=OUT, help="Destination CSV")
    args = parser.parse_args()

    features = model_features()
    assert_no_leakage(features)
    samples = pd.read_parquet(args.input, columns=["issue_d", TARGET, "sub_grade", *features])
    rows = []
    for as_of in pd.date_range(*OOT, freq="QS"):
        print(f"Backtesting {as_of.date()}", flush=True)
        rows.append(run_fold(samples, as_of, features))
    summary = pd.DataFrame(rows).round(6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
