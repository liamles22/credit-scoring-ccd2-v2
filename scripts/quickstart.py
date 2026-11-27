"""Fit and reload LightGBM, then try affordability scenarios on synthetic loans."""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.affordability import summarize_scenarios
from src.features import AFFORDABILITY, add_affordability
from src.features_config import assert_no_leakage
from src.lightgbm_features import fit_category_levels, lightgbm_frame, saved_category_levels

FEATURES = ["annual_inc", "dti", "purpose", *AFFORDABILITY]
CATEGORICAL = ["purpose"]


def synthetic_loans(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate features and outcomes from an illustrative, seeded process."""
    income = rng.uniform(25_000, 110_000, size=n)
    dti = rng.uniform(5, 50, size=n)
    amount = rng.uniform(2_000, 20_000, size=n)
    purpose = rng.choice(["car", "credit_card", "small_business"], size=n)
    risk = 0.012 + 0.002 * dti + 0.12 * amount / income + 0.025 * (purpose == "small_business")
    revolving = rng.uniform(0, 15_000, size=n)
    return pd.DataFrame({
        "annual_inc": income,
        "dti": dti,
        "loan_amnt": amount,
        "purpose": purpose,
        "revol_bal": revolving,
        "tot_cur_bal": revolving + rng.uniform(0, 40_000, size=n),
        "total_rev_hi_lim": revolving + rng.uniform(1_000, 20_000, size=n),
        "proposed_payment": amount / 32,
        "default_12m": (rng.random(n) < risk).astype("int8"),
    })


def run(output_dir: Path) -> None:
    assert_no_leakage(FEATURES)
    rng = np.random.default_rng(42)
    training = synthetic_loans(1600, rng)
    holdout = synthetic_loans(400, rng)
    holdout.loc[0, "purpose"] = "other"  # Exercise a category unseen during training.

    fit = add_affordability(training)
    evaluation = add_affordability(holdout)
    levels = fit_category_levels(fit, CATEGORICAL)
    model = lgb.LGBMClassifier(
        n_estimators=40, learning_rate=0.08, num_leaves=15, min_child_samples=30,
        n_jobs=1, random_state=42, verbosity=-1,
    )
    model.fit(lightgbm_frame(fit, FEATURES, levels), fit["default_12m"])
    expected = model.predict_proba(lightgbm_frame(evaluation, FEATURES, levels))[:, 1]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lightgbm.txt"
    model.booster_.save_model(str(model_path))
    restored = lgb.Booster(model_file=str(model_path))
    scored = lightgbm_frame(evaluation, FEATURES, saved_category_levels(restored, CATEGORICAL))
    probabilities = restored.predict(scored, num_threads=1)
    np.testing.assert_allclose(probabilities, expected)

    predictions = holdout[["annual_inc", "dti", "proposed_payment", "default_12m"]].assign(pd=probabilities)
    if not predictions["pd"].between(0, 1).all():
        raise ValueError("The reloaded model produced invalid probabilities")
    scenarios = summarize_scenarios(predictions)
    dataset = pd.concat([training.assign(sample="train"), holdout.assign(sample="holdout")], ignore_index=True)
    dataset.to_csv(output_dir / "synthetic_loans.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    scenarios.to_csv(output_dir / "affordability.csv", index=False)
    print(f"Scored {len(predictions):,} synthetic holdout loans. Outputs: {output_dir.resolve()}")
    print(scenarios[["scenario", "policy", "approved", "approval_rate"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"), help="Demo artifact directory")
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
