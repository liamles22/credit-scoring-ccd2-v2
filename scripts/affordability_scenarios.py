"""Run illustrative scenarios against a user-supplied CSV of funded loans."""

import argparse
from pathlib import Path

import pandas as pd

from src.affordability import Scenario, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV: annual_inc,dti,proposed_payment,pd,default_12m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pd", type=float, default=0.08, help="Illustrative threshold, not a legal limit")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    scenarios = [Scenario("base", 0.75, 0.55), Scenario("income_down_20pct", 0.75, 0.55, 0.20),
                 Scenario("expenses_up_20pct", 0.75, 0.55, expense_shock=0.20)]
    results = pd.concat([summarize(frame, scenario, args.max_pd) for scenario in scenarios], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
