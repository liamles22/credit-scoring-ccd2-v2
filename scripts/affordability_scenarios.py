"""Run illustrative scenarios against a user-supplied CSV of funded loans."""

import argparse
from pathlib import Path

import pandas as pd

from src.affordability import DEFAULT_MAX_PD, summarize_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV: annual_inc,dti,proposed_payment,pd,default_12m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pd", type=float, default=DEFAULT_MAX_PD, help="Illustrative threshold, not a legal limit")
    args = parser.parse_args()
    results = summarize_scenarios(pd.read_csv(args.input), args.max_pd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
