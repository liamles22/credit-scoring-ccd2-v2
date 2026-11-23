# Affordability scenarios: exploratory prototype

This prototype compares PD-only, affordability-only and combined screening on a common complete-case sample. It is an illustration, not a CCD2 compliance test, a validated affordability model or an operational credit decision. The research question is whether a scenario-based affordability screen selects a different funded-loan population from the existing risk ranking.

## Inputs and assumptions

Supply a CSV with `annual_inc` (annual gross income), `dti` (existing monthly debt as percent of gross monthly income), `proposed_payment` (monthly), `pd` (out-of-sample 12-month PD, from 0 to 1) and `default_12m` (0/1, missing if unknown). Inputs must be known at the simulated decision point. Provide an independently scored holdout cohort: this script does not train or validate the PD model. Do not treat in-sample predictions as out-of-sample.

Monthly gross income = annual income / 12; existing debt = gross income × DTI / 100; assumed net income = gross income × 0.75; assumed living costs = net income × 0.55. The base balance is net income minus living costs, debt and proposed payment. Stress scenarios reduce net income by 20% or increase living costs by 20%, holding other quantities fixed. These values are illustrative, not empirically estimated or legal thresholds. A positive balance is not proof of affordability. The PD cutoff 0.08 is similarly illustrative and configurable. Re-run with alternative assumptions; do not choose values based on the final test cohort.

The Lending Club DTI definition excludes mortgages and the new loan; it is not a comprehensive measure of obligations. Self-reported gross income is not verified take-home pay. Living costs, taxes and household composition are unknown. Original `installment` depends on the lender's offered rate; if it is used as `proposed_payment`, this is a post-offer scenario, not a score available before pricing. Avoid using it as a model feature or claiming this reconstructs the original underwriting. Check joint applications, units, missing values and timing before use.

Invalid or missing income, debt, payment or PD are excluded from **all** policies and reported. Missing outcomes remain excluded from default-rate denominators, with counts shown. The combined rule is the intersection of the other rules, so it generally approves fewer cases; report both approval and default rates rather than presenting this as an equal-approval comparison. A zero approval group has an undefined default rate.

The output reports observed defaults, missing outcomes and outcome coverage for each approved group. Coverage is observed outcomes divided by approved loans; it is undefined when no loans are approved. The default rate uses observed outcomes only and may be biased if outcomes are missing non-randomly. Report coverage alongside it and investigate missingness before comparing policies.

Only funded applicants have outcomes. Results cannot establish what would have happened to rejected applicants, prove causal effects or measure population-level approval. Data are US historical loans and cannot establish performance for EU consumers or BNPL. The predefined stress values do not represent estimated real-world probabilities. The scenario does not generate income paths or a probability of financial distress.

## Run

```bash
python -m scripts.affordability_scenarios path/to/holdout.csv --output reports/tables/affordability_scenarios.csv --max-pd 0.08
```

No data or fitted models are bundled. No empirical output is included until the input data and predictions are supplied. Record sample dates, source, selection, missingness, fitted model, scenario assumptions, and threshold selection when reporting results. Existing 2017 OOT and 2018 confirmation cohorts have already informed other decisions; do not describe either as untouched for this new analysis.

## Regulatory references

[CCD2, Article 18](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023L2225) addresses creditworthiness assessments. [EBA guidelines on loan origination and monitoring](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring) discuss documented information and expenditure; their scope depends on institution and product. Neither reference prescribes the illustrative numerical assumptions above. A national-law and product-specific review is required.
