# CCD2: evidence-based gap assessment

Assessment date: 20 November 2026. This is a desk-based research exercise against Directive (EU) 2023/2225, not a legal opinion or certification. Lending Club data describe US loans originated before the directive applied; no EU product or national implementing law has been assessed. National measures were due to apply from 20 November 2026; implementation must be checked for the relevant country.

## Verdict

**Compliance cannot be established.** The project estimates 12-month default risk; a probability of default alone does not establish that a borrower can sustainably meet payments. The project does not implement the full consumer-credit lifecycle. Evidence below comes from existing project artifacts and has not been independently reproduced for this assessment; loan-level data and fitted models are not included in the repository.

Statuses describe available evidence, not legal compliance: **partial** means relevant evidence exists but does not establish the requirement; **not evidenced** means the necessary records are unavailable; **outside scope** means the project does not implement the activity.

## Requirements and evidence

| Directive | Assessment topic | Project evidence | Status and missing evidence |
|---|---|---|---|
| Art. 18(1)–(3) | Assess creditworthiness in the consumer's interest using relevant information, including income, expenses and other financial circumstances, proportionate to the credit | `src/features_config.py` lists self-reported income and DTI; DTI excludes mortgage and the new loan. `reports/validation_report.md` describes a default model. | **Partial.** Living expenses, complete obligations, verification, product-specific payments and sustainable repayment are not demonstrated. |
| Art. 18(5) | Grant credit only when the assessment indicates obligations are likely to be met as required | `reports/tables/07_strategies.csv` explores approval strategies on funded US loans. | **Partial.** Risk ranking is not a complete affordability policy. Rejected applicants and their outcomes are unavailable; no EU operational decision records are supplied. |
| Art. 18(6) | Inform consumers in advance what information and independent evidence are needed and the timeframe | No customer application process is included. | **Not evidenced.** Obtain notices, requests and timestamps. |
| Art. 18(8) | When automated processing is used, inform consumers and enable human intervention, an explanation of the assessment and decision, expression of their point of view, and a review | Appendix A.4 of `reports/validation_report.md` describes experimental reason codes. | **Partial.** Wording lacks legal and consumer review, some factors lack direction, and the first reason changes across refits for about 10% of sampled cases. No notice, human review, challenge channel or delivery records are supplied. |
| Art. 18(9) | Notify consumers without delay of rejection and, where applicable, that the assessment used automated processing | Reasons are generated offline for loans above an experimental cutoff. | **Not evidenced.** Simulated declines of funded loans are not actual rejections; no notification or delivery records exist. |
| Art. 19 | Database access and information after rejection based on database consultation | Bureau-related fields appear in `src/features_config.py`. | **Not evidenced.** Database consultation and applicant notifications are unavailable. |
| Arts. 10–11 and 14 | Pre-contract information and adequate explanations | No EU product documents or customer journey. | **Outside scope.** Review disclosures, timing and assistance. |
| Art. 26 | Right of withdrawal | No contract administration or withdrawal workflow. | **Outside scope.** Review contract terms and operational records. |
| Art. 31 | National measures addressing excessive rates, APR or total cost | US pricing assumptions do not review EU national limits. | **Outside scope.** Check the relevant country's rules and product pricing. |

This is a targeted assessment, not an exhaustive checklist. Article 18 includes additional safeguards and restrictions. Product applicability and exemptions must be assessed under Article 2; the official text and national law govern interpretation.

## Next research steps

1. Choose a member state and a specific product; map its national implementing provisions and exemptions to a versioned requirements register.
2. With suitable consent and access, collect application-time income, essential expenses, existing debt payments, proposed payment and verification evidence. Define an affordability policy with domain and legal review; do not present an arbitrary threshold as a statutory rule.
3. Evaluate outcomes over time and across groups where lawful and statistically possible. Funded-only data cannot establish rejected applicants' outcomes; the ZIP3 area proxies in this repository do not establish individual fairness.
4. Test explanations with consumers and audit actual notice, delivery, human intervention and review. Offline reason codes alone do not establish these controls.
5. Inspect contracts, disclosures and servicing records separately; record evidence, owner and unresolved gaps for each requirement.

## Sources

- [Directive (EU) 2023/2225, official text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023L2225), especially Articles 2, 10–11, 14, 18–19, 26, 31 and 48.
- [European Commission: consumer protection in financial services](https://commission.europa.eu/topics/consumers/consumer-rights-and-complaints/consumer-financial-products-and-services/consumer-protection-financial-services_en), including the application date.
- [European Commission: October 2026 infringement decisions](https://malta.representation.ec.europa.eu/news/october-infringements-package-key-decisions-2026-10-01_en), on incomplete notification of transposition by 16 member states at that date.
