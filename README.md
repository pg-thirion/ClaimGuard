# ClaimGuard

Explainable insurance risk triage for synthetic short-term insurance claims. Python, scikit-learn, Streamlit and SQLite.

## Run

Python 3.11 or 3.12:
```bash
python -m venv .venv
# Activate your environment, then:
python -m pip install -r requirements.txt
streamlit run app.py
python -m pytest -v
```

Default storage is session-local SQLite. For a trusted single-user local installation only, set the environment variable CLAIMGUARD_DB to a private database path before starting. Never configure shared persistence on a public deployment. The demo has no authentication. Never enter real or personal data.

## Workspaces

| Workspace | Capabilities |
| --- | --- |
| Portfolio | KPIs, monthly activity, category exposure, risk distribution, risk/anomaly scatter, export |
| Queue | Either/both/individual signals, amount and status filters, category search, sorting, watchlist, CSV |
| Inspector | Exact logistic contributions, shared fictional policy/provider links, scenario comparison, JSON case packets |
| Cases | Analyst aliases, status, tags, evidence checklist, notes, watchlist, append-only events, hash verification, CSV |
| Evaluation | Logistic/random forest/prevalence benchmark, Brier score, calibration, capacity/lift, category slices, confusion matrix |
| Planner | Validation-only cost-sensitive threshold search and review workload |
| Drift | Portfolio stress tests, PSI and feature mean comparisons |
| Governance | Logistic fingerprint, split counts, feature dictionary, coefficients, usage limitations |

## Method

4,000 simulated claims in ZAR, with overlapping conditional distributions for a latent synthetic outcome. Fictional policy/provider identifiers and dates are generated for display and linking only. They are not features or evidence of fraud. Stratified 60/20/20 training, validation and test split. Fit scaling, logistic regression and Isolation Forest only on training; choose cost-sensitive thresholds on validation. Random forest and prevalence benchmarks are reported on test, not used to select the operational model. Risk score is a synthetic outcome estimate; anomaly percentile ranks unusualness relative to training. Exact local log-odds contributions are non-causal. PSI uses smoothed quantile bins; illustrative stress scenarios are not counterfactual causal claims.

## Safety and limitations

Educational research only. Never deny, delay or price real claims based on these scores. No real data or external API calls. Category analysis is not demographic fairness testing. Model fingerprint covers logistic coefficients, intercept and scaler, not an entire reproducibility manifest. Hash-chained events detect edits but can be rewritten by someone with database access and are not externally anchored. The journal stores latest full case snapshots as events. No identity, access controls or encrypted storage. A public deployment must remain synthetic and session-local. Real deployment requires representative temporal and out-of-sample evaluation, calibration, proxy and fairness testing, POPIA and privacy assessment, secure infrastructure and independent human oversight.

## Tests

Unit tests cover split isolation, schema validation, explanation fidelity, benchmark, metrics, calibration, capacity, drift, stress, queue, CSV injection and SQLite persistence/integrity. Streamlit AppTest covers all workspaces and sample case/queue interactions. GitHub Actions runs on Python 3.11 and 3.12.
