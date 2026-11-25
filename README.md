# ClaimGuard

Explainable insurance claim risk triage for a synthetic short-term insurance portfolio.

ClaimGuard combines supervised logistic regression with an independent Isolation Forest anomaly detector. Investigators can rank claims, inspect exact model contributions, evaluate threshold trade-offs and record review outcomes. All data is generated locally; no customer data, external APIs or API keys are required.

## Run locally

Python 3.11 or 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests with `python -m pytest -q`.

## Architecture

- `claimguard.py`: reproducible synthetic data generation, feature engineering, training, held-out evaluation and claim-level explanations.
- `app.py`: investigation queue, individual claim inspector, model evaluation and session-only review log.
- `tests/test_claimguard.py`: data, leakage, prediction, explanation and evaluation tests.

The generator creates fictional motor, property and household claims in South African rand. A latent simulated outcome influences overlapping distributions rather than a deterministic fraud rule. Claim identifiers, category and outcome are excluded from model inputs. Amount-to-cover ratio, policy age, prior claims, reporting delay, repeated invoice references and document gaps are illustrative indicators, not evidence of fraud.

Training and test samples are stratified and disjoint. Standardisation and anomaly reference distributions are fitted only on the training data. Logistic regression outputs a synthetic outcome probability; Isolation Forest outputs a separate percentile relative to training anomalies. Local explanations are signed contributions to the logistic log-odds, including the baseline intercept. They are exact for this model, not causal explanations.

Evaluation reports ROC-AUC, average precision, precision, recall, F1, false positives and false negatives at the selected threshold. Threshold selection on the displayed test set is exploratory and would require a separate validation set before deployment. No performance claims about real insurance portfolios are made.

## Responsible use

This is an educational portfolio prototype, not a production fraud model. Never use its scores to deny, delay or price real claims. No personal or protected attributes are generated or processed. This alone does not establish fairness. Real deployment requires lawful data processing, POPIA assessment, representative validation, calibration, subgroup and proxy analysis, independent governance, and meaningful human review. Synthetic labels are not real fraud evidence.

Review decisions are stored in the current browser session only. Exports contain synthetic data and review outcomes. No authentication or persistent audit store is included.

## Deployment

Run locally or connect this repository to Streamlit Community Cloud with `app.py` as the entrypoint. No live deployment is included.
