"""Synthetic-only insurance risk research. No real claim decisions."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "amount_ratio": "Claim amount / cover limit",
    "policy_age_days": "Policy age (days)",
    "prior_claims": "Previous claims (12 months)",
    "report_delay_days": "Reporting delay (days)",
    "invoice_reuse": "Repeated invoice reference",
    "document_gaps": "Missing supporting documents",
}


def generate_claims(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    if not 200 <= n <= 50000:
        raise ValueError("Use between 200 and 50,000 synthetic claims.")
    rng = np.random.default_rng(seed)
    outcome = rng.binomial(1, 0.10, n)
    category = rng.choice(["Motor", "Property", "Household"], n, p=[.55, .25, .20])
    cover = np.round(np.exp(rng.normal(11.6, .85, n)), -2).clip(10000, 2000000)
    ratio = rng.beta(1.8 + 1.8 * outcome, 7.5 - 2.5 * outcome)
    # Overlapping distributions intentionally permit both false positives and negatives.
    age = np.maximum(1, rng.exponential(600 / (1 + 1.2 * outcome), n)).astype(int)
    prior = rng.poisson(.55 + 1.0 * outcome, n)
    delay = rng.poisson(3 + 3 * outcome, n)
    reuse = rng.binomial(1, .035 + .24 * outcome, n)
    gaps = rng.binomial(4, .12 + .19 * outcome, n)
    amount = np.minimum(cover, np.maximum(100, np.round(cover * ratio, 2)))
    return pd.DataFrame({
        "claim_id": [f"CG-{i+1:06d}" for i in range(n)],
        "category": category,
        "claim_amount": amount,
        "cover_limit": cover,
        "policy_age_days": age,
        "prior_claims": prior,
        "report_delay_days": delay,
        "invoice_reuse": reuse,
        "document_gaps": gaps,
        "synthetic_outcome": outcome,
    })


def features(claims: pd.DataFrame) -> pd.DataFrame:
    frame = claims.copy()
    if (frame["cover_limit"] <= 0).any():
        raise ValueError("Cover limit must be positive.")
    frame["amount_ratio"] = frame["claim_amount"] / frame["cover_limit"]
    x = frame[list(FEATURES)].astype(float)
    if not np.isfinite(x.to_numpy()).all() or (x < 0).any().any():
        raise ValueError("Invalid model inputs.")
    return x


@dataclass
class ClaimModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    anomaly: IsolationForest
    anomaly_reference: np.ndarray
    train_ids: frozenset
    test: pd.DataFrame

    def score(self, claims: pd.DataFrame) -> pd.DataFrame:
        x = self.scaler.transform(features(claims))
        risk = self.classifier.predict_proba(x)[:, 1]
        raw = -self.anomaly.score_samples(x)
        percentile = np.searchsorted(self.anomaly_reference, raw, side="right") / len(self.anomaly_reference)
        return claims.drop(columns=["synthetic_outcome"], errors="ignore").assign(
            risk=risk, anomaly_percentile=percentile
        )

    def explain(self, claim: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        if len(claim) != 1:
            raise ValueError("Explain exactly one claim.")
        raw = features(claim)
        z = self.scaler.transform(raw)[0]
        contributions = z * self.classifier.coef_[0]
        details = pd.DataFrame({
            "indicator": list(FEATURES.values()),
            "value": raw.iloc[0].to_numpy(),
            "training_mean": self.scaler.mean_,
            "log_odds_contribution": contributions,
        })
        details["absolute"] = np.abs(contributions)
        return float(self.classifier.intercept_[0]), details.sort_values("absolute", ascending=False).drop(columns="absolute")

    def evaluate(self, threshold: float = .5) -> dict:
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between zero and one.")
        y = self.test["synthetic_outcome"]
        scores = self.score(self.test)["risk"]
        predicted = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        return {
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "prevalence": float(y.mean()),
        }


def train(n: int = 4000, seed: int = 42) -> ClaimModel:
    data = generate_claims(n, seed)
    train_data, test = train_test_split(
        data, test_size=.25, random_state=seed, stratify=data["synthetic_outcome"]
    )
    scaler = StandardScaler()
    x = scaler.fit_transform(features(train_data))
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(x, train_data["synthetic_outcome"])
    anomaly = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1)
    anomaly.fit(x)
    reference = np.sort(-anomaly.score_samples(x))
    return ClaimModel(scaler, classifier, anomaly, reference, frozenset(train_data["claim_id"]), test.reset_index(drop=True))
