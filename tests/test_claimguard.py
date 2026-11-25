import numpy as np
import pandas as pd
import pytest
from claimguard import generate_claims, features, train, FEATURES

@pytest.fixture(scope="module")
def model():
    return train(2000, 42)

def test_reproducible_data():
    a = generate_claims(500, 12)
    pd.testing.assert_frame_equal(a, generate_claims(500, 12))
    assert a.claim_id.is_unique
    assert a.claim_amount.between(0, a.cover_limit).all()
    assert 0 < a.synthetic_outcome.mean() < .25

def test_no_label_leakage():
    data = generate_claims(500)
    x = features(data)
    assert list(x.columns) == list(FEATURES)
    pd.testing.assert_frame_equal(x, features(data.assign(synthetic_outcome=1-data.synthetic_outcome)))

def test_train_test_disjoint(model):
    assert not model.train_ids.intersection(model.test.claim_id)

def test_scores(model):
    result = model.score(model.test)
    assert "synthetic_outcome" not in result
    assert result.risk.between(0,1).all()
    assert result.anomaly_percentile.between(0,1).all()

def test_explanation_exact(model):
    sample = model.test.iloc[[0]]
    intercept, detail = model.explain(sample)
    logit = intercept + detail.log_odds_contribution.sum()
    probability = 1 / (1 + np.exp(-logit))
    assert probability == pytest.approx(model.score(sample).risk.iloc[0])

def test_evaluation(model):
    metrics = model.evaluate(.5)
    assert sum(metrics[key] for key in ["tn","fp","fn","tp"]) == len(model.test)
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["roc_auc"] > .5
    assert model.evaluate(0)["fn"] == 0
    assert model.evaluate(1)["fp"] == 0

def test_invalid_input():
    sample = generate_claims(200)
    with pytest.raises(ValueError):
        features(sample.assign(cover_limit=0))
    with pytest.raises(ValueError):
        generate_claims(10)
