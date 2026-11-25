import io
import numpy as np
import pandas as pd
import pytest
from claimguard import train, features, generate_claims
from analytics import *
from cases import CaseStore

@pytest.fixture(scope="module")
def model():
    return train(1200,42)

def test_three_disjoint_splits(model):
    a,b,c=map(lambda f:set(f.claim_id),[model.training,model.validation,model.test])
    assert not (a&b or a&c or b&c)
    assert len(a|b|c)==1200

@pytest.mark.parametrize("field,value",[("invoice_reuse",2),("document_gaps",5),("prior_claims",.5),("claim_amount",1e20),("report_delay_days",np.inf)])
def test_schema(field,value):
    with pytest.raises(ValueError):
        features(generate_claims(200).assign(**{field:value}))

def test_model_identity(model):
    assert model_id(model)==model_id(model)
    assert len(model_id(model))==16

def test_benchmark(model):
    table=benchmark(model)
    assert len(table)==3 and table.roc_auc.between(0,1).all()
    assert set(table.n)=={len(model.test)}

def test_threshold_validation(model):
    best,table=threshold_plan(model,1,5)
    assert set(table.n)=={len(model.validation)}
    assert table.loc[table.threshold==best,"illustrative_cost"].iloc[0]==table.illustrative_cost.min()
    with pytest.raises(ValueError): threshold_plan(model,0,0)

def test_calibration(model):
    y=model.test.synthetic_outcome
    table=calibration(y,model.score(model.test).risk)
    assert table["count"].sum()==len(y)
    assert table.observed.between(0,1).all()

def test_capacity(model):
    table=capacity_curve(model.test.synthetic_outcome,model.score(model.test).risk)
    assert table.recall.is_monotonic_increasing
    assert table.iloc[-1].recall==1
    assert table.iloc[-1].lift==pytest.approx(1)

def test_segments(model):
    assert segments(model,.5).n.sum()==len(model.test)

def test_metrics_single_class():
    m=metrics([0,0],[.1,.8])
    assert m["roc_auc"] is None and m["fp"]==1

def test_psi():
    x=np.arange(100)
    assert psi(x,x)==pytest.approx(0)
    assert psi(x,x+100)>0

def test_stress_and_drift(model):
    data,base=stress(model)
    assert np.allclose(base.change,0)
    data,shift=stress(model,2,5)
    assert (data.claim_amount<=data.cover_limit).all()
    assert shift.change.mean()>0
    assert drift(model,data).psi.ge(0).all()

def test_queue_modes(model):
    scored=model.score(model.test)
    a=queue(scored,.5,"Model risk")
    b=queue(scored,.5,"Anomaly")
    both=queue(scored,.5,"Both signals")
    either=queue(scored,.5,"Either signal")
    assert set(both.claim_id)==set(a.claim_id)&set(b.claim_id)
    assert set(either.claim_id)==set(a.claim_id)|set(b.claim_id)
    assert queue(scored,mode="All claims",query="[invalid").empty
    assert queue(scored,mode="All claims",categories=["Motor"]).category.eq("Motor").all()
    assert queue(scored,mode="All claims",minimum=10000).claim_amount.ge(10000).all()

def test_related(model):
    scored=model.score(model.test)
    row=scored.iloc[0]
    links=related(scored,row)
    assert row.claim_id not in set(links.claim_id)
    assert ((links.policy_id==row.policy_id)|(links.provider_id==row.provider_id)).all()

@pytest.mark.parametrize("value",["=1+1"," +1","-1","@SUM(A1)","\tX"])
def test_csv_formula(value):
    result=pd.read_csv(io.StringIO(safe_csv(pd.DataFrame({"notes":[value]}))))
    assert result.notes.iloc[0].startswith("'")

def test_case_persistence_and_history(tmp_path):
    path=tmp_path/"cases.db"
    s=CaseStore(path)
    s.record("CG-000001",notes="first",watch=True,tags=["Documentation"],checklist=["Policy verified"])
    s.record("CG-000001",status="Escalated",assignee="Analyst A",notes="second")
    assert s.verify() and len(s.history("CG-000001"))==2
    s.close()
    s=CaseStore(path)
    assert s.latest()["CG-000001"]["status"]=="Escalated"
    assert s.verify()
    s.db.execute("UPDATE events SET payload='{}' WHERE id=1")
    s.db.commit()
    assert not s.verify()
    s.close()

def test_case_rejects_invalid():
    s=CaseStore()
    with pytest.raises(ValueError): s.record("bad")
    with pytest.raises(ValueError): s.record("CG-000001",tags=["unknown"])
    with pytest.raises(ValueError): s.record("CG-000001",notes="x"*2001)
    s.close()
