"""Reproducible evaluation and triage utilities for synthetic data only."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score)
from claimguard import FEATURES, features

def metrics(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all() or not 0 <= threshold <= 1 or not np.isin(y,[0,1]).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError("Invalid evaluation inputs.")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0,1]).ravel()
    return dict(n=len(y), roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if y.sum() else None,
        brier=float(brier_score_loss(y,p)), precision=float(precision_score(y,p>=threshold,zero_division=0)),
        recall=float(recall_score(y,p>=threshold,zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def threshold_plan(model, false_positive_cost=1., false_negative_cost=5.):
    if not np.isfinite([false_positive_cost,false_negative_cost]).all() or min(false_positive_cost,false_negative_cost)<0 or false_positive_cost+false_negative_cost==0:
        raise ValueError("Use non-negative, finite costs, not both zero.")
    y=model.validation.synthetic_outcome
    p=model.score(model.validation).risk
    rows=[]
    for t in np.linspace(.01,.99,99):
        m=metrics(y,p,t)
        rows.append({"threshold":float(t), **m, "review_count":m["tp"]+m["fp"],
            "illustrative_cost":m["fp"]*false_positive_cost+m["fn"]*false_negative_cost})
    frame=pd.DataFrame(rows)
    best=frame.sort_values(["illustrative_cost","threshold"],ascending=[True,False]).iloc[0]
    return float(best.threshold),frame

def benchmark(model):
    x=features(model.training)
    rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=8,random_state=42,n_jobs=1)
    rf.fit(x,model.training.synthetic_outcome)
    y=model.test.synthetic_outcome
    return pd.DataFrame([
        {"model":"Logistic regression",**metrics(y,model.score(model.test).risk)},
        {"model":"Random forest (benchmark)",**metrics(y,rf.predict_proba(features(model.test))[:,1])},
        {"model":"Prevalence baseline",**metrics(y,np.repeat(model.training.synthetic_outcome.mean(),len(y)))}
    ])

def calibration(y,p,bins=10):
    if bins<2: raise ValueError("At least two bins required.")
    frame=pd.DataFrame({"observed":np.asarray(y),"predicted":np.asarray(p)})
    frame["bin"]=pd.cut(frame.predicted,np.linspace(0,1,bins+1),include_lowest=True)
    return frame.groupby("bin",observed=True).agg(count=("observed","size"),predicted=("predicted","mean"),observed=("observed","mean")).reset_index(drop=True)

def capacity_curve(y,p):
    ranked=pd.DataFrame({"y":np.asarray(y),"p":np.asarray(p)}).sort_values("p",ascending=False,kind="stable")
    total=ranked.y.sum()
    prevalence=ranked.y.mean()
    rows=[]
    for share in [.01,.05,.10,.20,.30,.50,1.]:
        k=max(1,int(np.ceil(len(ranked)*share)))
        captured=int(ranked.head(k).y.sum())
        rows.append(dict(capacity=share,reviewed=k,captured=captured,recall=captured/total if total else 0,
                         precision=captured/k,lift=captured/k/prevalence if prevalence else 0))
    return pd.DataFrame(rows)

def segments(model, threshold):
    data=model.test.assign(risk=model.score(model.test).risk)
    return pd.DataFrame([{"category":k,**metrics(g.synthetic_outcome,g.risk,threshold)}
                         for k,g in data.groupby("category")])

def psi(reference,current):
    a,b=np.asarray(reference,dtype=float),np.asarray(current,dtype=float)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite non-empty samples required.")
    edges=np.unique(np.quantile(a,np.linspace(0,1,11)))[1:-1]
    if not len(edges):
        edges=np.array([a[0]]) if np.all(a==a[0]) else np.array([np.median(a)])
    edges=np.r_[-np.inf,edges,np.inf]
    def proportions(v):
        counts=np.histogram(v,bins=edges)[0]+.5
        return counts/counts.sum()
    x,y=proportions(a),proportions(b)
    return float(np.sum((y-x)*np.log(y/x)))

def drift(model, current):
    a,b=features(model.training),features(current)
    return pd.DataFrame([dict(feature=FEATURES[k],psi=psi(a[k],b[k]),training_mean=a[k].mean(),current_mean=b[k].mean()) for k in FEATURES])

def stress(model, amount_multiplier=1., extra_delay=0):
    if not 1 <= amount_multiplier <= 3 or not 0 <= extra_delay <= 30 or extra_delay%1:
        raise ValueError("Invalid stress parameters.")
    data=model.test.copy()
    before=model.score(data).risk
    data["claim_amount"]=np.minimum(data.cover_limit,data.claim_amount*amount_multiplier)
    data["report_delay_days"]+=extra_delay
    after=model.score(data).risk
    return data, pd.DataFrame({"baseline":before,"scenario":after,"change":after-before})

def queue(scored,threshold=.5,mode="Either signal",anomaly=.95,minimum=0,categories=(),query="",sort="risk"):
    if mode not in ["Model risk","Anomaly","Either signal","Both signals","All claims"] or sort not in ["risk","anomaly_percentile","claim_amount","reported_date"]:
        raise ValueError("Invalid queue mode or sort.")
    r,a=scored.risk>=threshold,scored.anomaly_percentile>=anomaly
    masks={"Model risk":r,"Anomaly":a,"Either signal":r|a,"Both signals":r&a,"All claims":pd.Series(True,index=scored.index)}
    result=scored[masks[mode] & (scored.claim_amount>=minimum)].copy()
    if categories: result=result[result.category.isin(categories)]
    if query: result=result[result.claim_id.str.contains(query,case=False,regex=False)]
    result["priority"]=np.select([(result.risk>=threshold)&(result.anomaly_percentile>=anomaly),result.risk>=threshold,result.anomaly_percentile>=anomaly],["Both signals","Model risk","Anomaly"],default="Routine")
    return result.sort_values(sort,ascending=False,kind="stable")

def related(scored,claim):
    match=(scored.policy_id==claim.policy_id)|(scored.provider_id==claim.provider_id)
    result=scored[match & (scored.claim_id!=claim.claim_id)].copy()
    result["relationship"]=np.where(result.policy_id==claim.policy_id,"Shared synthetic policy","Shared synthetic provider")
    return result.sort_values("risk",ascending=False)

def safe_csv(frame):
    def clean(value):
        if isinstance(value,str) and (value.lstrip().startswith(("=","+","-","@")) or value.startswith(("\t","\r","\n"))):
            return "'"+value
        return value
    return frame.apply(lambda col: col.map(clean)).to_csv(index=False)

def model_id(model):
    values=np.r_[model.classifier.intercept_,model.classifier.coef_.ravel(),model.scaler.mean_,model.scaler.scale_]
    return hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()[:16]
