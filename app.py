"""ClaimGuard synthetic investigation workspace."""
import json
import os
import pandas as pd
import streamlit as st
from claimguard import train
from analytics import (benchmark, calibration, capacity_curve, drift, metrics, model_id,
                       queue, related, safe_csv, segments, stress, threshold_plan)
from cases import CaseStore, STATUSES, TAGS, CHECKLIST

st.set_page_config(page_title="ClaimGuard | Risk intelligence",page_icon=":material/shield:",layout="wide")
PAGES=["Portfolio overview","Investigation queue","Claim inspector","Case journal","Model evaluation","Threshold planner","Drift & scenarios","Model governance"]

@st.cache_resource(show_spinner="Training synthetic models...")
def load_model():
    return train()

@st.cache_data
def benchmark_table(_model, version):
    return benchmark(_model)

@st.cache_data
def plan_table(_model, version, fp, fn):
    return threshold_plan(_model,fp,fn)

model=load_model()
version=model_id(model)
scored=model.score(model.test).sort_values("risk",ascending=False)
if "case_store" not in st.session_state:
    st.session_state.case_store=CaseStore(os.getenv("CLAIMGUARD_DB",":memory:"))
store=st.session_state.case_store
latest=store.latest()
st.sidebar.title("ClaimGuard")
st.sidebar.caption("Insurance risk intelligence | Synthetic lab")
page=st.sidebar.radio("Workspace",PAGES)
threshold=st.sidebar.slider("Review threshold",.05,.95,.50,.05)
anomaly=st.sidebar.slider("Anomaly threshold",.80,1.,.95,.01)
st.sidebar.warning("Synthetic claims only. No automatic decisions. Never enter personal data.")
st.sidebar.caption("Model "+version)
flagged=queue(scored,threshold,"Either signal",anomaly)

def money(value):
    return f"R {value:,.0f}"

def download(frame,name):
    st.download_button("Download "+name,safe_csv(frame),name+".csv","text/csv")

def overview():
    st.title("Portfolio overview")
    st.caption("Held-out synthetic claims | Independent model and anomaly signals")
    a,b,c,d=st.columns(4)
    a.metric("Claims",f"{len(scored):,}")
    b.metric("Review queue",len(flagged))
    c.metric("Flagged claim value",money(flagged.claim_amount.sum()))
    d.metric("Cases recorded",len(latest))
    a,b=st.columns(2)
    with a:
        st.subheader("Monthly claim activity")
        monthly=scored.assign(month=scored.reported_date.dt.strftime("%Y-%m"), flagged=scored.claim_id.isin(flagged.claim_id)).groupby("month").agg(claims=("claim_id","size"),flagged=("flagged","sum"))
        st.line_chart(monthly)
    with b:
        st.subheader("Portfolio exposure by category")
        st.bar_chart(scored.groupby("category").claim_amount.sum())
    a,b=st.columns(2)
    with a:
        st.subheader("Risk distribution")
        buckets=pd.cut(scored.risk,[0,.2,.4,.6,.8,1],include_lowest=True,labels=["0–20%","20–40%","40–60%","60–80%","80–100%"])
        st.bar_chart(buckets.value_counts(sort=False))
    with b:
        st.subheader("Risk vs anomaly")
        st.scatter_chart(scored,x="risk",y="anomaly_percentile",color="category")
    st.subheader("Category breakdown")
    frame=scored.assign(flagged=scored.claim_id.isin(flagged.claim_id)).groupby("category").agg(claims=("claim_id","size"),value=("claim_amount","sum"),flagged=("flagged","sum"),mean_risk=("risk","mean"))
    st.dataframe(frame)
    download(frame.reset_index(),"portfolio-summary")

def investigation_queue():
    st.title("Investigation queue")
    a,b,c=st.columns(3)
    mode=a.selectbox("Signal filter",["Either signal","Both signals","Model risk","Anomaly","All claims"])
    cats=b.multiselect("Claim categories",sorted(scored.category.unique()))
    sort=c.selectbox("Rank by",["risk","anomaly_percentile","claim_amount","reported_date"])
    a,b,c=st.columns(3)
    query=a.text_input("Find claim ID")
    minimum=b.number_input("Minimum claim value (ZAR)",min_value=0,value=0,step=1000)
    statuses=c.multiselect("Case status",STATUSES)
    watched=st.checkbox("Watchlist only")
    frame=queue(scored,threshold,mode,anomaly,minimum,cats,query,sort)
    frame["status"]=frame.claim_id.map(lambda k:latest.get(k,{}).get("status","Pending"))
    frame["watch"]=frame.claim_id.map(lambda k:latest.get(k,{}).get("watch",False))
    if statuses: frame=frame[frame.status.isin(statuses)]
    if watched: frame=frame[frame.watch]
    st.write(f"{len(frame):,} matching claims")
    st.dataframe(frame[["claim_id","category","reported_date","claim_amount","risk","anomaly_percentile","priority","status","watch"]],hide_index=True,column_config={
        "risk":st.column_config.ProgressColumn("Risk",min_value=0,max_value=1,format="%.3f"),
        "anomaly_percentile":st.column_config.ProgressColumn("Anomaly",min_value=0,max_value=1,format="%.3f")})
    if not frame.empty:
        st.selectbox("Select a claim",frame.claim_id.tolist(),key="queue_claim")
        if st.button("Open claim inspector"):
            st.session_state.selected_claim=st.session_state.queue_claim
            st.info("Selection saved. Open Claim inspector from the sidebar.")
    download(frame,"investigation-queue")

def inspector():
    st.title("Claim inspector")
    ids=scored.claim_id.tolist()
    default=st.session_state.get("selected_claim",ids[0])
    selected=st.selectbox("Claim ID",ids,index=ids.index(default) if default in ids else 0)
    row=scored.loc[scored.claim_id==selected].iloc[0]
    original=model.test.loc[model.test.claim_id==selected]
    intercept,explanation=model.explain(original)
    a,b,c=st.columns(3)
    a.metric("Model risk",f"{row.risk:.1%}")
    b.metric("Anomaly percentile",f"{row.anomaly_percentile:.1%}")
    c.metric("Claim amount",money(row.claim_amount))
    st.caption(f"{row.category} | {row.policy_id} | {row.provider_id} | Cover {money(row.cover_limit)}")
    st.info("Review indicators are not evidence of fraud. Shared synthetic identifiers are illustrative only.")
    t1,t2,t3,t4=st.tabs(["Explanation","Related claims","Scenario comparison","Case review"])
    with t1:
        st.caption(f"Baseline log-odds: {intercept:.3f}. Contributions are exact, additive log-odds, not causal effects.")
        st.bar_chart(explanation.set_index("indicator").log_odds_contribution,horizontal=True)
        st.dataframe(explanation.round(3),hide_index=True)
    with t2:
        links=related(scored,row)
        st.dataframe(links[["claim_id","relationship","category","claim_amount","risk"]],hide_index=True)
        download(links,"related-claims")
    with t3:
        st.caption("Hypothetical comparison only. Do not use it to optimise or alter real claims.")
        multiplier=st.slider("Claim value multiplier",1.,3.,1.,.1,key="single_multiplier")
        delay=st.slider("Extra reporting days",0,30,0,key="single_delay")
        changed=original.copy()
        changed["claim_amount"]=(changed.claim_amount*multiplier).clip(upper=changed.cover_limit)
        changed["report_delay_days"]+=delay
        new=float(model.score(changed).risk.iloc[0])
        st.metric("Hypothetical model risk",f"{new:.1%}",f"{new-row.risk:+.1%}")
    with t4:
        prev=latest.get(selected,{})
        with st.form("case_"+selected):
            status=st.selectbox("Case status",STATUSES,index=STATUSES.index(prev.get("status","Pending")))
            assignee=st.text_input("Synthetic analyst alias",value=prev.get("assignee",""),max_chars=80)
            notes=st.text_area("Investigation notes (fictional only)",value=prev.get("notes",""),max_chars=2000)
            tags=st.multiselect("Case tags",TAGS,default=prev.get("tags",[]))
            checks=st.multiselect("Evidence checklist",CHECKLIST,default=prev.get("checklist",[]))
            watch=st.checkbox("Add to watchlist",value=prev.get("watch",False))
            if st.form_submit_button("Save case event"):
                store.record(selected,status,assignee,notes,tags,checks,watch)
                st.success("Case event recorded.")
        hist=store.history(selected)
        if hist: st.dataframe(pd.DataFrame(hist),hide_index=True)
        packet={"model_id":version,"claim":json.loads(row.to_json(date_format="iso")),"baseline_log_odds":intercept,
                "explanation":explanation.to_dict("records"),"case_history":hist,"notice":"Synthetic research only"}
        st.download_button("Download case packet",json.dumps(packet,indent=2,allow_nan=False),"case-"+selected+".json","application/json")

def journal():
    st.title("Case journal")
    st.caption("Append-only application events. Hash chaining detects edits, but is not externally anchored or immutable.")
    a,b,c=st.columns(3)
    a.metric("Cases",len(latest))
    b.metric("Watchlisted",sum(x["watch"] for x in latest.values()))
    c.metric("Journal integrity","Verified" if store.verify() else "Failed")
    events=store.history()
    if not events:
        st.info("No case events yet. Record a review in Claim inspector.")
        return
    current=pd.DataFrame(latest.values())
    st.subheader("Current case states")
    st.bar_chart(current.status.value_counts())
    st.dataframe(current,hide_index=True)
    st.subheader("Full event history")
    frame=pd.DataFrame(events)
    st.dataframe(frame,hide_index=True)
    download(frame,"case-journal")

def evaluation():
    st.title("Model evaluation")
    st.caption("Untouched test split | Exploratory dashboard; no real-world performance claims")
    y,p=model.test.synthetic_outcome,model.score(model.test).risk
    m=metrics(y,p,threshold)
    cols=st.columns(5)
    for col,key,label in zip(cols,["roc_auc","average_precision","precision","recall","brier"],["ROC-AUC","Average precision","Precision","Recall","Brier score"]):
        col.metric(label,f"{m[key]:.3f}")
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame([[m["tn"],m["fp"]],[m["fn"],m["tp"]]],index=["Negative","Positive"],columns=["Not flagged","Flagged"]))
    st.subheader("Model benchmark")
    results=benchmark_table(model,version)
    st.dataframe(results,hide_index=True)
    download(results,"model-benchmark")
    st.subheader("Calibration")
    cal=calibration(y,p)
    st.line_chart(cal,x="predicted",y="observed")
    st.dataframe(cal,hide_index=True)
    st.subheader("Review capacity and lift")
    cap=capacity_curve(y,p)
    st.line_chart(cap,x="capacity",y=["recall","precision"])
    st.dataframe(cap,hide_index=True)
    st.subheader("Category-level performance")
    st.dataframe(segments(model,threshold),hide_index=True)

def planner():
    st.title("Threshold planner")
    st.caption("Threshold recommendation uses validation data only. Costs are hypothetical relative units.")
    a,b=st.columns(2)
    fp=a.number_input("False positive cost",min_value=.1,max_value=1000.,value=1.)
    fn=b.number_input("False negative cost",min_value=.1,max_value=1000.,value=5.)
    best,table=plan_table(model,version,fp,fn)
    row=table.loc[table.threshold==best].iloc[0]
    a,b,c=st.columns(3)
    a.metric("Validation threshold",f"{best:.0%}")
    b.metric("Expected validation reviews",int(row.review_count))
    c.metric("Illustrative cost",f"{row.illustrative_cost:.1f}")
    st.line_chart(table,x="threshold",y="illustrative_cost")
    st.line_chart(table,x="threshold",y=["precision","recall"])
    st.dataframe(table,hide_index=True)
    download(table,"threshold-analysis")

def scenarios():
    st.title("Drift & scenarios")
    st.caption("Stress tests compare the held-out distribution with an artificial scenario.")
    a,b=st.columns(2)
    multiplier=a.slider("Portfolio amount multiplier",1.,3.,1.,.1)
    delay=b.slider("Additional reporting delay",0,30,0)
    data,changes=stress(model,multiplier,delay)
    a,b,c=st.columns(3)
    a.metric("Baseline mean risk",f"{changes.baseline.mean():.1%}")
    b.metric("Scenario mean risk",f"{changes.scenario.mean():.1%}")
    c.metric("Additional flagged",int((changes.scenario>=threshold).sum()-(changes.baseline>=threshold).sum()))
    st.subheader("Population stability index (PSI)")
    table=drift(model,data)
    st.bar_chart(table.set_index("feature").psi)
    st.dataframe(table,hide_index=True)
    st.caption("PSI uses smoothed training quantile bins. It is a descriptive indicator, not a formal fairness or drift verdict.")
    download(table,"drift-analysis")

def governance():
    st.title("Model governance")
    st.write("Version: "+version)
    st.dataframe(pd.DataFrame({"split":["Training","Validation","Test"],"claims":[len(model.training),len(model.validation),len(model.test)],"positive_rate":[model.training.synthetic_outcome.mean(),model.validation.synthetic_outcome.mean(),model.test.synthetic_outcome.mean()]}),hide_index=True)
    st.subheader("Model coefficients")
    st.dataframe(pd.DataFrame({"feature":list(__import__("claimguard").FEATURES.values()),"coefficient":model.classifier.coef_[0]}),hide_index=True)
    st.subheader("Limitations and data controls")
    st.write("Synthetic-only research. No real claims, automatic rejection, protected attributes or external API calls. Category comparisons are not demographic fairness validation. Model ID fingerprints logistic parameters only. Random split is not a temporal deployment simulation.")
    st.write("Case persistence: "+("local database, single-user mode" if os.getenv("CLAIMGUARD_DB") else "session-only SQLite database"))
    st.write("Journal hashes are not externally anchored. Production needs identity, authorisation, encryption, privacy and retention controls, real-data validation and independent oversight.")
    download(pd.DataFrame({"feature":list(__import__("claimguard").FEATURES.keys()),"description":list(__import__("claimguard").FEATURES.values())}),"feature-dictionary")

dict(zip(PAGES,[overview,investigation_queue,inspector,journal,evaluation,planner,scenarios,governance]))[page]()
