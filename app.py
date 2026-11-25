"""Streamlit investigation console for simulated claims."""
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from claimguard import train

st.set_page_config(page_title="ClaimGuard | Risk intelligence", page_icon=":material/shield:", layout="wide")

@st.cache_resource(show_spinner="Training synthetic research models...")
def load_model():
    return train()

model = load_model()
scored = model.score(model.test).sort_values("risk", ascending=False)
if "reviews" not in st.session_state:
    st.session_state.reviews = {}

st.sidebar.title("ClaimGuard")
st.sidebar.caption("Insurance risk intelligence / Research")
page = st.sidebar.radio("Workspace", ["Investigation queue", "Claim inspector", "Model evaluation", "Review log"])
threshold = st.sidebar.slider("Review threshold", .05, .95, .50, .05)
st.sidebar.caption("Threshold applies to model risk only. Anomaly percentile is independent.")
st.sidebar.warning("Synthetic data only. Human review required. No automated claim decisions.")
flagged = scored[scored.risk >= threshold]

if page == "Investigation queue":
    st.title("Investigation queue")
    st.caption("Held-out synthetic claims | Ranked by model-estimated risk")
    a, b, c, d = st.columns(4)
    a.metric("Claims scored", f"{len(scored):,}")
    b.metric("Flagged for review", f"{len(flagged):,}")
    c.metric("Flagged claim value", f"R {flagged.claim_amount.sum():,.0f}")
    d.metric("Reviewed this session", len(st.session_state.reviews))
    left, right = st.columns(2)
    with left:
        st.subheader("Risk distribution")
        buckets = pd.cut(scored.risk, bins=[0,.2,.4,.6,.8,1], include_lowest=True, labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
        st.bar_chart(buckets.value_counts(sort=False))
    with right:
        st.subheader("Flagged claims by category")
        st.bar_chart(flagged.category.value_counts().reindex(["Motor","Property","Household"], fill_value=0))
    a, b, c = st.columns(3)
    cats = a.multiselect("Claim category", sorted(scored.category.unique()))
    query = b.text_input("Search claim ID")
    only = c.checkbox("Flagged only", value=True)
    view = flagged if only else scored
    if cats:
        view = view[view.category.isin(cats)]
    if query:
        view = view[view.claim_id.str.contains(query, case=False, regex=False)]
    view = view.assign(review_status=view.claim_id.map(lambda x: st.session_state.reviews.get(x, {}).get("status", "Pending")))
    display = view[["claim_id","category","claim_amount","risk","anomaly_percentile","review_status"]]
    st.dataframe(display, hide_index=True, use_container_width=True, column_config={
        "claim_amount": st.column_config.NumberColumn("Claim value (ZAR)", format="R %.2f"),
        "risk": st.column_config.ProgressColumn("Model risk", min_value=0, max_value=1, format="%.2f"),
        "anomaly_percentile": st.column_config.ProgressColumn("Anomaly percentile", min_value=0, max_value=1, format="%.2f"),
    })
    st.download_button("Export filtered synthetic queue", view.to_csv(index=False), "claimguard-queue.csv", "text/csv")
elif page == "Claim inspector":
    st.title("Claim inspector")
    selected = st.selectbox("Claim ID (highest model risk first)", scored.claim_id.tolist())
    row = scored[scored.claim_id == selected].iloc[0]
    original = model.test[model.test.claim_id == selected]
    baseline, explanation = model.explain(original)
    a, b, c = st.columns(3)
    a.metric("Model risk", f"{row.risk:.1%}")
    b.metric("Anomaly percentile", f"{row.anomaly_percentile:.1%}")
    c.metric("Claim value", f"R {row.claim_amount:,.2f}")
    st.write(f"Category: **{row.category}** | Cover limit: **R {row.cover_limit:,.2f}**")
    if row.risk >= threshold:
        st.warning("Above review threshold. Investigate supporting evidence; this is not a finding of fraud.")
    else:
        st.info("Below review threshold. A low score does not establish that a claim is legitimate.")
    st.subheader("Why this model score?")
    st.caption(f"Baseline log-odds: {baseline:.3f}. Positive contributions increase model risk; negative contributions reduce it. Contributions are not causal.")
    st.bar_chart(explanation.set_index("indicator")["log_odds_contribution"], horizontal=True)
    st.dataframe(explanation.round(3), hide_index=True, use_container_width=True)
    st.subheader("Human review")
    previous = st.session_state.reviews.get(selected, {})
    choices = ["Pending", "Request supporting evidence", "Escalate for manual review", "Close simulated review"]
    with st.form("review_" + selected):
        status = st.selectbox("Review status", choices, index=choices.index(previous.get("status","Pending")))
        notes = st.text_area("Synthetic investigation notes", value=previous.get("notes",""), max_chars=2000, help="Do not enter personal information.")
        if st.form_submit_button("Record review"):
            st.session_state.reviews[selected] = {"status": status, "notes": notes, "recorded_at": datetime.now(timezone.utc).isoformat()}
            st.success("Review recorded in this session.")
elif page == "Model evaluation":
    st.title("Model evaluation")
    st.caption("Held-out synthetic test set. Metrics do not imply real-world performance.")
    metrics = model.evaluate(threshold)
    columns = st.columns(5)
    for column, key, title in zip(columns, ["roc_auc","average_precision","precision","recall","f1"], ["ROC-AUC","Average precision","Precision","Recall","F1"]):
        column.metric(title, f"{metrics[key]:.3f}")
    st.write(f"Synthetic outcome prevalence: **{metrics['prevalence']:.1%}**. Threshold: **{threshold:.0%}**.")
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame([[metrics["tn"],metrics["fp"]],[metrics["fn"],metrics["tp"]]], index=["Simulated negative","Simulated positive"],columns=["Not flagged","Flagged"]), use_container_width=True)
    st.subheader("Precision and recall across thresholds")
    curve = pd.DataFrame([{"threshold": t, **model.evaluate(t)} for t in [.05,.10,.20,.30,.40,.50,.60,.70,.80,.90,.95]])
    st.line_chart(curve.set_index("threshold")[["precision","recall"]])
    st.info("Changing the threshold here is exploratory test-set analysis. A real system requires a separate validation set, calibration, fairness testing and governance.")
    st.write("Isolation Forest is an independent anomaly signal, not a calibrated fraud probability. Its percentile is relative to the training population.")
else:
    st.title("Review log")
    st.caption("Latest review per synthetic claim. Session-only storage, not an immutable audit trail.")
    log = pd.DataFrame([{"claim_id": key, **value} for key, value in st.session_state.reviews.items()])
    if log.empty:
        st.info("No reviews recorded. Open a claim in the inspector to record one.")
    else:
        st.dataframe(log, hide_index=True, use_container_width=True)
        # Guard spreadsheet applications against formula interpretation in user-entered notes.
        export = log.copy()
        export["notes"] = export.notes.map(lambda s: "'" + s if s.lstrip().startswith(("=", "+", "-", "@")) or s.startswith(("\t","\r","\n")) else s)
        st.download_button("Export review log", export.to_csv(index=False), "claimguard-reviews.csv", "text/csv")
