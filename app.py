import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="ReopenRisk | Ticket Triage AI", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("models/reopen_pipeline.pkl")


bundle = load_bundle()
preprocessor = bundle["preprocessor"]
model = bundle["model"]
threshold = bundle["threshold"]
feature_names = [name.split("__")[-1] for name in bundle["feature_names"]]

st.title("ReopenRisk: Support Ticket Reopen Predictor")
st.markdown(
    "Predicts whether a resolved ticket will reopen and explains the exact "
    "drivers using **SHAP**."
)

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Ticket Closure Parameters")
    category = st.selectbox(
        "Category",
        ["Technical", "Billing", "Account Access", "API Integration"],
    )
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
    first_response_mins = st.slider("First Response Time (mins)", 1, 300, 45)
    resolution_hours = st.slider(
        "Total Resolution Time (hours)",
        0.5,
        72.0,
        8.0,
    )
    num_messages = st.slider("Back-and-Forth Messages", 2, 30, 8)
    escalation_count = st.selectbox("Escalations", [0, 1, 2, 3], index=1)
    resolution_note_words = st.slider(
        "Closing Note Length (words)", 5, 200, 35
    )
    past_customer_tickets = st.number_input(
        "Customer Past Tickets", min_value=0, max_value=20, value=2
    )

input_df = pd.DataFrame(
    [
        {
            "first_response_mins": first_response_mins,
            "resolution_hours": resolution_hours,
            "num_messages": num_messages,
            "escalation_count": escalation_count,
            "resolution_note_words": resolution_note_words,
            "past_customer_tickets": past_customer_tickets,
            "category": category,
            "priority": priority,
        }
    ]
)

X_proc = preprocessor.transform(input_df)
prob = float(model.predict_proba(X_proc)[0, 1])
is_high_risk = prob >= threshold

with col2:
    st.subheader("Risk Assessment & SHAP Explanation")
    m1, m2 = st.columns(2)
    m1.metric("Reopen Probability", f"{prob:.1%}")
    decision = (
        "HIGH RISK — Hold & QA" if is_high_risk else "SAFE TO CLOSE"
    )
    m2.metric("Triage Decision", decision)

    explainer = shap.TreeExplainer(model)
    shap_exp = explainer(X_proc)
    shap_exp.feature_names = feature_names

    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap.plots.waterfall(shap_exp[0], max_display=8, show=False)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)
