import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Social Media & Student Mental Health", page_icon="🧠", layout="centered")

BASE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(BASE, "models")


@st.cache_resource
def load_all():
    meta = joblib.load(os.path.join(MD, "meta.joblib"))
    models = {n: joblib.load(os.path.join(MD, f + ".joblib")) for n, f in meta["file_names"].items()}
    return meta, models


meta, models = load_all()

st.title("🧠 Social Media Usage & Student Mental Health")
st.write("Advanced Data Science – FA2 Case Study. Enter a student's details and click **Predict** to estimate the "
         "mental health score (1 to 10, higher = better wellbeing).")
st.caption("This is a classroom project built on survey data. It shows patterns, not causes, and is not a medical tool.")

with st.sidebar:
    st.header("Model selection")
    choice = st.selectbox("Choose a model", ["Compare all models"] + list(models))

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", *meta["ranges"]["age"], value=21)
    gender = st.selectbox("Gender", ["Female", "Male"])
    level = st.selectbox("Academic level", ["High School", "Undergraduate", "Graduate"], index=1)
    country = st.selectbox("Country", meta["all_countries"], index=meta["all_countries"].index("India"))
    relation = st.selectbox("Relationship status", ["Single", "In Relationship", "Complicated"])
with c2:
    platform = st.selectbox("Most used platform", meta["all_platforms"], index=meta["all_platforms"].index("Instagram"))
    usage = st.slider("Daily social media use (hours)", 0.5, 10.0, 4.5, step=0.1)
    sleep = st.slider("Sleep per night (hours)", 3.0, 11.0, 7.0, step=0.1)
    conflicts = st.slider("Conflicts over social media (0-5)", 0, 5, 2)
    academic = st.selectbox("Does social media affect your academic performance?", ["No", "Yes"])


def build_row():
    return pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Academic_Level": level,
        "Avg_Daily_Usage_Hours": usage,
        "Affects_Academic_Performance": int(academic == "Yes"),
        "Sleep_Hours_Per_Night": sleep,
        "Relationship_Status": relation,
        "Conflicts_Over_Social_Media": conflicts,
        "Grouped_country": country if country in meta["top_countries"] else "Other",
        "Grouped_platform": platform if platform in meta["top_platforms"] else "Other",
    }])


def label(score):
    return "Good wellbeing 🟢" if score >= 7 else ("Moderate 🟡" if score >= 5.5 else "Needs attention 🔴")


if st.button("Predict", type="primary"):
    row = build_row()
    names = list(models) if choice == "Compare all models" else [choice]
    preds = {n: float(min(10, max(1, models[n].predict(row)[0]))) for n in names}

    if len(names) == 1:
        s = preds[names[0]]
        st.metric(f"Predicted mental health score ({names[0]})", f"{s:.1f} / 10")
        st.progress(s / 10)
        st.write("**Status:**", label(s))
    else:
        avg = sum(preds.values()) / len(preds)
        st.subheader("Predictions from all models")
        st.table(pd.DataFrame({"Predicted score": {n: f"{v:.2f}" for n, v in preds.items()}}))
        st.metric("Average of all models", f"{avg:.1f} / 10")
        st.progress(avg / 10)
        st.write("**Status:**", label(avg))

    with st.expander("Wellness tips"):
        if sleep < 7:
            st.write("• Aim for 7-8 hours of sleep; shorter sleep went with lower scores in this data.")
        if usage > 5:
            st.write("• Try to reduce daily social media time, for example with app timers or phone-free hours.")
        if conflicts >= 3:
            st.write("• Frequent conflicts over social media may be a sign to set boundaries or talk to someone you trust.")
        st.write("• If you feel low or stressed for long periods, please speak to a counsellor or a trusted person.")
