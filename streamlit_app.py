"""
Streamlit UI for the student academic-risk classifier.

Loads the trained pipeline saved by train.py and predicts whether a
student is at risk of failing (final grade below 10/20), based purely on
demographic and behavioral inputs collected before any grades exist.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "student_risk_pipeline.joblib"

st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"No trained model found at {MODEL_PATH}. Run `python train.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


def get_feature_importances(pipeline) -> pd.Series | None:
    """Returns a Series of feature -> importance, or None if unavailable."""
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return None

    readable_names = [name.split("__", 1)[-1] for name in feature_names]
    return pd.Series(values, index=readable_names).sort_values(ascending=False)


st.title("🎓 Student Risk Predictor")
st.caption(
    "Predicts whether a student is at risk of failing (final grade below "
    "10/20) using only demographic and behavioral factors — no grades "
    "required, so at-risk students can be flagged before any exam happens."
)
st.markdown("---")

pipeline = load_model()

with st.form("student_form"):
    st.subheader("Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.number_input("Age", min_value=15, max_value=22, value=17)
    with col2:
        address = st.selectbox("Home address", ["U", "R"], help="Urban / Rural")
        famsize = st.selectbox("Family size", ["LE3", "GT3"], help="≤3 / >3 members")
        pstatus = st.selectbox("Parents' cohabitation", ["T", "A"], help="Together / Apart")
    with col3:
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        reason = st.selectbox(
            "Reason for choosing school", ["home", "reputation", "course", "other"]
        )
        traveltime = st.selectbox("Home-to-school travel time", [1, 2, 3, 4])

    st.subheader("Family background")
    col4, col5, col6 = st.columns(3)
    with col4:
        medu = st.selectbox("Mother's education", [0, 1, 2, 3, 4], index=2)
        mjob = st.selectbox(
            "Mother's job", ["teacher", "health", "services", "at_home", "other"]
        )
        famsup = st.selectbox("Family educational support", ["yes", "no"])
    with col5:
        fedu = st.selectbox("Father's education", [0, 1, 2, 3, 4], index=2)
        fjob = st.selectbox(
            "Father's job", ["teacher", "health", "services", "at_home", "other"]
        )
        paid = st.selectbox("Extra paid classes", ["yes", "no"], index=1)
    with col6:
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
        famrel = st.slider("Family relationship quality", 1, 5, 4)

    st.subheader("Academic & lifestyle")
    col7, col8, col9 = st.columns(3)
    with col7:
        studytime = st.selectbox("Weekly study time (1: <2h, 4: >10h)", [1, 2, 3, 4])
        failures = st.selectbox("Past class failures", [0, 1, 2, 3])
        schoolsup = st.selectbox("School educational support", ["yes", "no"], index=1)
    with col8:
        activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"], index=1)
    with col9:
        absences = st.number_input("Number of absences", min_value=0, max_value=93, value=4)
        freetime = st.slider("Free time after school", 1, 5, 3)
        goout = st.slider("Going out with friends", 1, 5, 3)

    col10, col11 = st.columns(2)
    with col10:
        dalc = st.slider("Workday alcohol consumption", 1, 5, 1)
        health = st.slider("Current health status", 1, 5, 4)
    with col11:
        walc = st.slider("Weekend alcohol consumption", 1, 5, 1)

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    input_data = pd.DataFrame([{
        "age": age, "Medu": medu, "Fedu": fedu, "traveltime": traveltime,
        "studytime": studytime, "failures": failures, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
        "health": health, "absences": absences,
        "school": school, "sex": sex, "address": address, "famsize": famsize,
        "Pstatus": pstatus, "Mjob": mjob, "Fjob": fjob, "reason": reason,
        "guardian": guardian, "schoolsup": schoolsup, "famsup": famsup,
        "paid": paid, "activities": activities, "nursery": nursery,
        "higher": higher, "internet": internet, "romantic": romantic,
    }])

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Student is **AT RISK** of failing")
    else:
        st.success("✅ Student is **NOT AT RISK**")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Risk Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred" if prediction == 1 else "darkgreen"},
            "steps": [
                {"range": [0, 33], "color": "#d4edda"},
                {"range": [33, 66], "color": "#fff3cd"},
                {"range": [66, 100], "color": "#f8d7da"},
            ],
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    importances = get_feature_importances(pipeline)
    if importances is not None:
        st.subheader("Top Factors Driving This Model's Predictions")
        top_features = importances.head(10).sort_values()
        fig_bar = go.Figure(go.Bar(
            x=top_features.values, y=top_features.index, orientation="h"
        ))
        fig_bar.update_layout(
            height=350, margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption(
    "Data: UCI Student Performance dataset (Cortez & Silva, 2008) · "
    "Built with Streamlit, scikit-learn, and MLflow"
)
