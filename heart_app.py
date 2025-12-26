import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.big-title {
    font-size: 2.3rem;
    font-weight: 800;
}
.section {
    padding: 1.2rem;
    border-radius: 14px;
    background-color: #fafafa;
    margin-bottom: 1.2rem;
}
.prediction-box {
    padding: 1.6rem;
    border-radius: 14px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
}
.high {
    background-color: #ffebee;
    color: #b71c1c;
}
.low {
    background-color: #e8f5e9;
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    model_bundle = joblib.load("heart_xgboost_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    return model_bundle["model"], model_bundle["features"], scaler

model, feature_names, scaler = load_model()

# =========================
# Header
# =========================
st.markdown('<div class="big-title">❤️ Heart Disease Risk Assessment</div>', unsafe_allow_html=True)
st.markdown("""
This application estimates **your likelihood of heart disease** based on common clinical measurements.

🩺 **Not a diagnosis** — educational & early-risk screening only.
""")

st.markdown("---")

# =========================
# Step 1: Personal Info
# =========================
st.markdown("## Step 1: Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 100, 50)
    sex = st.radio("Biological Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0

with col2:
    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina (highest risk)",
            "Atypical Angina",
            "Non-anginal Pain",
            "No Symptoms"
        ]
    )
    cp = [
        "Typical Angina (highest risk)",
        "Atypical Angina",
        "Non-anginal Pain",
        "No Symptoms"
    ].index(cp_label)

st.markdown("---")

# =========================
# Step 2: Medical Data
# =========================
st.markdown("## Step 2: Heart & Medical Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

with col2:
    restecg_label = st.selectbox(
        "Resting ECG",
        ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"]
    )
    restecg = ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"].index(restecg_label)

    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

with col3:
    exang = st.radio("Exercise-induced chest pain", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)

    slope_label = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope_label)

st.markdown("---")

# =========================
# Step 3: Advanced Findings
# =========================
st.markdown("## Step 3: Diagnostic Findings")

col1, col2 = st.columns(2)

with col1:
    ca = st.selectbox("Major vessels affected (0–3)", [0, 1, 2, 3])

with col2:
    thal_label = st.selectbox(
        "Thalassemia Test Result",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )
    thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal_label) + 1

st.markdown("---")

# =========================
# Prediction
# =========================
if st.button("Assess Heart Disease Risk", use_container_width=True):

    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([input_data])[feature_names]
    input_scaled = scaler.transform(input_df)

    # 🔥 PROBABILITY-BASED DECISION
    proba = model.predict_proba(input_scaled)[0][1]
    threshold = 0.40  # medical-risk optimized

    is_high_risk = proba >= threshold

    st.markdown("## Assessment Result")

    if is_high_risk:
        st.markdown(
            f'<div class="prediction-box high">⚠️ HIGH RISK<br>{proba*100:.1f}% probability</div>',
            unsafe_allow_html=True
        )
        st.error("Please consult a cardiologist for further evaluation.")
    else:
        st.markdown(
            f'<div class="prediction-box low">✅ LOW RISK<br>{proba*100:.1f}% probability</div>',
            unsafe_allow_html=True
        )
        st.success("Maintain healthy habits and regular check-ups.")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        title={"text": "Heart Disease Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#c8e6c9"},
                {"range": [40, 70], "color": "#fff9c4"},
                {"range": [70, 100], "color": "#ffcdd2"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Entered Data"):
        st.json(input_data)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("About This App")
    st.markdown("""
- **Model:** XGBoost (tuned)
- **Threshold:** 0.40 (medical risk optimized)
- **Dataset:** UCI Heart Disease
- **Purpose:** Early risk awareness
    """)

    st.header("Medical Disclaimer")
    st.caption("This tool does NOT provide medical diagnosis.")
