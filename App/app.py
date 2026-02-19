import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import os

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# =============================
# Load Model
# =============================
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "Model", "churn_model_production.pkl")
    return joblib.load(model_path)

model = load_model()

# =============================
# Header
# =============================
st.title("📊 Telco Customer Churn Intelligence Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Model ROC-AUC", "0.84")
col2.metric("Recall (Churn)", "0.75")
col3.metric("Accuracy", "0.75")

st.markdown("---")

# =============================
# Sidebar - Customer Input
# =============================
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
services_count = st.sidebar.slider("Number of Active Services", 0, 6, 2)

# =============================
# Feature Engineering
# =============================
clv_ratio = monthly_charges * tenure / (monthly_charges + 1)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract_length = contract_map[contract]
fiber_no_support = int(internet == "Fiber optic" and tech_support == "No")
low_service_loyal = int(tenure > 24 and services_count <= 2)
charge_per_service = monthly_charges / (services_count + 1)

input_data = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
    "Total Charges": monthly_charges * tenure,
    "Services Count": services_count,
    "CLV_Ratio": clv_ratio,
    "Contract_Length": contract_length,
    "Fiber_No_Support": fiber_no_support,
    "Low_Service_Loyal": low_service_loyal,
    "Charge_Per_Service": charge_per_service,
    "Contract": contract,
    "Internet Service": internet,
    "Tech Support": tech_support,
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "High Charges": int(monthly_charges > 80),
    "High Risk Customer": int(contract == "Month-to-month" and internet == "Fiber optic")
}

input_df = pd.DataFrame([input_data])

# =============================
# Session State
# =============================
if "proba" not in st.session_state:
    st.session_state.proba = None
    st.session_state.input_df = None

# =============================
# Prediction Button
# =============================
if st.sidebar.button("🔍 Predict Churn Risk"):
    st.session_state.input_df = input_df
    st.session_state.proba = model.predict_proba(input_df)[0][1]

# =============================
# Display Results
# =============================
if st.session_state.proba is not None:

    proba = st.session_state.proba

    # Risk Level
    if proba < 0.35:
        level = "Low"
        color = "#1f8f4e"
    elif proba < 0.65:
        level = "Medium"
        color = "#f39c12"
    else:
        level = "High"
        color = "#e74c3c"

    # Risk Badge
    st.markdown(
        f"""
        <div style="padding:20px;background-color:{color};
        color:white;border-radius:12px;text-align:center;font-size:24px;">
        {level.upper()} RISK CUSTOMER ({proba*100:.2f}%)
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba*100,
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 35], 'color': "green"},
                {'range': [35, 65], 'color': "yellow"},
                {'range': [65, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, width='stretch')

    # Confidence
    st.subheader("Model Confidence")
    confidence_score = abs(proba - 0.5) * 2

    if confidence_score > 0.6:
        st.success(f"High Confidence ({confidence_score*100:.1f}%)")
    elif confidence_score > 0.3:
        st.warning(f"Moderate Confidence ({confidence_score*100:.1f}%)")
    else:
        st.error(f"Low Confidence ({confidence_score*100:.1f}%)")

    # Business Impact
    st.subheader("Estimated Business Impact")
    estimated_clv = monthly_charges * tenure
    expected_loss = proba * estimated_clv

    st.info(f"""
    Estimated Customer Lifetime Value: ${estimated_clv:,.2f}  
    Expected Revenue at Risk: ${expected_loss:,.2f}
    """)

    # Recommendation
    st.subheader("Recommended Action")
    if level == "High":
        st.error("⚠ Immediate retention offer & discount recommended.")
    elif level == "Medium":
        st.warning("📞 Loyalty follow-up call suggested.")
    else:
        st.success("✅ Customer stable. Maintain engagement.")

    # What-if Simulation
    st.subheader("What-If Simulation")

    if st.button("Simulate Upgrade to 2-Year Contract"):
        simulated_df = st.session_state.input_df.copy()
        simulated_df["Contract"] = "Two year"
        simulated_df["Contract_Length"] = 2
        simulated_df["High Risk Customer"] = 0

        new_proba = model.predict_proba(simulated_df)[0][1]

        col1, col2 = st.columns(2)
        col1.metric("Current Risk", f"{proba*100:.2f}%")
        col2.metric(
            "After 2-Year Contract",
            f"{new_proba*100:.2f}%",
            delta=f"{(new_proba - proba)*100:.2f}%"
        )

    # Feature Importance
    st.subheader("Top Risk Drivers")

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        def clean_name(name):
            return name.replace("num__", "").replace("cat__", "").replace("_", " ")

        cleaned = [clean_name(f) for f in feature_names]

        fi_df = pd.DataFrame({
            "Feature": cleaned,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        st.bar_chart(fi_df.set_index("Feature"))

    # SHAP Explanation
    st.subheader("Explainable AI (SHAP)")

    try:
        preprocessor = model.named_steps["preprocessor"]
        X_processed = preprocessor.transform(input_df)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = preprocessor.get_feature_names_out()
        cleaned = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

        shap_df = pd.DataFrame(shap_values, columns=cleaned)
        impacts = shap_df.iloc[0]
        top_features = impacts.abs().sort_values(ascending=False).head(3).index

        for feature in top_features:
            value = impacts[feature]
            if value > 0:
                st.error(f"{feature} increased risk by +{value:.3f}")
            else:
                st.success(f"{feature} reduced risk by {value:.3f}")

    except:
        st.info("SHAP explanation not supported for this model.")

    # Executive Insight
    st.subheader("Executive Insight Summary")

    if level == "High":
        st.info("This customer is highly likely to churn. Immediate intervention required.")
    elif level == "Medium":
        st.info("Customer shows moderate churn signals. Proactive engagement recommended.")
    else:
        st.info("Customer appears stable with low churn probability.")
