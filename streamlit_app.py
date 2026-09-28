from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import streamlit as st


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

THRESHOLD = 0.35

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = mlflow.sklearn.load_model(
        str(MODEL_PATH)
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    explainer = shap.TreeExplainer(classifier)

    return (
        model,
        preprocessor,
        feature_names,
        explainer,
    )


model, preprocessor, feature_names, explainer = load_model()


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def clean_feature_name(feature_name):

    return (
        feature_name
        .replace("num__", "")
        .replace("cat__", "")
    )


def predict_churn(customer_df):

    # Prediction
    probability = model.predict_proba(
        customer_df
    )[0, 1]

    prediction = int(
        probability >= THRESHOLD
    )

    # Preprocess for SHAP
    transformed_customer = preprocessor.transform(
        customer_df
    )

    if hasattr(transformed_customer, "toarray"):
        transformed_customer = (
            transformed_customer.toarray()
        )

    # SHAP
    shap_values = explainer.shap_values(
        transformed_customer
    )

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    customer_shap_values = shap_values[0]

    explanations = []

    for feature, shap_value in zip(
        feature_names,
        customer_shap_values,
    ):

        explanations.append(
            {
                "feature": clean_feature_name(feature),
                "shap_value": float(shap_value),
            }
        )

    explanations = sorted(
        explanations,
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return (
        probability,
        prediction,
        explanations[:5],
    )


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 Telco Customer Churn Predictor")

st.write(
    """
    Predict the probability that a telecom customer
    will churn using an XGBoost machine learning model.
    """
)

st.divider()


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"],
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"],
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12,
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service",
        ],
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "Fiber optic",
            "DSL",
            "No",
        ],
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )


with col3:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )


# --------------------------------------------------
# Contract & Billing
# --------------------------------------------------

st.divider()

st.subheader("💳 Contract & Billing")

col4, col5, col6 = st.columns(3)

with col4:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )


with col5:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )


with col6:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0,
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Churn",
    type="primary",
    use_container_width=True,
):

    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    probability, prediction, explanations = (
        predict_churn(customer)
    )

    stay_probability = 1 - probability

    # Risk Level
    if probability < 0.35:
        risk = "🟢 Low Risk"

    elif probability <= 0.60:
        risk = "🟡 Medium Risk"

    else:
        risk = "🔴 High Risk"

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "⚠️ Customer is likely to churn"
        )
    else:
        st.success(
            "✅ Customer is likely to stay"
        )

    result1, result2, result3 = st.columns(3)

    result1.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%",
    )

    result2.metric(
        "Stay Probability",
        f"{stay_probability * 100:.2f}%",
    )

    result3.metric(
        "Risk Level",
        risk,
    )

    st.progress(float(probability))

    # --------------------------------------------------
    # SHAP Explanation
    # --------------------------------------------------

    st.subheader("🔎 Why This Prediction?")

    st.write(
        "Top factors influencing the model's prediction:"
    )

    for item in explanations:

        feature = item["feature"]
        value = item["shap_value"]

        if value > 0:

            st.write(
                f"🔺 **{feature}** "
                f"({value:.4f}) — increases churn risk"
            )

        else:

            st.write(
                f"🔻 **{feature}** "
                f"({value:.4f}) — decreases churn risk"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "XGBoost • Scikit-learn • MLflow • SHAP • Streamlit"
)
