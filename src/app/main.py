from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

THRESHOLD = 0.35

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model"


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = mlflow.sklearn.load_model(
    str(MODEL_PATH)
)

# Separate preprocessing and XGBoost classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# SHAP explainer is created once when the API starts
explainer = shap.TreeExplainer(classifier)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predict customer churn using a trained XGBoost model.",
    version="1.2.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def clean_feature_name(feature_name: str) -> str:
    """
    Make transformed feature names easier to read.

    Example:
    cat__Contract_Month-to-month
        ->
    Contract_Month-to-month
    """

    feature_name = feature_name.replace(
        "num__",
        ""
    )

    feature_name = feature_name.replace(
        "cat__",
        ""
    )

    return feature_name


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "telco-churn-api",
        "model": "XGBoost",
        "threshold": THRESHOLD,
        "version": "1.2.0",
        "explainability": "SHAP"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    # ----------------------------------------------
    # Convert request to DataFrame
    # ----------------------------------------------

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    # ----------------------------------------------
    # Normal model prediction
    # ----------------------------------------------

    probability = model.predict_proba(
        customer_df
    )[0, 1]

    prediction = int(
        probability >= THRESHOLD
    )

    # ----------------------------------------------
    # Transform customer using same preprocessing
    # used during model training
    # ----------------------------------------------

    transformed_customer = preprocessor.transform(
        customer_df
    )

    # Convert sparse matrix if necessary
    if hasattr(transformed_customer, "toarray"):
        transformed_customer = (
            transformed_customer.toarray()
        )

    # ----------------------------------------------
    # Calculate SHAP values
    # ----------------------------------------------

    shap_values = explainer.shap_values(
        transformed_customer
    )

    shap_values = np.asarray(shap_values)

    # Handle possible SHAP output dimensions
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    customer_shap_values = shap_values[0]

    # ----------------------------------------------
    # Pair features with SHAP contributions
    # ----------------------------------------------

    explanations = []

    for feature, shap_value in zip(
        feature_names,
        customer_shap_values
    ):

        shap_value = float(shap_value)

        explanations.append(
            {
                "feature": clean_feature_name(
                    feature
                ),
                "shap_value": round(
                    shap_value,
                    4
                ),
                "direction": (
                    "increases_churn"
                    if shap_value > 0
                    else "decreases_churn"
                )
            }
        )

    # ----------------------------------------------
    # Select most influential features
    # ----------------------------------------------

    explanations = sorted(
        explanations,
        key=lambda item: abs(
            item["shap_value"]
        ),
        reverse=True
    )

    top_explanations = explanations[:5]

    # ----------------------------------------------
    # API response
    # ----------------------------------------------

    return {
        "churn_probability": round(
            float(probability),
            4
        ),

        "prediction": (
            "Churn"
            if prediction == 1
            else "No Churn"
        ),

        "threshold": THRESHOLD,

        "shap_explanations": top_explanations
    }