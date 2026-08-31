from pathlib import Path

import mlflow.sklearn
import pandas as pd

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


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predict customer churn using a trained XGBoost model.",
    version="1.0.0"
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
# Health endpoint
# --------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "threshold": THRESHOLD
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    probability = model.predict_proba(
        customer_df
    )[0, 1]

    prediction = int(
        probability >= THRESHOLD
    )

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
        "threshold": THRESHOLD
    }