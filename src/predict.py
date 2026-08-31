import mlflow
import mlflow.sklearn
import pandas as pd


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Telco-Churn-Classification"
THRESHOLD = 0.35

mlflow.set_tracking_uri(TRACKING_URI)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1,
)

run_id = runs.iloc[0]["run_id"]

print("Loading model from run:", run_id)

model_uri = f"runs:/{run_id}/model"

model = mlflow.sklearn.load_model(model_uri)

customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 477.50,
}])

probability = model.predict_proba(customer)[0, 1]

prediction = int(probability >= THRESHOLD)

print(f"Churn probability: {probability:.4f}")
print(f"Threshold: {THRESHOLD}")
print(
    "Prediction:",
    "Churn" if prediction == 1 else "No Churn"
)