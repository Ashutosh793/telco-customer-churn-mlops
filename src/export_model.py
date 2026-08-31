import mlflow
import mlflow.sklearn
from pathlib import Path


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Telco-Churn-Classification"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model"

mlflow.set_tracking_uri(TRACKING_URI)


# Find latest successful run
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1,
)

run_id = runs.iloc[0]["run_id"]

print(f"Loading model from run: {run_id}")


# Load model from MLflow
model_uri = f"runs:/{run_id}/model"

model = mlflow.sklearn.load_model(model_uri)


# Create models directory
MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# Save model locally
mlflow.sklearn.save_model(
    sk_model=model,
    path=str(MODEL_PATH),
    skops_trusted_types=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
    ],
)

print(f"Model exported to: {MODEL_PATH}")