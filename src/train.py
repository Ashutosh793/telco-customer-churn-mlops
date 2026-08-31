from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.35

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# --------------------------------------------------
# Load and clean data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce",
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1,
})


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    col
    for col in X.columns
    if col not in numerical_features
]

numerical_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore"),
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numerical_pipeline,
        numerical_features,
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features,
    ),
])


# --------------------------------------------------
# XGBoost
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.01,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=1,
)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model),
])


# --------------------------------------------------
# Train
# --------------------------------------------------

# --------------------------------------------------
# MLflow
# --------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Telco-Churn-Classification")


with mlflow.start_run(run_name="tuned-xgboost"):

    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    model_pipeline.fit(X_train, y_train)

    # ----------------------------------------------
    # Evaluate
    # ----------------------------------------------

    probabilities = model_pipeline.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    # ----------------------------------------------
    # Log parameters
    # ----------------------------------------------

    mlflow.log_params({
        "model_type": "XGBoost",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.01,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "threshold": THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    })

    # ----------------------------------------------
    # Log metrics
    # ----------------------------------------------

    mlflow.log_metrics(metrics)

    # ----------------------------------------------
    # Log complete pipeline
    # ----------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        name="model",
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )

    # ----------------------------------------------
    # Console output
    # ----------------------------------------------

    print("\nModel Metrics")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")