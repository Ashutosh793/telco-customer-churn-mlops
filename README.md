# Telco Customer Churn – End-to-End ML & MLOps System

An end-to-end machine learning system for predicting telecom customer churn, exposing predictions through a FastAPI backend and an interactive Gradio frontend, with Dockerized deployment on AWS ECS Fargate, automated CI/CD using GitHub Actions, model explainability with SHAP, experiment tracking with MLflow, monitoring with CloudWatch, and ECS service auto scaling.

---

## Live Application

**Frontend:**  
http://telco-churn-ui-alb-1250685601.us-east-1.elb.amazonaws.com

The application allows users to enter customer information and receive:

- Churn / No Churn prediction
- Churn probability
- Stay probability
- Low / Medium / High risk classification
- SHAP-based explanation of the most influential features

---

## Business Problem

Customer churn directly affects revenue, acquisition costs, and long-term customer value.

The goal of this project is to build a machine learning system that identifies customers who are likely to leave a telecom provider so retention teams can intervene earlier.

Rather than stopping at model training, this project operationalizes the model as a complete production-style ML application.

---

## Key Features

- End-to-end customer churn classification pipeline
- Data preprocessing with Scikit-learn pipelines
- Logistic Regression, Random Forest, and XGBoost model comparison
- Hyperparameter tuning for XGBoost
- Custom classification threshold optimized for churn detection
- MLflow experiment tracking
- FastAPI REST API for real-time inference
- SHAP explainability for individual predictions
- Interactive Gradio web application
- Separate Docker containers for backend and frontend
- AWS ECS Fargate deployment
- Application Load Balancers for UI and API services
- Amazon ECR container registry
- GitHub Actions CI/CD
- Post-deployment smoke testing
- CloudWatch monitoring and logging
- SNS alerting
- ECS CPU-based auto scaling

---

## Machine Learning Pipeline

The raw Telco Customer Churn dataset contains customer demographics, services, account information, billing details, and churn status.

The training pipeline performs:

1. Data loading and cleaning
2. Conversion of `TotalCharges` to numeric values
3. Removal of non-predictive `customerID`
4. Binary encoding of the churn target
5. Stratified train/test split
6. Standardization of numerical features
7. One-hot encoding of categorical features
8. Model training
9. Probability-based evaluation
10. Custom threshold application

The preprocessing and XGBoost classifier are combined into a single Scikit-learn pipeline to ensure the same transformations are applied during both training and inference.

---

## Model Development

Three primary models were evaluated:

| Model | ROC-AUC | Accuracy | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8420 | 0.8055 | 0.5588 | 0.6040 |
| Random Forest | 0.8198 | 0.7821 | 0.4786 | 0.5383 |
| XGBoost | 0.8434 | 0.8062 | 0.5321 | 0.5931 |

Cross-validation produced approximately:

| Model | Mean CV ROC-AUC |
|---|---:|
| Logistic Regression | 0.8461 |
| Random Forest | 0.8202 |
| XGBoost | 0.8462 |

After hyperparameter tuning, XGBoost achieved a CV ROC-AUC of approximately **0.8510**.

---

## Final XGBoost Model

The deployed XGBoost model uses:

```python
XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.01,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
```

The training code logs model parameters, evaluation metrics, and the complete Scikit-learn pipeline using MLflow.

---

## Threshold Optimization

The default classification threshold of `0.50` was not used for the deployed model.

Because identifying customers at risk of churn is more important than maximizing accuracy alone, threshold tuning was performed using out-of-fold predictions from the training data.

The final classification threshold was set to:

```text
0.35
```

At this threshold, the model achieved approximately:

| Metric | Value |
|---|---:|
| Accuracy | 0.7835 |
| Precision | 0.5729 |
| Recall | 0.7246 |
| F1 Score | 0.6399 |
| ROC-AUC | 0.8486 |

The lower threshold increases recall, allowing the system to identify a larger percentage of customers who may churn.

---

## Model Explainability with SHAP

The deployed API uses SHAP to explain individual XGBoost predictions.

For each request, the API returns the five most influential transformed features and whether each feature pushes the prediction toward or away from churn.

Example response:

```json
{
  "churn_probability": 0.5633,
  "prediction": "Churn",
  "threshold": 0.35,
  "shap_explanations": [
    {
      "feature": "Contract_Month-to-month",
      "shap_value": 0.5774,
      "direction": "increases_churn"
    },
    {
      "feature": "InternetService_Fiber optic",
      "shap_value": 0.2419,
      "direction": "increases_churn"
    }
  ]
}
```

Positive SHAP values push the prediction toward churn, while negative SHAP values push it toward staying.

---

## FastAPI Backend

The trained ML pipeline is exposed using FastAPI.

Main endpoints:

```text
GET /
POST /predict
GET /docs
```

The prediction endpoint accepts raw customer information, applies the same preprocessing pipeline used during training, generates the XGBoost churn probability, applies the `0.35` threshold, calculates SHAP values, and returns the prediction plus explanations.

---

## Gradio Frontend

The Gradio frontend provides an interactive interface for entering customer information and viewing prediction results.

The UI displays:

- Prediction
- Risk Level
- Churn Probability
- Stay Probability
- SHAP Explanation

---

## System Architecture

```text
                         User
                          |
                          v
                UI Application Load Balancer
                          |
                          v
                  Gradio Frontend
                   AWS ECS Fargate
                          |
                          v
                API Application Load Balancer
                          |
                          v
                     FastAPI API
                   AWS ECS Fargate
                          |
                          v
                Preprocessing Pipeline
                          |
                          v
                       XGBoost
                          |
                          v
                         SHAP
```

### CI/CD Architecture

```text
GitHub
   |
   v
GitHub Actions
   |
   v
Docker Build
   |
   v
Amazon ECR
   |
   v
AWS ECS Fargate
   |
   v
Application Load Balancer
```

---

## AWS Deployment

The application is deployed using:

- **Amazon ECS Fargate** – serverless container execution
- **Amazon ECR** – Docker image registry
- **Application Load Balancer** – traffic routing and health checks
- **CloudWatch** – logs and infrastructure metrics
- **SNS** – alert notifications
- **IAM / OIDC** – secure GitHub Actions authentication
- **ECS Service Auto Scaling** – dynamic task scaling

The frontend and backend run as separate ECS services.

---

## CI/CD

GitHub Actions automates continuous integration and deployment.

Three workflows are used:

```text
.github/workflows/
├── ci.yml
├── deploy.yml
└── deploy-ui.yml
```

### API Deployment

```text
Git Push
   ↓
GitHub Actions
   ↓
Docker Build
   ↓
Push API Image to Amazon ECR
   ↓
Trigger ECS Rolling Deployment
   ↓
Wait for Deployment Completion
```

### UI Deployment

```text
UI Code Change
   ↓
GitHub Actions
   ↓
Build Dockerfile.ui
   ↓
Push UI Image to Amazon ECR
   ↓
Trigger UI ECS Deployment
   ↓
Wait for Deployment Completion
   ↓
HTTP Smoke Test
```

The UI workflow performs a post-deployment smoke test and fails if the public application does not return HTTP `200`.

GitHub Actions authenticates with AWS using OpenID Connect instead of storing long-lived AWS access keys in GitHub.

---

## Monitoring and Auto Scaling

Application logs and ECS metrics are monitored using Amazon CloudWatch.

Monitoring includes:

- ECS CPU utilization
- ECS memory utilization
- Application logs
- Load balancer target health
- ECS deployment events

A CloudWatch alarm monitors sustained high CPU utilization and sends alerts through Amazon SNS.

The backend ECS service uses target-tracking auto scaling:

```text
Minimum tasks: 1
Maximum tasks: 3
Target CPU utilization: 60%
```

---

## Project Structure

```text
telco-customer-churn-mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── deploy.yml
│       └── deploy-ui.yml
│
├── data/
│   └── raw/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── app/
│   │   └── main.py
│   ├── ui/
│   │   └── app.py
│   ├── train.py
│   ├── predict.py
│   └── export_model.py
│
├── Dockerfile
├── Dockerfile.ui
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ashutosh793/telco-customer-churn-mlops.git
cd telco-customer-churn-mlops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Gradio frontend

In another terminal:

```bash
python -m src.ui.app
```

Frontend:

```text
http://127.0.0.1:7860
```

---

## Docker

### Build Backend Image

```bash
docker build -t telco-churn-api .
```

Run it:

```bash
docker run -p 8000:8000 telco-churn-api
```

### Build UI Image

```bash
docker build -f Dockerfile.ui -t telco-churn-ui .
```

Run it:

```bash
docker run -p 7860:7860 telco-churn-ui
```

---

## Technologies Used

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- MLflow

### Backend

- FastAPI
- Pydantic

### Frontend

- Gradio

### MLOps / DevOps

- Docker
- GitHub Actions
- Amazon ECR
- AWS ECS Fargate
- Application Load Balancer
- Amazon CloudWatch
- Amazon SNS
- AWS IAM
- GitHub OIDC

---

## Future Improvements

- Immutable Docker deployments using Git commit SHA
- HTTPS using AWS Certificate Manager
- Custom domain through Route 53
- More extensive API integration tests
- Data drift and concept drift monitoring
- Automated model retraining
- Model registry promotion workflow
- Infrastructure as Code using Terraform
- Separate lightweight dependency files for frontend and backend images

---

## Disclaimer

This project is intended for educational and portfolio purposes.

Model predictions should not be treated as production business decisions without additional validation, monitoring, governance, security controls, and domain-specific evaluation.
