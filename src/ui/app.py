import gradio as gr
import requests
import os

# --------------------------------------------------
# API Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://telco-churn-alb-1746633897.us-east-1.elb.amazonaws.com/predict"
)


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
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
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        probability = result["churn_probability"]
        prediction = result["prediction"]

        # --------------------------------------------------
        # Model Prediction
        # --------------------------------------------------

        if prediction == "Churn":
            status = "⚠️ Customer likely to churn"
        else:
            status = "✅ Customer likely to stay"

        # --------------------------------------------------
        # Risk Level
        # --------------------------------------------------

        if probability < 0.35:
            risk_level = "🟢 Low Risk"

        elif probability <= 0.60:
            risk_level = "🟡 Medium Risk"

        else:
            risk_level = "🔴 High Risk"

        # --------------------------------------------------
        # SHAP Explanations
        # --------------------------------------------------

        shap_explanations = result.get(
            "shap_explanations",
            []
        )

        explanation_lines = []

        for item in shap_explanations:

            feature = item["feature"]
            shap_value = item["shap_value"]
            direction = item["direction"]

            if direction == "increases_churn":
                arrow = "↑"
                effect = "increases churn risk"
            else:
                arrow = "↓"
                effect = "decreases churn risk"

            explanation_lines.append(
                f"{arrow} {feature}: "
                f"{shap_value:.4f} "
                f"({effect})"
            )

        if explanation_lines:
            explanation_text = "\n".join(
                explanation_lines
            )
        else:
            explanation_text = (
                "No SHAP explanations available"
            )

        # --------------------------------------------------
        # Return UI Results
        # --------------------------------------------------

        return (
            status,
            risk_level,
            f"{probability * 100:.2f}%",
            f"{(1 - probability) * 100:.2f}%",
            explanation_text,
        )

    except requests.RequestException as exc:

        return (
            "API request failed",
            "N/A",
            "N/A",
            "N/A",
            str(exc),
        )


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------

with gr.Blocks(
    title="Telco Customer Churn Predictor"
) as demo:

    gr.Markdown(
        """
        # Telco Customer Churn Predictor

        Enter the customer's information below to estimate
        their probability of leaving the telecom company.
        """
    )

    # --------------------------------------------------
    # Customer Information
    # --------------------------------------------------

    with gr.Row():

        with gr.Column():

            gr.Markdown(
                "### Customer Information"
            )

            gender = gr.Dropdown(
                ["Male", "Female"],
                label="Gender",
                value="Male",
            )

            senior_citizen = gr.Dropdown(
                [0, 1],
                label="Senior Citizen",
                value=0,
            )

            partner = gr.Dropdown(
                ["Yes", "No"],
                label="Partner",
                value="No",
            )

            dependents = gr.Dropdown(
                ["Yes", "No"],
                label="Dependents",
                value="No",
            )

            tenure = gr.Slider(
                minimum=0,
                maximum=72,
                value=12,
                step=1,
                label="Tenure (Months)",
            )

        # --------------------------------------------------
        # Phone & Internet
        # --------------------------------------------------

        with gr.Column():

            gr.Markdown(
                "### Phone & Internet"
            )

            phone_service = gr.Dropdown(
                ["Yes", "No"],
                label="Phone Service",
                value="Yes",
            )

            multiple_lines = gr.Dropdown(
                [
                    "Yes",
                    "No",
                    "No phone service",
                ],
                label="Multiple Lines",
                value="No",
            )

            internet_service = gr.Dropdown(
                [
                    "DSL",
                    "Fiber optic",
                    "No",
                ],
                label="Internet Service",
                value="Fiber optic",
            )

            online_security = gr.Dropdown(
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
                label="Online Security",
                value="No",
            )

            online_backup = gr.Dropdown(
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
                label="Online Backup",
                value="No",
            )

        # --------------------------------------------------
        # Additional Services
        # --------------------------------------------------

        with gr.Column():

            gr.Markdown(
                "### Additional Services"
            )

            device_protection = gr.Dropdown(
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
                label="Device Protection",
                value="No",
            )

            tech_support = gr.Dropdown(
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
                label="Tech Support",
                value="No",
            )

            streaming_tv = gr.Dropdown(
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
                label="Streaming TV",
                value="No",
            )

            streaming_movies = gr.Dropdown(
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
                label="Streaming Movies",
                value="No",
            )

    # --------------------------------------------------
    # Contract & Billing
    # --------------------------------------------------

    gr.Markdown(
        "### Contract & Billing"
    )

    with gr.Row():

        contract = gr.Dropdown(
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            label="Contract",
            value="Month-to-month",
        )

        paperless_billing = gr.Dropdown(
            ["Yes", "No"],
            label="Paperless Billing",
            value="Yes",
        )

        payment_method = gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
            value="Electronic check",
        )

    with gr.Row():

        monthly_charges = gr.Number(
            label="Monthly Charges ($)",
            value=70.0,
        )

        total_charges = gr.Number(
            label="Total Charges ($)",
            value=840.0,
        )

    # --------------------------------------------------
    # Prediction Button
    # --------------------------------------------------

    predict_button = gr.Button(
        "Predict Churn",
        variant="primary",
    )

    # --------------------------------------------------
    # Prediction Results
    # --------------------------------------------------

    gr.Markdown(
        "## Prediction"
    )

    with gr.Row():

        prediction_output = gr.Textbox(
            label="Prediction",
        )

        risk_output = gr.Textbox(
            label="Risk Level",
        )

        churn_probability = gr.Textbox(
            label="Churn Probability",
        )

        stay_probability = gr.Textbox(
            label="Stay Probability",
        )

    # --------------------------------------------------
    # SHAP Explanation
    # --------------------------------------------------

    shap_output = gr.Textbox(
        label="Why This Prediction? (SHAP)",
        lines=6,
    )

    # --------------------------------------------------
    # Button Action
    # --------------------------------------------------

    predict_button.click(

        fn=predict_churn,

        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        ],

        outputs=[
            prediction_output,
            risk_output,
            churn_probability,
            stay_probability,
            shap_output,
        ],
    )


# --------------------------------------------------
# Launch Application
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )