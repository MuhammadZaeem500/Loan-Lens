import gradio as gr
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ============================================================
# FEATURES USED DURING MODEL TRAINING
# ============================================================

FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_loan(
    dependents,
    education,
    self_employed,
    annual_income,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_assets
):

    try:

        # ----------------------------------------------------
        # Basic input validation
        # ----------------------------------------------------

        numeric_values = [
            dependents,
            annual_income,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets,
            commercial_assets,
            luxury_assets,
            bank_assets
        ]

        if any(value is None for value in numeric_values):
            return "Invalid Input", "Please complete all fields."

        if dependents < 0:
            return "Invalid Input", "Dependents cannot be negative."

        if annual_income < 0:
            return "Invalid Input", "Annual income cannot be negative."

        if loan_amount < 0:
            return "Invalid Input", "Loan amount cannot be negative."

        if loan_term <= 0:
            return "Invalid Input", "Loan term must be greater than 0."

        if cibil_score < 300 or cibil_score > 900:
            return "Invalid Input", "CIBIL score must be between 300 and 900."

        if any(
            value < 0
            for value in [
                residential_assets,
                commercial_assets,
                luxury_assets,
                bank_assets
            ]
        ):
            return "Invalid Input", "Asset values cannot be negative."


        # ----------------------------------------------------
        # ENCODE CATEGORICAL VALUES
        # ----------------------------------------------------

        education_encoded = 1 if education == "Graduate" else 0
        self_employed_encoded = 1 if self_employed == "Yes" else 0


        # ----------------------------------------------------
        # CREATE DATAFRAME FOR APPLICANT
        # ----------------------------------------------------

        applicant = pd.DataFrame([{
            "no_of_dependents": dependents,
            "education": education_encoded,
            "self_employed": self_employed_encoded,
            "income_annum": annual_income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets,
            "commercial_assets_value": commercial_assets,
            "luxury_assets_value": luxury_assets,
            "bank_asset_value": bank_assets
        }])


        # ----------------------------------------------------
        # ENSURE SAME FEATURE ORDER AS TRAINING
        # ----------------------------------------------------

        applicant = applicant[FEATURES]


        # ----------------------------------------------------
        # SCALE INPUT DATA
        # ----------------------------------------------------

        applicant_scaled = scaler.transform(applicant)


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(applicant_scaled)[0]

        probabilities = model.predict_proba(applicant_scaled)[0]

        approval_probability = probabilities[1]


        # ----------------------------------------------------
        # FORMAT RESULT
        # ----------------------------------------------------

        if prediction == 1:
            status = "Approved"
        else:
            status = "Rejected"

        probability_text = f"{approval_probability * 100:.2f}%"

        return status, probability_text


    except Exception as error:

        return "Prediction Error", str(error)


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

.gradio-container {
    max-width: 1300px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    margin-bottom: 5px;
}

#subtitle {
    text-align: center;
    margin-bottom: 25px;
}

#predict-button {
    font-size: 16px;
    font-weight: 600;
}

#result-section {
    margin-top: 20px;
}

"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Loan Lens - Loan Approval Prediction",
    theme=gr.themes.Soft(),
    css=custom_css
) as app:

    gr.Markdown(
        """
        # Loan Lens
        ## Loan Approval Prediction System

        Enter the applicant's financial and personal information below
        to estimate the probability of loan approval using a trained
        Logistic Regression model.
        """,
        elem_id="main-title"
    )


    # ========================================================
    # INPUT SECTION
    # ========================================================

    with gr.Row():

        # ----------------------------------------------------
        # LEFT COLUMN
        # ----------------------------------------------------

        with gr.Column():

            gr.Markdown("### Applicant Information")

            dependents = gr.Number(
                label="Number of Dependents",
                value=0,
                minimum=0
            )

            education = gr.Dropdown(
                choices=[
                    "Graduate",
                    "Not Graduate"
                ],
                label="Education",
                value="Graduate"
            )

            self_employed = gr.Dropdown(
                choices=[
                    "Yes",
                    "No"
                ],
                label="Self Employed",
                value="No"
            )

            annual_income = gr.Number(
                label="Annual Income",
                value=0,
                minimum=0
            )

            loan_amount = gr.Number(
                label="Loan Amount",
                value=0,
                minimum=0
            )

            loan_term = gr.Number(
                label="Loan Term (Years)",
                value=1,
                minimum=1
            )


        # ----------------------------------------------------
        # RIGHT COLUMN
        # ----------------------------------------------------

        with gr.Column():

            gr.Markdown("### Credit & Asset Information")

            cibil_score = gr.Number(
                label="CIBIL Score",
                value=600,
                minimum=300,
                maximum=900
            )

            residential_assets = gr.Number(
                label="Residential Assets Value",
                value=0,
                minimum=0
            )

            commercial_assets = gr.Number(
                label="Commercial Assets Value",
                value=0,
                minimum=0
            )

            luxury_assets = gr.Number(
                label="Luxury Assets Value",
                value=0,
                minimum=0
            )

            bank_assets = gr.Number(
                label="Bank Assets Value",
                value=0,
                minimum=0
            )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = gr.Button(
        "Predict Loan Approval",
        variant="primary",
        elem_id="predict-button"
    )


    # ========================================================
    # RESULT SECTION
    # ========================================================

    gr.Markdown(
        "## Prediction Result",
        elem_id="result-section"
    )

    with gr.Row():

        prediction_output = gr.Textbox(
            label="Loan Prediction",
            interactive=False
        )

        probability_output = gr.Textbox(
            label="Approval Probability",
            interactive=False
        )


    # ========================================================
    # BUTTON EVENT
    # ========================================================

    predict_button.click(

        fn=predict_loan,

        inputs=[
            dependents,
            education,
            self_employed,
            annual_income,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets,
            commercial_assets,
            luxury_assets,
            bank_assets
        ],

        outputs=[
            prediction_output,
            probability_output
        ]
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    gr.Markdown(
        """
        ---
        **Note:** Loan Lens is a machine learning demonstration project.
        Predictions are generated by a Logistic Regression model and
        should not be treated as a real financial decision.
        """
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.launch()