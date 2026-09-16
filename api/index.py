from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Loan Lens API")

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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
    "bank_asset_value",
]


class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.get("/api")
def home():
    return {
        "status": "online",
        "message": "Loan Lens Prediction API"
    }


@app.post("/api/predict")
def predict_loan(application: LoanApplication):

    education_value = 1 if application.education == "Graduate" else 0
    self_employed_value = 1 if application.self_employed == "Yes" else 0

    applicant = pd.DataFrame([{
        "no_of_dependents": application.no_of_dependents,
        "education": education_value,
        "self_employed": self_employed_value,
        "income_annum": application.income_annum,
        "loan_amount": application.loan_amount,
        "loan_term": application.loan_term,
        "cibil_score": application.cibil_score,
        "residential_assets_value": application.residential_assets_value,
        "commercial_assets_value": application.commercial_assets_value,
        "luxury_assets_value": application.luxury_assets_value,
        "bank_asset_value": application.bank_asset_value,
    }])

    applicant = applicant[FEATURES]

    applicant_scaled = scaler.transform(applicant)

    prediction = model.predict(applicant_scaled)[0]
    probability = model.predict_proba(applicant_scaled)[0][1]

    status = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": status,
        "approval_probability": round(float(probability) * 100, 2)
    }