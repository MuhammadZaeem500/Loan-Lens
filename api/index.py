from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Configure CORS to allow requests from your Vercel frontend and local environments
origins = [
    "https://loan-lens-sage.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated Pydantic schema with ALL fields sent by your frontend
class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    cibil_score: int
    loan_amount: float
    loan_term: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

@app.post("/api")
def predict_loan(data: LoanApplication):
    print("Received prediction request:", data.dict())

    # 1. Prediction logic based on CIBIL score
    prediction = "Approved" if data.cibil_score >= 650 else "Rejected"

    # 2. Dynamic probability calculation (or use your model's predict_proba)
    approved_pct = round(min(max((data.cibil_score - 300) / 600 * 100, 5.0), 95.0), 2)
    rejected_pct = round(100.0 - approved_pct, 2)

    # 3. Return response with the 'probabilities' key included
    return {
        "status": "success",
        "prediction": prediction,
        "probabilities": f"Approved Probability: {approved_pct}%\nRejected Probability: {rejected_pct}%"
    }