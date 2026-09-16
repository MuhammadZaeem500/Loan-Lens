from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = [
    "https://loan-lens-sage.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Can also use ["*"] if you want to allow all origins temporarily
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    cibil_score: int
    loan_amount: float

# Route matched to your Render backend endpoint structure
@app.post("/api")
def predict_loan(data: LoanApplication):
    print("Received prediction request:", data.dict())

    # Mock or real model prediction logic here
    prediction = "Approved" if data.cibil_score >= 650 else "Rejected"

    return {
        "status": "success",
        "prediction": prediction
    }