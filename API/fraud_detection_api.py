from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np



#importing the data
data =  pd.read_csv("data/cleaned_data.csv")

app = FastAPI(title="Fraud Detection API")


#load Model
with open ("model/Fraud_Detection.pkl", "rb") as f:
    model = joblib.load(f)


#Defining what input the model expects 

class FraudInput(BaseModel):
    age: int
    transaction_amount: int
    account_balance: int
    gender: str
    state: str
    city: str
    bank_branch: str
    account_type: str
    transaction_date: str
    transaction_device: str
    transaction_location: str
    device_type: str
    transaction_currency: str


@app.get("/")
def home():
    return {"message": "Bank Fraud Detection API is running"}


@app.post("/predict")
def predict_fraud(data: FraudInput):
    features = np.array([[
        data.age,
        data.transaction_amount,
        data.account_balance,
        data.gender,
        data.state,
        data.city,
        data.bank_branch,
        data.account_type,
        data.transaction_date,
        data.transaction_device,
        data.transaction_location,
        data.device_type,
        data.transaction_currency
    ]])
    prediction = model.predict(features)[0]

    return {
        "predicted_fraud": prediction
    }


if __name__ == "__main__":
    uvicorn.run("fraud_detection_api:app", host="0.0.0.0", port=8000)