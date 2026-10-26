from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

from src.data_ingestion import ingest_data
from src.data_cleaning import clean_data

data = clean_data(ingest_data())

app = FastAPI(title="Fraud Detection API")


#load Model
MODEL_PATH = r"C:\Users\HP\Desktop\Machine Learning Projects\Bank Fraud Detection - Transactions\Fraud_Detection.pkl"

model = joblib.load(MODEL_PATH)


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
    features = pd.DataFrame([data.model_dump()])
    prediction = model.predict(features)

    return {
        "predicted_fraud": int(prediction[0])
    }
