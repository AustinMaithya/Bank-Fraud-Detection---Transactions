from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd




data =  pd.read_csv("C:\\Users\\HP\Desktop\\Machine Learning Projects\\Bank Fraud Detection - Transactions\\data\\cleaned_data.csv")

app = FastAPI(title="Fraud Detection API")


#load Model
with open ("model\Fraud_Detection.pkl", "rb") as f :
    model =  pickle.load(f)


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
    features = [[
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
    ]]
    prediction = model.predict(features)

    return {
        "predicted_fraud": int(prediction[0])
    }

