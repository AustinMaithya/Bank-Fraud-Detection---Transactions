import joblib
from hyperparametertuning import tune_xgboost
from data_ingestion import ingest_data
from data_cleaning import clean_data
from data_preprocessing import split_data


df = clean_data(ingest_data())
X_train, X_test, y_train, y_test = split_data(df)
best_model = tune_xgboost(X_train,y_train)


def save_model(best_model):
    """  
    The model is saved into a pickle file which can be used to create
    an api for model deployment

    """

    joblib.dump(best_model,  "Fraud_Detection.pkl")
    print("\n Model Saved Successfully")


    


if __name__ == "__main__":
    save_model(best_model)

