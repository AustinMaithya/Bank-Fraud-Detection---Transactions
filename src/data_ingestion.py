# Data Ingestion was performed using the kaggle API.
import shutil
shutil.unpack_archive("C:\\Users\\HP\\Desktop\\Machine Learning Projects\\Bank Fraud Detection - Transactions\\bank-transaction-fraud-detection.zip", "data")
def ingest_data():
    """Unpacks data and saves it to a CSV file."""
        
    import pandas as pd
    
    
    df =  pd.read_csv("C:\\Users\\HP\\Desktop\\Machine Learning Projects\\Bank Fraud Detection - Transactions\\data\\Bank_Transaction_Fraud_Detection.csv")
    print("Data Ingestion Completed")

    return df

if __name__ == "__main__":
    df = ingest_data()
    