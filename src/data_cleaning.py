import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_ingestion import ingest_data
df = ingest_data()
def clean_data(df):
    """Cleans the data by handling missing values and duplicates."""
    df = ingest_data()
    # Drop duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Dropped {duplicates} duplicate rows.")
    # Handle missing values (if any)
    missing_count = df.isnull().sum().sum()

    if missing_count > 0:
        # Simple placeholder strategy; choose imputations appropriate
        # to each column's meaning and data type.
        df = df.fillna(0)
        print(f"Filled {missing_count} missing values.")

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()

    return df



def save_cleaned_data(df):
    """ Saved cleaned dataframe to csv"""
    df = clean_data(df)
    cleaned_df = df.copy()
    cleaned_df.to_csv("data/cleaned_data.csv")
    print("Clean Data Saved Successfully")

    return cleaned_df


def date_formaatting(cleaned_df):
    """Formats the date column to datetime type."""
    cleaned_df['transaction_date'] = pd.to_datetime(cleaned_df['transaction_date'], errors='coerce')
    return cleaned_df



if __name__ == "__main__":
    clean_data(df)
    cleaned_df = clean_data(df)
    date_formaatting(cleaned_df)
    save_cleaned_data(cleaned_df)
    print(cleaned_df.head())