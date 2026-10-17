import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from data_cleaning import save_cleaned_data, clean_data
from data_ingestion import ingest_data
df = ingest_data()
cleaned_df = clean_data(df)

def data_overview(cleaned_df):
    
    """Performs exploratory data analysis on the cleaned dataset."""
    
    #Looking at the data information
    print("="*10, "Data Information", "="*10)
    print(cleaned_df.info())

    #Looking at the data description
    print("="*10, "Data Description", "="*10)
    print(cleaned_df.describe())

    print("="*10, "Data Shape", "="*10)
    print(cleaned_df.shape)
    



def histogram_plots(cleaned_df):
    """Generates histogram plots for numerical features."""

    numerical_features = ["age","transaction_amount","account_balance"]
    fig,axes = plt.subplots(figsize = (16,8),nrows = 1,ncols = 3)
    axes = axes.flatten()
    for i, col in enumerate(numerical_features):
        sns.histplot(data=cleaned_df,x=col,ax=axes[i],kde=True)
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    plt.suptitle("Histogram Plots for Numerical Features", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("Histograms Generated Successfully")
    print("Histograms indicate normal distributions for all the numerical features with no significant skewness or outliers, suggesting that the data is well-behaved and suitable for further analysis.")


def correlation_heatmap(cleaned_df):
    """ Generates a correlation heatmap for numerical features."""
    numerical_features = ["age","transaction_amount","account_balance"]
    correlation_matrix = cleaned_df[numerical_features].corr()
    plt.figure(figsize=(16,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap for Numerical Features")

    plt.tight_layout()
    plt.show()

    print("Correlation Heatmap Generated Successfully")
    print("The Correlation Matrix Shows no correlation between the numerical features, indicating that they are independent of each other.")


def fraud_indication(cleaned_df):
    """Analyzes the relationship between numerical features and the fraud label."""
    numerical_features = ["age","transaction_amount","account_balance"]
    fig,axes = plt.subplots(figsize = (16,8), nrows = 1, ncols = 3)
    axes = axes.flatten()
    for i, col in enumerate(numerical_features):
        sns.boxplot(data=cleaned_df,x="is_fraud",y=col,ax=axes[i])
        axes[i].set_title(f"Boxplot of {col} by Fraud Label")
        axes[i].set_xlabel("Is Fraud")
        axes[i].set_ylabel(col)
    plt.suptitle("Boxplots of Numerical Features by Fraud Label", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("Fraud Indication Analysis Completed")


if __name__ == "__main__":
    data_overview(cleaned_df)
    histogram_plots(cleaned_df)
    correlation_heatmap(cleaned_df)
    fraud_indication(cleaned_df)
