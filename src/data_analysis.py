import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def load_clean_data():
    clean_df =  pd.read_csv("data/cleaned_data.csv")

    return clean_df

if __name__ == "__main__":
    clean_df  = load_clean_data()

    print(clean_df.head())