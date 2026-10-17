from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from data_cleaning import clean_data
from data_ingestion import ingest_data
from sklearn.impute import SimpleImputer
df = ingest_data()
df = clean_data(df)

def preprocess_data(df):
    """Preprocesses the data by handling missing values, encoding categorical variables, and scaling numerical features."""

    numerical_features = ["age", "transaction_amount", "account_balance"]
    categorical_features = ["gender", "state", "city","bank_branch","account_type", "transaction_date","transaction_device","transaction_location","device_type","transaction_currency"]
    
    #Creating the preprocessing pipeline for numerical features and categorical features 

    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler() ),
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor

def split_data(df):
    """Splits the data into training and testing sets."""
    numerical_features = ["age", "transaction_amount", "account_balance"]
    categorical_features = ["gender", "state", "city","bank_branch","account_type", "transaction_date","transaction_device","transaction_location","device_type","transaction_currency"]
    X = df[categorical_features + numerical_features]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print(X)
    print(y)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test



def main():
    preprocess_data(df)
    split_data(df)
    print("Data Preprocessing Completed")
    
    



if __name__ == "__main__":
    main()