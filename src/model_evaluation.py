from model_training_and_testing import model_train,get_models
from data_cleaning import clean_data
from data_ingestion import ingest_data
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.model_selection import cross_val_score
from data_preprocessing import preprocess_data,split_data
from sklearn.pipeline import Pipeline
from model_training_and_testing import model_train
from sklearn.model_selection import StratifiedKFold
import pandas as pd

df = clean_data(ingest_data())

def evaluate_data(df):
    
    """model evaluation of the trained data"""
    trained_models, X_test, y_test = model_train(df)
    results  = {}
   
    for name, pipeline in trained_models.items():
        # Predict class labels and probabilities
        y_pred = pipeline.predict(X_test)
        precision = precision_score(y_test, y_pred )
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy_score":accuracy,
            "confusion_matrix": cm
        }
        print(f"\n{'=' * 50}")
        print(f"Model: {name}")
        print(f"{'=' * 50}")
        print("\nclassification report :" )
        print(classification_report( y_test, y_pred))
    
        print("Confusion matrix:")
        print(cm)

    return results


def cross_validation(df):
    """
    Cross-validate the models to compare their performance
    across multiple folds of the training data.
    """

    # Split the dataset into training and testing data
    numerical_features = ["age", "transaction_amount", "account_balance"]
    categorical_features = ["gender", "state", "city","bank_branch","account_type", "transaction_date","transaction_device","transaction_location","device_type","transaction_currency"]
    X = df[categorical_features + numerical_features]
    y = df["is_fraud"]


    # Use stratified folds to preserve the class distribution
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_results = []

    for name, model in get_models().items():

        # Create a fresh preprocessing and modelling pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocess_data()),
            ("model", model)
        ])

        # Evaluate the model across five folds
        cross_validation_results = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        cv_results.append({
            "model": name,
            "f1_mean": cross_validation_results.mean(),
            "f1_std": cross_validation_results.std()
        })

    # Return all models ranked by their mean F1 score
    cv_results_df = pd.DataFrame(cv_results)

    return cv_results_df.sort_values(
        by="f1_mean",
        ascending=False
    ).reset_index(drop=True)


    







if __name__ == "__main__":
    df = clean_data(ingest_data())

    print("\n" + "=" * 60)
    print("BANK FRAUD DETECTION - MODEL EVALUATION")
    print("=" * 60)

    evaluation_results = evaluate_data(df)

   

    print("\n" + "=" * 60)
    print("BANK FRAUD DETECTION - CROSS-VALIDATION")
    print("=" * 60)

    cv_results = cross_validation(df)


    print("\nProcess completed.")



    
