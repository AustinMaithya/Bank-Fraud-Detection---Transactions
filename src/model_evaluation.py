from model_training_and_testing import model_train
from data_cleaning import clean_data
from data_ingestion import ingest_data
from sklearn.metrics import precision_score, recall_score,f1_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.model_selection import cross_val_score
from data_preprocessing import preprocess_data,split_data
from sklearn.pipeline import Pipeline
from model_training_and_testing import model_train
from sklearn.model_selection import StratifiedKFold

df = ingest_data()
df = clean_data(df)

def evaluate_data(df):
    
    """model evaluation of the trained data"""
    trained_models, X_test, y_test = model_train(df)
    results  = {}
   
    for name, pipeline in trained_models.items():

        # Predict class labels and probabilities
        y_pred = pipeline.predict(X_test)


        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
       
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
           
            "confusion_matrix": cm,
        }

        print(f"\n{'=' * 50}")
        print(f"Model: {name}")
        print(f"{'=' * 50}")
        print(classification_report(
            y_test, y_pred, zero_division=0 ))
        
        print("Confusion matrix:")
        print(cm)

    return results


def cross_validation(df):
    """
    Cross-validate the models to compare their performance
    across multiple folds of the training data.
    """

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = split_data(df)

    # Retrieve the models from your training module
    trained_models = model_train(df)

    # Use stratified folds to preserve the class distribution
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, model in trained_models.items():

        # Create a fresh preprocessing and modelling pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocess_data(X_train)),
            ("model", model)
        ])

        # Evaluate the model across five folds
        cross_validation_results = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        results.append({
            "model": name,
            "f1_mean": cross_validation_results.mean(),
            "f1_std": cross_validation_results.std()
        })

    # Return all models ranked by their mean F1 score
    results_df = pd.DataFrame(results)

    return results_df.sort_values(
        by="f1_mean",
        ascending=False
    ).reset_index(drop=True)


    






if __name__ == "__main__":
    df = clean_data(ingest_data())
    results = evaluate_data(df)
    cross_validation(df)
    print("\nModel evaluation and cross validation completed.")



    
