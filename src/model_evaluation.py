from model_training_and_testing import model_train
from data_cleaning import clean_data
from data_ingestion import ingest_data
from sklearn.metrics import precision_score, recall_score,f1_score,classification_report,confusion_matrix,roc_auc_score
from data_preprocessing import preprocess_data,split_data
df = ingest_data()
df = clean_data(df)

def evaluate_data(df):
    
    """model evaluation of the trained data"""
    trained_models, X_test, y_test = model_train(df)
    results  = {}
   
    for name, pipeline in trained_models.items():

        # Predict class labels and probabilities
        y_pred = pipeline.predict(X_test)

        # Requires a classifier that supports predict_proba
        y_proba = pipeline.predict_proba(y_test)[:, 1]

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }

        print(f"\n{'=' * 50}")
        print(f"Model: {name}")
        print(f"{'=' * 50}")
        print(classification_report(
            y_test, y_pred, zero_division=0
        ))
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("Confusion matrix:")
        print(cm)

    return results


if __name__ == "__main__":
    df = clean_data(ingest_data())
    results = evaluate_data(df)
    print("\nModel evaluation completed.")



    
