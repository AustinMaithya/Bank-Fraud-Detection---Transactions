from data_preprocessing import preprocess_data, split_data
from data_cleaning import clean_data
from data_ingestion import ingest_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
df = ingest_data()
df = clean_data(df)



def model_train(df):
    preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoosting":GradientBoostingClassifier(),
    "LinearSVC":LinearSVC(),
    "XGBoost": XGBClassifier()
        }
    
    """  Model training occurs here where the models are fit into the data """
    trained_models = {}
    for name,model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model",model)
        ])

        pipeline.fit(X_train,y_train)
        trained_models[name] = pipeline
        print(f"{name} training completed.")



        
    return trained_models, X_test, y_test
    
    



if __name__ == "__main__":
    model_train(df)
    trained_models, X_test, y_test = model_train(df)

    print("Model training completed.")

