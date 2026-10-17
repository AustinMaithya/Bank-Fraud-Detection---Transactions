from data_preprocessing import preprocess_data, split_data
from data_cleaning import clean_data
from data_ingestion import ingest_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
df = ingest_data()
df = clean_data(df)

X_train, X_test, y_train, y_test = split_data(df)
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoosting":GradientBoostingClassifier(),
    "SVC":SVC(),
    "XGBoost": XGBClassifier()
}


def model_train(models):
    """  Model training occurs here where the models are fit into the data """
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = preprocess_data(df)

    for model in models.values():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model",model)
        ])

        pipeline.fit(X_train,y_train)
        y_pred = pipeline.predict(X_test)
        y_probna = pipeline.predict_proba(y_test)



if __name__ == "__main__":
    model_train(preprocessor,X_train, X_test, y_train, y_test,models)
    

