import pandas as pd
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from xgboost import XGBClassifier
from data_preprocessing import split_data,preprocess_data
from data_cleaning import clean_data
from data_ingestion import ingest_data
from scipy.stats import randint,loguniform,uniform
from sklearn.pipeline import Pipeline
df = clean_data(ingest_data())
X_train, X_test, y_train, y_test = split_data(df)

def tune_xgboost(X_train,y_train):

    """"
    Tuning Xgboost classifier using randomized search cv which returns 
    the best model 
    """

     #Creating a pipeline 
    pipeline = Pipeline([
       ( "preprocessor", preprocess_data()),
       ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1   
       ))
    ])


    #Search across complexity and class weighting with parameter distribution
    param_distributions = {
        "model__n_estimators": randint(100, 501),
        "model__max_depth": randint(3, 9),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__min_child_weight": randint(1, 11),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__gamma": uniform(0, 5),
        "model__reg_alpha": loguniform(1e-4, 10),
        "model__reg_lambda": loguniform(1e-2, 100),
        
    }

    #Cross Validation 
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    #Carrying out the Randomised search
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=42,
        verbose=2,
        error_score="raise",
        return_train_score=True
    )

    #Fitting the Randomizartion into the values 
    search.fit(X_train, y_train)

    # Collect the tuning results.
    search_results = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
        [[
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "params"
        ]]
        .reset_index(drop=True)
    )

    print("\nBEST XGBOOST PARAMETERS")
    print(search.best_params_)

    print("\nBEST CROSS-VALIDATED AVERAGE PRECISION")
    print(f"{search.best_score_:.4f}")

    print("\nTOP RANDOMIZED SEARCH RESULTS")
    print(search_results.head(10).to_string(index=False))

    best_model = search.best_estimator_

    return best_model,search 


if __name__ ==  "__main__":
    tune_xgboost(X_train,y_train)