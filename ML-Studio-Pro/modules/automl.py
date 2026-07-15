import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


#  PROBLEM TYPE DETECTION 
def detect_problem_type(df, target_col):
    y = df[target_col]

    if y.dtype == "object":
        return "classification"

    # numeric but low unique → classification
    if pd.api.types.is_numeric_dtype(y) and y.nunique() < max(10, int(0.05 * len(y))):
        return "classification"

    return "regression"


#  DATA PREPARATION 
def prepare_data(df, target_col):
    df = df.copy()

    # Safety: remove missing rows (should be cleaned earlier)
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    stratify = y if y.nunique() < 20 else None

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )


#  MODEL TRAINING 
def train_models(X_train, X_test, y_train, y_test, problem_type):
    results = []
    trained_models = {}

    if problem_type == "classification":

        models = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000))
            ]),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "SVC": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC())
            ]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier())
            ]),
            "NaiveBayes": GaussianNB()
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
                })

                trained_models[name] = model

            except Exception as e:
                print(f"[ERROR] {name}: {e}")

    else:  # regression

        models = {
            "LinearRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(random_state=42),
            "SVR": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR())
            ])
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append({
                    "Model": name,
                    "R2 Score": round(r2_score(y_test, y_pred), 4),
                    "MAE": round(mean_absolute_error(y_test, y_pred), 4),
                    "RMSE": round(rmse, 4)
                })

                trained_models[name] = model

            except Exception as e:
                print(f"[ERROR] {name}: {e}")

    results_df = pd.DataFrame(results)

    return results_df, trained_models


#  BEST MODEL 
def get_best_model(results_df, problem_type):
    if results_df.empty:
        return None

    if problem_type == "classification":
        best_row = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]
    else:
        best_row = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]

    return best_row["Model"]