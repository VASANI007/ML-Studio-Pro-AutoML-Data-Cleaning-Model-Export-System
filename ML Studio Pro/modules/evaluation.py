import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score
)


#  VALIDATION 
def _validate_inputs(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")


#  CLASSIFICATION 
def evaluate_classification(y_true, y_pred, y_proba=None):
    try:
        _validate_inputs(y_true, y_pred)

        result = {
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "F1 Score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)
        }

        # ROC AUC (only if probabilities available)
        if y_proba is not None:
            try:
                result["ROC AUC"] = round(roc_auc_score(y_true, y_proba, multi_class="ovr"), 4)
            except:
                result["ROC AUC"] = None

        return result

    except Exception as e:
        print(f"[ERROR] Classification evaluation failed: {e}")
        return {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1 Score": 0,
            "ROC AUC": None
        }


#  REGRESSION 
def evaluate_regression(y_true, y_pred):
    try:
        _validate_inputs(y_true, y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAPE safe (avoid division by zero)
        y_true_safe = np.where(y_true == 0, 1e-8, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        return {
            "R2 Score": round(r2_score(y_true, y_pred), 4),
            "MAE": round(mean_absolute_error(y_true, y_pred), 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4)
        }

    except Exception as e:
        print(f"[ERROR] Regression evaluation failed: {e}")
        return {
            "R2 Score": 0,
            "MAE": 0,
            "RMSE": 0,
            "MAPE": 0
        }


#  UNIFIED 
def evaluate_model(y_true, y_pred, problem_type, y_proba=None):
    if problem_type == "classification":
        return evaluate_classification(y_true, y_pred, y_proba)

    elif problem_type == "regression":
        return evaluate_regression(y_true, y_pred)

    else:
        return {}


#  MULTI MODEL TABLE 
def build_results_table(results_list):
    if not results_list:
        return pd.DataFrame()

    df = pd.DataFrame(results_list)

    # Sort automatically
    if "F1 Score" in df.columns:
        df = df.sort_values(by="F1 Score", ascending=False)

    elif "R2 Score" in df.columns:
        df = df.sort_values(by="R2 Score", ascending=False)

    return df.reset_index(drop=True)


#  BEST MODEL 
def select_best_model(results_df, problem_type):
    if results_df is None or results_df.empty:
        return None

    if problem_type == "classification" and "F1 Score" in results_df.columns:
        return results_df.iloc[0]

    elif problem_type == "regression" and "R2 Score" in results_df.columns:
        return results_df.iloc[0]

    return None