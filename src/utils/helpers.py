"""
helpers.py

Utility functions for model evaluation and preprocessing.
These are shared across training and prediction modules.
"""

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)
import xgboost as xgb


# -------------------------------------------------------------------
# Evaluation helpers for sklearn models
# -------------------------------------------------------------------
def eval_sklearn_model(model, X_val, y_val):
    """
    Evaluate a scikit-learn model on validation data.
    Returns dictionary with standard metrics.
    """
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return {
        "f1": f1_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, proba),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
    }


# -------------------------------------------------------------------
# Evaluation helpers for XGBoost models
# -------------------------------------------------------------------
def eval_xgb_model(model: xgb.Booster, dval, y_val):
    """
    Evaluate an XGBoost model using DMatrix validation set.
    """
    proba = model.predict(dval)
    preds = (proba >= 0.5).astype(int)

    return {
        "f1": f1_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, proba),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
    }


# -------------------------------------------------------------------
# Helper for converting input dict from FastAPI to model-ready dict
# -------------------------------------------------------------------
def convert_request_to_dict(request_obj):
    """
    Convert a FastAPI Pydantic request model into
    a dictionary record for prediction.

    This keeps prediction and API layers clean.
    """
    return {
        "gender": request_obj.gender,
        "age": request_obj.age,
        "hypertension": request_obj.hypertension,
        "heart_disease": request_obj.heart_disease,
        "smoking_history": request_obj.smoking_history,
        "bmi": request_obj.bmi,
        "HbA1c_level": request_obj.HbA1c_level,
        "blood_glucose_level": request_obj.blood_glucose_level,
    }
