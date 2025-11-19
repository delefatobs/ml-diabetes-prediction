import os
import pickle
from itertools import product

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
)

import xgboost as xgb


DATA_PATH = "data/diabetes.csv"
MODEL_PATH =  "models/trained_model.pkl"
TARGET_COL = "diabetes"


def load_and_clean_data():
    """Load dataset, drop duplicates, normalize categorical columns, drop missing."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Remove duplicates
    df = df.drop_duplicates()

    # Normalize string columns
    for col in ["gender", "smoking_history"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df = df.dropna()

    return df


# --------------------------------------------------------------
# Evaluation Helpers (same as notebook)
# --------------------------------------------------------------

def eval_sklearn_model(model, Xv, yv):
    proba = model.predict_proba(Xv)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "f1": f1_score(yv, preds),
        "roc_auc": roc_auc_score(yv, proba),
        "precision": precision_score(yv, preds),
        "recall": recall_score(yv, preds),
    }


def eval_xgb_model(model, dval, yv):
    proba = model.predict(dval)
    preds = (proba >= 0.5).astype(int)
    return {
        "f1": f1_score(yv, preds),
        "roc_auc": roc_auc_score(yv, proba),
        "precision": precision_score(yv, preds),
        "recall": recall_score(yv, preds),
    }


# --------------------------------------------------------------
# TRAINING PIPELINE
# --------------------------------------------------------------

def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data()

    # Target & features
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    # Train/val/test split: 60/20/20
    df_train_full, df_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    df_train, df_val, y_train, y_val = train_test_split(
        df_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    # DictVectorizer encoding
    dv = DictVectorizer(sparse=False)

    X_train_enc = dv.fit_transform(df_train.to_dict(orient="records"))
    X_val_enc = dv.transform(df_val.to_dict(orient="records"))
    X_test_enc = dv.transform(df_test.to_dict(orient="records"))

    # DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_enc, label=y_train)
    dval = xgb.DMatrix(X_val_enc, label=y_val)
    dtest = xgb.DMatrix(X_test_enc, label=y_test)

    # ----------------------------------------------------------
    # Baseline models
    # ----------------------------------------------------------

    print("Training baseline Logistic Regression...")
    log_reg = LogisticRegression(max_iter=2000, n_jobs=-1)
    log_reg.fit(X_train_enc, y_train)

    print("Training baseline Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_enc, y_train)

    # Baseline XGBoost
    print("Training baseline XGBoost...")
    xgb_params_base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    baseline_xgb = xgb.train(
        params=xgb_params_base,
        dtrain=dtrain,
        num_boost_round=200,
        evals=[(dval, "validation")],
        verbose_eval=False,
    )

    # Collect metrics
    scores = {
        "Logistic Regression": eval_sklearn_model(log_reg, X_val_enc, y_val),
        "Random Forest": eval_sklearn_model(rf, X_val_enc, y_val),
        "Baseline XGBoost": eval_xgb_model(baseline_xgb, dval, y_val),
    }

    # ----------------------------------------------------------
    # Hyperparameter Tuning (grid search)
    # ----------------------------------------------------------

    print("Tuning XGBoost...")

    eta_values = [0.01, 0.1, 0.3]
    depth_values = [3, 6, 10]
    rounds_values = [100, 200, 400]

    best_f1 = -1
    best_xgb = None
    best_params = None
    best_metrics = None

    for eta, depth, rounds in product(eta_values, depth_values, rounds_values):
        params = xgb_params_base.copy()
        params["eta"] = eta
        params["max_depth"] = depth

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=rounds,
            evals=[(dval, "validation")],
            verbose_eval=False,
        )

        metrics = eval_xgb_model(model, dval, y_val)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_xgb = model
            best_params = {"eta": eta, "max_depth": depth, "num_boost_round": rounds}
            best_metrics = metrics

    scores["Tuned XGBoost"] = best_metrics

    # ----------------------------------------------------------
    # Model Selection
    # ----------------------------------------------------------

    best_model_name = max(
        scores.keys(), key=lambda m: (scores[m]["f1"], scores[m]["roc_auc"])
    )

    if best_model_name == "Tuned XGBoost":
        final_model = best_xgb
    elif best_model_name == "Baseline XGBoost":
        final_model = baseline_xgb
    elif best_model_name == "Random Forest":
        final_model = rf
    else:
        final_model = log_reg

    print(f"\nBest model: {best_model_name}")
    print(scores[best_model_name])

    # ----------------------------------------------------------
    # Test Evaluation
    # ----------------------------------------------------------

    print("\nEvaluating on test set...")

    if "XGBoost" in best_model_name:
        proba_test = final_model.predict(dtest)
    else:
        proba_test = final_model.predict_proba(X_test_enc)[:, 1]

    preds_test = (proba_test >= 0.5).astype(int)

    print("Test F1:", f1_score(y_test, preds_test))
    print("Test ROC AUC:", roc_auc_score(y_test, proba_test))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds_test))

    # ----------------------------------------------------------
    # Save model + DictVectorizer
    # ----------------------------------------------------------

    artefacts = {
        "dv": dv,
        "model": final_model,
        "best_model_name": best_model_name,
        "validation_scores": scores,
        "best_params": best_params,
    }

    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(artefacts, f_out)

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
