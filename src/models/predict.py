import pickle
import xgboost as xgb
import numpy as np

MODEL_PATH = "models/trained_model.pkl"

class DiabetesModel:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            artifacts = pickle.load(f)

        self.dv = artifacts["dv"]
        self.model = artifacts["model"]
        self.best_name = artifacts["best_model_name"]

    def predict(self, data: dict):
        """
        Predict label + probability for a single patient record.
        """
        X = self.dv.transform([data])

        # XGBoost booster vs sklearn
        if "XGBoost" in self.best_name:
            dmatrix = xgb.DMatrix(X)
            proba = float(self.model.predict(dmatrix)[0])
        else:
            proba = float(self.model.predict_proba(X)[0, 1])

        pred = int(proba >= 0.5)

        return pred, proba
