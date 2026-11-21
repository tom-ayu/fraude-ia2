"""Definición de modelos"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data import SEED, make_scaler


def make_models():
    return {
        "LogReg": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100, n_jobs=-1,
                                 eval_metric="logloss", random_state=SEED),
    }


def make_pipeline(model):
    return Pipeline([("scaler", make_scaler()), ("model", model)])