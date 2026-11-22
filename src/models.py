"""Modelos y pipeline"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data import SEED, make_scaler


def make_models(weighted=False, spw=1.0):
    cw = "balanced" if weighted else None
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight=cw, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=cw,
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100,
                                 scale_pos_weight=spw if weighted else 1.0, n_jobs=-1,
                                 eval_metric="logloss", random_state=SEED),
    }


def make_pipeline(model, smote=False):
    steps = [("scaler", make_scaler())]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    return Pipeline(steps + [("model", model)])