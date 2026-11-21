"""Métricas: validación cruzada (train) y evaluación final (test)."""
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from data import SEED

SCORING = ["precision", "recall", "f1", "roc_auc", "average_precision"]


def cv_report(pipe, X, y, k=5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
    out = {}
    for m in SCORING:
        out[m] = r[f"test_{m}"].mean()
        out[f"{m}_std"] = r[f"test_{m}"].std()
    return out


def test_report(pipe, X, y, threshold=0.5):
    p = pipe.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}