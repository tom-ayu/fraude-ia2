"""Métricas: validación cruzada (train) y evaluación final (test)"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from data import SEED

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def metrics(y, p, t=0.5):
    pred = (p >= t).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p)}


def cv_oof(pipe, X, y, k=5):
    """Probabilidades out-of-fold en train y los índices de cada fold"""
    folds = list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y))
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = clone(pipe).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
    return oof, folds


def cv_report(oof, y, folds, t=0.5):
    rows = [metrics(y.iloc[va], oof[va], t) for _, va in folds]
    out = {}
    for m in rows[0]:
        vals = [r[m] for r in rows]
        out[m], out[f"{m}_std"] = np.mean(vals), np.std(vals)
    return out


def best_threshold(oof, y):
    """Threshold que maximiza F1 sobre las predicciones out-of-fold"""
    return max(THRESHOLDS, key=lambda t: f1_score(y, (oof >= t).astype(int), zero_division=0))


def test_report(pipe, X, y, t=0.5):
    p = pipe.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
    return {**metrics(y, p, t), "tn": tn, "fp": fp, "fn": fn, "tp": tp}