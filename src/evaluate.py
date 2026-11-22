"""Métricas, validación cruzada (train), evaluación final (test) y gráficos."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from data import SEED

FOLDS = 5
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def metrics(y, p, t=0.5):
    pred = (p >= t).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p)}


def cv_oof(pipe, X, y, k=FOLDS):
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


def test_report(p, y, t=0.5):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
    return {**metrics(y, p, t), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def plot_pr_curves(y, probs, path):
    """Curvas precisión-recall en test: un panel por modelo, una línea por técnica."""
    models = list(dict.fromkeys(m for _, m in probs))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for (tech, m), p in probs.items():
            if m == model:
                prec, rec, _ = precision_recall_curve(y, p)
                ax.plot(rec, prec, label=f"{tech} (AP {average_precision_score(y, p):.3f})")
        ax.axhline(y.mean(), ls=":", color="gray")
        ax.set(title=model, xlabel="recall")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_thresholds(y, oofs, path):
    """Precisión, recall y F1 vs threshold con predicciones out-of-fold; marca el threshold elegido."""
    fig, axes = plt.subplots(1, len(oofs), figsize=(5 * len(oofs), 4), sharey=True, squeeze=False)
    for ax, (model, oof) in zip(axes[0], oofs.items()):
        preds = [(oof >= t).astype(int) for t in THRESHOLDS]
        ax.plot(THRESHOLDS, [precision_score(y, q, zero_division=0) for q in preds], label="precision")
        ax.plot(THRESHOLDS, [recall_score(y, q) for q in preds], label="recall")
        ax.plot(THRESHOLDS, [f1_score(y, q, zero_division=0) for q in preds], label="f1")
        ax.axvline(best_threshold(oof, y), ls="--", color="gray")
        ax.set(title=model, xlabel="threshold")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
