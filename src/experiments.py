"""Compara técnicas de balanceo × modelos. Uso: python src/experiments.py"""
import time
from pathlib import Path

import pandas as pd

from data import load_data, split_data
from evaluate import best_threshold, cv_oof, cv_report, test_report
from models import make_models, make_pipeline

OUT = Path(__file__).resolve().parents[1] / "results"
X_tr, X_te, y_tr, y_te = split_data(load_data())
spw = (y_tr == 0).sum() / (y_tr == 1).sum()

TECHNIQUES = {  # nombre: (modelos, usar SMOTE)
    "baseline": (make_models(), False),
    "smote": (make_models(), True),
    "class_weights": (make_models(weighted=True, spw=spw), False),
}

cv_rows, test_rows = [], []
for tech, (models, smote) in TECHNIQUES.items():
    for name, model in models.items():
        t0 = time.time()
        pipe = make_pipeline(model, smote)
        oof, folds = cv_oof(pipe, X_tr, y_tr)
        cv_rows.append({"technique": tech, "model": name, **cv_report(oof, y_tr, folds)})
        fitted = pipe.fit(X_tr, y_tr)
        test_rows.append({"technique": tech, "model": name, "threshold": 0.5,
                          **test_report(fitted, X_te, y_te)})
        if tech == "baseline":  # threshold elegido con predicciones out-of-fold, nunca con test
            t = float(best_threshold(oof, y_tr))
            test_rows.append({"technique": "threshold", "model": name, "threshold": t,
                              **test_report(fitted, X_te, y_te, t)})
        print(f"{tech} {name}: {time.time() - t0:.0f}s")

keys = ["technique", "model"]
cv = pd.DataFrame(cv_rows).set_index(keys)
test = pd.DataFrame(test_rows).set_index(keys)
OUT.mkdir(exist_ok=True)
cv.to_csv(OUT / "experiments_cv.csv")
test.to_csv(OUT / "experiments_test.csv")

pd.set_option("display.width", 250)
print("\nCV 5 folds (train, threshold 0.5):\n", cv[["precision", "recall", "f1", "average_precision"]].round(3))
print("\nTest:\n", test[["threshold", "precision", "recall", "f1", "average_precision", "fp", "fn"]].round(3))