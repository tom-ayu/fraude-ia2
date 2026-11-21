"""Baseline sin balanceo. Uso: python src/baseline.py"""
import time
from pathlib import Path

import pandas as pd

from data import load_data, split_data
from evaluate import cv_report, test_report
from models import make_models, make_pipeline

OUT = Path(__file__).resolve().parents[1] / "results"
X_tr, X_te, y_tr, y_te = split_data(load_data())

cv_rows, test_rows = [], []
for name, model in make_models().items():
    t = time.time()
    pipe = make_pipeline(model)
    cv_rows.append({"model": name, **cv_report(pipe, X_tr, y_tr)})
    test_rows.append({"model": name, **test_report(pipe.fit(X_tr, y_tr), X_te, y_te)})
    print(f"{name}: {time.time() - t:.0f}s")

cv = pd.DataFrame(cv_rows).set_index("model")
test = pd.DataFrame(test_rows).set_index("model")
OUT.mkdir(exist_ok=True)
cv.to_csv(OUT / "baseline_cv.csv")
test.to_csv(OUT / "baseline_test.csv")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("\nCV 5 folds (train):\n", cv.round(3))
print("\nTest:\n", test.round(3))