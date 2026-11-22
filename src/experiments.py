"""Compara técnicas de balanceo x modelos. Uso: python src/experiments.py [--sample-frac 0.2] [--gpu]"""
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn import set_config

from data import SEED, column_types, load_data, split_data
from evaluate import FOLDS, best_threshold, cv_oof, cv_report, plot_pr_curves, plot_thresholds, test_report
from models import TOP_K, make_models, make_pipeline, sparse_columns

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=0.2, help="fracción estratificada del dataset (1.0 = completo)")
    ap.add_argument("--gpu", action="store_true", help="XGBoost en GPU (los resultados difieren un poco de CPU)")
    ap.add_argument("--mlp", action="store_true", help="incluye el MLP de Keras")
    args = ap.parse_args()
    set_config(print_changed_only=False)
    tag = f"{args.sample_frac:g}"

    t_start = time.time()
    df = load_data(args.sample_frac)
    X_tr, X_te, y_tr, y_te = split_data(df)
    del df
    drop = sparse_columns(X_tr)
    X_tr, X_te = X_tr.drop(columns=drop), X_te.drop(columns=drop)
    cat, num = column_types(X_tr)
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    print(f"train {len(X_tr):,} | test {len(X_te):,} | columnas eliminadas (>90% nulos): {len(drop)} | "
          f"categóricas {len(cat)} | numéricas {len(num)} | peso de clase {spw:.1f}")

    techniques = {  # nombre: (modelos, usar SMOTENC)
        "baseline": (make_models(gpu=args.gpu, mlp=args.mlp), False),
        "smotenc": (make_models(gpu=args.gpu, mlp=args.mlp), True),
        "class_weights": (make_models(weighted=True, spw=spw, gpu=args.gpu, mlp=args.mlp), False),
    }

    cv_rows, test_rows, probs, oofs = [], [], {}, {}
    for tech, (models, smote) in techniques.items():
        for name, model in models.items():
            pipe = make_pipeline(model, cat, num, smote)
            t0 = time.time()
            oof, folds = cv_oof(pipe, X_tr, y_tr)
            t1 = time.time()
            p = pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
            sec = {"cv_seconds": round(t1 - t0, 1), "fit_seconds": round(time.time() - t1, 1)}
            probs[(tech, name)] = p
            cv_rows.append({"technique": tech, "model": name, **cv_report(oof, y_tr, folds), **sec})
            test_rows.append({"technique": tech, "model": name, "threshold": 0.5, **test_report(p, y_te), **sec})
            if tech == "baseline":  # threshold elegido con predicciones out-of-fold, nunca con test
                oofs[name] = oof
                t = float(best_threshold(oof, y_tr))
                test_rows.append({"technique": "threshold", "model": name, "threshold": t,
                                  **test_report(p, y_te, t), **sec})
            print(f"{tech} {name}: cv {sec['cv_seconds']:.0f}s | fit {sec['fit_seconds']:.0f}s")

    keys = ["technique", "model"]
    cv = pd.DataFrame(cv_rows).set_index(keys)
    test = pd.DataFrame(test_rows).set_index(keys)
    OUT.mkdir(exist_ok=True)
    cv.to_csv(OUT / f"experiments_cv_{tag}.csv")
    test.to_csv(OUT / f"experiments_test_{tag}.csv")
    plot_pr_curves(y_te, probs, OUT / f"pr_curves_{tag}.png")
    plot_thresholds(y_tr, oofs, OUT / f"thresholds_{tag}.png")

    pd.set_option("display.width", 250)
    print(f"\nCV {FOLDS} folds (train, threshold 0.5):\n",
          cv[["precision", "recall", "f1", "average_precision", "cv_seconds"]].round(3))
    print("\nTest (temporal):\n",
          test[["threshold", "precision", "recall", "f1", "average_precision", "fp", "fn", "fit_seconds"]].round(3))

    print(f"\nSemilla {SEED} (muestra, folds, modelos, SMOTENC) | sample_frac {args.sample_frac} | folds {FOLDS} | "
          f"top-k categorías {TOP_K} | gpu {args.gpu}")
    for name, model in techniques["class_weights"][0].items():
        print(f"{name}: {model}")
    print(make_pipeline(model, cat, num, True).named_steps["smote"])
    print(f"Tiempo total: {time.time() - t_start:.0f}s | resultados en {OUT}")


if __name__ == "__main__":
    main()
