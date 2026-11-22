"""Valida la muestra y el split temporal. Uso: python src/validate_data.py"""
import sys

from data import CAT_COLS, ID, TARGET, TIME, column_types, load_data, split_data

fails = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FALLA'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        fails.append(name)


df = load_data()
X_tr, X_te, y_tr, y_te = split_data(df)
X, y = df.drop(columns=[TARGET, TIME]), df[TARGET]
cat, num = column_types(X)

check("filas de la muestra (esperadas ~118 mil)", 117_500 <= len(df) <= 118_700, f"{len(df):,}")
check("columnas de features (391 transaction + 40 identity + has_identity)", X.shape[1] == 432, str(X.shape[1]))
check("fraude en la muestra cercano a 3.50%", abs(y.mean() - 0.0350) < 0.0005,
      f"{int(y.sum()):,} fraudes ({y.mean():.3%})")
check("tamaño del split 70/30", len(X_tr) == int(len(df) * 0.7) and len(X_tr) + len(X_te) == len(df),
      f"train {len(X_tr):,} | test {len(X_te):,}")
check("fraude en train y test entre 3% y 4%", all(0.03 < v.mean() < 0.04 for v in (y_tr, y_te)),
      f"train {y_tr.mean():.3%} | test {y_te.mean():.3%}")
check("max(TransactionDT train) <= min(TransactionDT test)",
      df[TIME].loc[X_tr.index].max() <= df[TIME].loc[X_te.index].min(),
      f"{df[TIME].loc[X_tr.index].max()} <= {df[TIME].loc[X_te.index].min()}")
check("índices de train y test disjuntos", set(X_tr.index).isdisjoint(X_te.index))
check("sin TransactionID, TransactionDT ni isFraud en las features", not {ID, TIME, TARGET} & set(X.columns))
check("sin filas duplicadas en la muestra", int(df.duplicated().sum()) == 0)
check("31 categóricas, todas conocidas", len(cat) == 31 and set(cat) == CAT_COLS, str(len(cat)))
check("401 numéricas en float32/int8", len(num) == 401 and X[num].dtypes.astype(str).isin(["float32", "int8"]).all(),
      str(len(num)))
check("has_identity binaria y cercana a 24.4%",
      set(X["has_identity"].unique()) <= {0, 1} and 0.23 < X["has_identity"].mean() < 0.26,
      f"{X['has_identity'].mean():.1%}")
check("sin imputar ni transformar (nulos intactos, ninguna estadística calculada en data.py)",
      int(X.isna().sum().sum()) > 0, f"{int(X.isna().sum().sum()):,} nulos")
print(f"Memoria de X: {X.memory_usage(deep=True).sum() / 1e9:.2f} GB")

print("TODO OK" if not fails else f"FALLAS: {len(fails)}")
sys.exit(1 if fails else 0)
