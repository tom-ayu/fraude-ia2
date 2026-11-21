"""Valida los datos, el split y el escalado. Uso: python src/validate_data.py"""
from pandas.util import hash_pandas_object # pylint: disable=no-name-in-module

from data import TARGET, SCALE_COLS, load_data, make_scaler, split_data

raw = load_data(drop_duplicates=False)
print(f"Original: {len(raw):,} filas | Nulos: {int(raw.isna().sum().sum())} | "
      f"Fraudes: {int(raw[TARGET].sum())} | Duplicados: {int(raw.duplicated().sum()):,}")

df = load_data()
print(f"Sin duplicados: {len(df):,} filas | Fraudes: {int(df[TARGET].sum())} ({df[TARGET].mean():.3%})")

v = df[[f"V{i}" for i in range(1, 29)]]
print(f"V1-V28: media máx. en valor absoluto {v.mean().abs().max():.3f} | desv. entre {v.std().min():.2f} y {v.std().max():.2f}")

X_tr, X_te, y_tr, y_te = split_data(df)
for name, y in [("Train", y_tr), ("Test", y_te)]:
    print(f"{name}: {len(y):,} filas | fraudes {int(y.sum())} ({y.mean():.3%})")

solapadas = hash_pandas_object(X_te, index=False).isin(set(hash_pandas_object(X_tr, index=False))).sum()
print(f"Filas de test idénticas a alguna de train: {int(solapadas)}")

scaler = make_scaler().fit(X_tr)
Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
print("Time y Amount escalados (media, desv.):")
for c in SCALE_COLS:
    print(f"  {c}: train ({Z_tr[c].mean():.3f}, {Z_tr[c].std():.3f}) | test ({Z_te[c].mean():.3f}, {Z_te[c].std():.3f})")
