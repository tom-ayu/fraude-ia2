"""Carga, muestreo y split temporal de IEEE-CIS."""
from pathlib import Path

import pandas as pd

SEED = 42
ID, TIME, TARGET = "TransactionID", "TransactionDT", "isFraud"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CAT_COLS = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            *(f"M{i}" for i in range(1, 10)),
            *(f"id_{i}" for i in (12, 15, 16, 23, *range(27, 32), *range(33, 39))),
            "DeviceType", "DeviceInfo"}


def _dtype(col):
    if col in CAT_COLS:
        return "category"
    if col in (ID, TIME):
        return "int32"
    return "int8" if col == TARGET else "float32"


def _read(name, data_dir):
    path = data_dir / f"{name}.csv"
    cols = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: _dtype(c) for c in cols})


def load_data(sample_frac=0.2, data_dir=DATA_DIR):
    """Une transacciones e identidad, quita duplicados y devuelve la muestra estratificada ordenada por TransactionDT."""
    tr = _read("train_transaction", data_dir)
    idn = _read("train_identity", data_dir)
    idn_ids = idn[ID]
    df = tr.merge(idn, on=ID, how="left")
    del tr, idn
    df = df[~df.drop(columns=ID).duplicated()].copy()
    df["has_identity"] = df[ID].isin(idn_ids).astype("int8")
    if sample_frac < 1:
        df = df.groupby(TARGET).sample(frac=sample_frac, random_state=SEED)
    return df.sort_values(TIME, kind="stable").drop(columns=ID).reset_index(drop=True)


def split_data(df, train_frac=0.70):
    """Split temporal: el tramo más antiguo es train y el más reciente es test. TransactionDT no queda en X."""
    k = int(len(df) * train_frac)
    X, y = df.drop(columns=[TARGET, TIME]), df[TARGET]
    return X.iloc[:k], X.iloc[k:], y.iloc[:k], y.iloc[k:]


def column_types(X):
    """Devuelve (categóricas, numéricas) según el dtype de X."""
    return X.select_dtypes("category").columns.tolist(), X.select_dtypes("number").columns.tolist()
