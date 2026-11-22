"""Perfil de IEEE-CIS. Uso: python src/profile_ieee.py [--full]"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
REPORT = ROOT / "perfil_ieee.txt"
FILES = ["train_transaction", "train_identity", "test_transaction", "test_identity", "sample_submission"]
LIGHT_COLS = ["TransactionID", "TransactionDT", "isFraud", "ProductCD"]
DAY = 86400

lines = []


def out(text=""):
    print(text)
    lines.append(text)


def read(name, **kw):
    return pd.read_csv(D / f"{name}.csv", **kw)


def dt_range(dt):
    return f"{dt.min()} a {dt.max()} ({(dt.max() - dt.min()) / DAY:.0f} días)"


def rate_table(df, by):
    for key, g in df.groupby(by, observed=True):
        out(f"  {key}: {len(g):,} filas | fraude {g.isFraud.mean():.2%}")


def section_files():
    out("## 1. Los 5 archivos")
    cols = {f: read(f, nrows=0).columns.tolist() for f in FILES}
    ids = {f: read(f, usecols=["TransactionID"]).TransactionID for f in FILES}
    for f in FILES:
        out(f"{f}: {len(ids[f]):,} filas | {len(cols[f])} columnas | tiene isFraud: {'isFraud' in cols[f]}")
    out(f"Columnas de train_transaction ausentes en test_transaction: "
        f"{sorted(set(cols['train_transaction']) - set(cols['test_transaction']))}")
    tr, te = set(ids["train_transaction"]), set(ids["test_transaction"])
    out(f"IDs en común train/test: {len(tr & te)}")
    out(f"IDs de sample_submission iguales a los de test_transaction: {set(ids['sample_submission']) == te}")
    out(f"Valores de isFraud en sample_submission: {read('sample_submission', usecols=['isFraud']).isFraud.unique().tolist()}")
    for kind in ["train", "test"]:
        cov = ids[f"{kind}_transaction"].isin(ids[f"{kind}_identity"]).mean()
        out(f"Cobertura de identidad en {kind}: {cov:.1%}")
    for kind in ["train", "test"]:
        dt = read(f"{kind}_transaction", usecols=["TransactionDT"]).TransactionDT
        out(f"TransactionDT {kind}: {dt_range(dt)}")
    return ids


def section_train_light(ids):
    out()
    out("## 2. Train (ligero)")
    tr = read("train_transaction", usecols=LIGHT_COLS)
    tr["has_identity"] = tr.TransactionID.isin(ids["train_identity"])
    out(f"Filas: {len(tr):,} | fraude: {tr.isFraud.sum():,} ({tr.isFraud.mean():.2%}) | "
        f"TransactionID duplicados: {tr.TransactionID.duplicated().sum()}")
    out("Tasa de fraude según has_identity:")
    rate_table(tr, "has_identity")
    out("Tasa de fraude por bloques de 30 días de TransactionDT:")
    rate_table(tr.assign(bloque=tr.TransactionDT // (30 * DAY)), "bloque")
    out("Tasa de fraude por ProductCD:")
    rate_table(tr, "ProductCD")
    tr = tr.sort_values("TransactionDT")
    k = int(len(tr) * 0.7)
    out(f"Fraude en el 70% inicial: {tr.isFraud.iloc[:k].mean():.2%} | en el 30% final: {tr.isFraud.iloc[k:].mean():.2%}")


def section_full():
    out()
    out("## 3. Dataset unido (--full)")
    df = read("train_transaction").merge(read("train_identity"), on="TransactionID", how="left")
    out(f"Forma: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    out(f"Filas duplicadas (sin TransactionID): {df.drop(columns='TransactionID').duplicated().sum():,}")
    cat = df.select_dtypes(exclude="number")
    out(f"Categóricas ({cat.shape[1]}), valores distintos:")
    out(cat.nunique().sort_values(ascending=False).to_string())
    miss = df.isna().mean()
    out(f"Columnas con más de 50% de nulos: {(miss > 0.5).sum()} | más de 90%: {(miss > 0.9).sum()} | sin nulos: {(miss == 0).sum()}")
    out(f"Columnas con más de 90% de nulos: {miss[miss > 0.9].index.tolist()}")


def main():
    full = "--full" in sys.argv[1:]
    ids = section_files()
    section_train_light(ids)
    if full:
        section_full()
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPerfil guardado en {REPORT}")


if __name__ == "__main__":
    main()
