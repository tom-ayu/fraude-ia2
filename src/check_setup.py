"""Verifica el entorno: python src/check_setup.py [ruta_csv]"""
import sys
from importlib import import_module

LIBS = ["numpy", "pandas", "sklearn", "imblearn", "xgboost",
        "tensorflow", "keras", "matplotlib", "joblib"]

ok = True
print(f"Python {sys.version.split()[0]}")
for name in LIBS:
    try:
        print(f"[OK]    {name} {import_module(name).__version__}")
    except Exception as e:
        ok = False
        print(f"[FALTA] {name}: {e}")

if len(sys.argv) > 1:  #comprobar que un CSV se puede leer
    try:
        import pandas as pd
        df = pd.read_csv(sys.argv[1])
        print(f"[OK]    {sys.argv[1]}: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    except Exception as e:
        ok = False
        print(f"[ERROR] {sys.argv[1]}: {e}")

print("TODO LISTO" if ok else "HAY PROBLEMAS")
sys.exit(0 if ok else 1)
