"""Carga, split y escalado del dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "Class"
SCALE_COLS = ["Time", "Amount"]
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"


def load_data(path=DATA_PATH, drop_duplicates=True):
    """Carga el CSV; por defecto elimina filas duplicadas exactas (evita solape train/test)."""
    df = pd.read_csv(path)
    return df.drop_duplicates().reset_index(drop=True) if drop_duplicates else df


def split_data(df, test_size=0.30, seed=SEED):
    """Split estratificado. Devuelve X_train, X_test, y_train, y_test (sin escalar)."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def make_scaler():
    """Escala solo Time y Amount; las demás columnas pasan sin cambios.
    Se usa dentro de un Pipeline para ajustarse solo con train (o con cada fold)."""
    return ColumnTransformer(
        [("scale", StandardScaler(), SCALE_COLS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
