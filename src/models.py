"""Preprocesamiento, modelos y pipeline."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from data import SEED

TOP_K = 15
MAX_NULL = 0.90


class TopKCategories(BaseEstimator, TransformerMixin):
    """Conserva las k categorías más frecuentes del train; el resto pasa a "otros" y el nulo a "faltante"."""

    def __init__(self, k=TOP_K):
        self.k = k

    def fit(self, X, y=None):
        self.top_ = {c: set(X[c].value_counts().index[:self.k]) for c in X.columns}
        return self

    def transform(self, X):
        out = {}
        for c in X.columns:
            s = X[c].astype(object)
            out[c] = np.where(s.isna(), "faltante", np.where(s.isin(self.top_[c]), s, "otros"))
        return pd.DataFrame(out, index=X.index, dtype=object)


def sparse_columns(X_train, max_null=MAX_NULL):
    """Columnas con más de max_null de nulos en train. Se eliminan de train y test antes de armar el pipeline."""
    return X_train.columns[X_train.isna().mean() > max_null].tolist()


def make_preprocessor(cat_cols, num_cols):
    """Categóricas primero (top-k y OrdinalEncoder), luego numéricas (mediana y escalado)."""
    cat = Pipeline([("top", TopKCategories()),
                    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("cat", cat, cat_cols), ("num", num, num_cols)])


class KerasMLP(ClassifierMixin, BaseEstimator):
    """MLP de dos capas ocultas; con weighted=True pondera la clase positiva con n_neg/n_pos del train recibido."""

    def __init__(self, weighted=False, epochs=20, batch_size=512, dropout=0.3):
        self.weighted = weighted
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

    def fit(self, X, y):
        import keras

        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        X, y = np.asarray(X, dtype="float32"), np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.model_ = keras.Sequential([
            keras.layers.Input((X.shape[1],)),
            keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(self.dropout),
            keras.layers.Dense(32, activation="relu"), keras.layers.Dropout(self.dropout),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        self.model_.compile(optimizer="adam", loss="binary_crossentropy")
        cw = {0: 1.0, 1: float((y == 0).sum() / (y == 1).sum())} if self.weighted else None
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, class_weight=cw, verbose=0)
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), batch_size=4096, verbose=0).ravel()
        return np.column_stack([1 - p, p])


def make_models(weighted=False, spw=1.0, gpu=False, mlp=False):
    cw = "balanced" if weighted else None
    models = {
        "LogReg": LogisticRegression(max_iter=5000, class_weight=cw, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=cw,
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100,
                                 scale_pos_weight=spw if weighted else 1.0, n_jobs=-1,
                                 device="cuda" if gpu else "cpu",
                                 eval_metric="logloss", random_state=SEED),
    }
    if mlp:
        models["MLP"] = KerasMLP(weighted=weighted)
    return models


def make_pipeline(model, cat_cols, num_cols, smote=False):
    steps = [("pre", make_preprocessor(cat_cols, num_cols))]
    if smote:
        steps.append(("smote", SMOTENC(categorical_features=list(range(len(cat_cols))), random_state=SEED)))
    return Pipeline(steps + [("model", model)])
