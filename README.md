# Detección de fraude bancario: técnicas de balanceo × modelos

Proyecto de Inteligencia Artificial 2 (UDLA). Compara baseline, SMOTE, class weights y
ajuste de threshold en Logistic Regression, Random Forest, XGBoost y un MLP (Keras),
sobre el [Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) de Kaggle.

## Prerrequisitos
- Python 3.10 – 3.13 con `pip` (verificar con `python --version`)
- ~3 GB de disco libre (TensorFlow pesa ~350 MB) y conexión a internet para la instalación
- Acceso al dataset de Kaggle

## Instalación
```
python -m venv .venv
.venv\Scripts\activate # en Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Si falla, usar las versiones exactas: `pip install -r requirements-lock.txt`.

## Datos
Descargar `creditcard.csv` y guardarlo en `data/` (no se versiona en git).

## Verificación
```
python src/check_setup.py data/creditcard.csv
```
Debe terminar con `TODO LISTO`.

## Qué se instala
| Librería | Uso |
|---|---|
| numpy, pandas, scipy | Datos y cálculo numérico |
| scikit-learn | Modelos clásicos, métricas, validación cruzada |
| imbalanced-learn | SMOTE y pipelines con remuestreo |
| xgboost | Gradient boosting |
| tensorflow-cpu, keras | Red neuronal (MLP) en CPU |
| matplotlib | Gráficos |
| joblib | Paralelismo y guardado de modelos |

## Estructura
```
data/      ruta para el dataset
src/       código fuente
results/   tablas y figuras generadas (no se suben al repo)
```
Todos los experimentos usan `random_state=42`.
