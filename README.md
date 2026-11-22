# Detección de fraude: técnicas de balanceo x modelos

Proyecto de Inteligencia Artificial 2 (UDLA). Compara baseline, SMOTENC, class weights y ajuste de threshold
en Logistic Regression, Random Forest, XGBoost y un MLP (Keras), sobre el dataset
[IEEE-CIS Fraud Detection](https://ieee-dataport.org/documents/ieee-cis-fraud-detection), descargado de IEEE DataPort.

## Prerrequisitos
- Python 3.10 a 3.13 con `pip` (verificar con `python --version`)
- ~3 GB de disco libre (TensorFlow pesa ~350 MB) y conexión a internet para la instalación
- RAM libre: `validate_data.py` y `profile_ieee.py --full` necesitan entre 3 y 5 GB. Con `--sample-frac 1.0` el consumo es mayor y no está medido.

## Instalación
```
python -m venv .venv
.venv\Scripts\activate # en Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Si falla, usar las versiones exactas: `pip install -r requirements-lock.txt`.

## Datos
Descargar el dataset IEEE-CIS Fraud Detection y guardar directamente en `data/` (no en subcarpetas) estos dos archivos, que no se versionan en git:
- `train_transaction.csv`
- `train_identity.csv`

Los archivos `test_*.csv` y `sample_submission.csv` no se usan: no traen `isFraud` y corresponden a un período posterior con etiquetas no publicadas.

## Verificación
```
python src/check_setup.py data/train_transaction.csv
```
Debe terminar con `TODO LISTO`.

## Uso
Perfil del dataset (modo ligero por defecto; `--full` une los dos CSV y necesita ~5 GB de RAM libres). Escribe `perfil_ieee.txt`:
```
python src/profile_ieee.py [--full]
```
Validación de la carga, la muestra y el split temporal:
```
python src/validate_data.py
```
Experimentos (técnica x modelo, CV estratificada de 5 folds en train y evaluación en el test temporal):
```
python src/experiments.py [--sample-frac 0.2] [--mlp] [--gpu]
```
- `--sample-frac`: fracción estratificada del dataset (por defecto 0.2; 1.0 usa todo).
- `--mlp`: incluye el MLP de Keras.
- `--gpu`: XGBoost en GPU (apagado por defecto; los resultados difieren un poco de CPU). TensorFlow se instala solo en CPU.

## Reproducir los resultados
1. Instalar dependencias y copiar los dos CSV a `data/`.
2. `python src/validate_data.py` y comprobar que termina con `TODO OK`.
3. `python src/experiments.py --sample-frac 0.2 --mlp`.

Cada corrida escribe en `results/`, con la fracción de muestra en el nombre: `experiments_cv_<frac>.csv`, `experiments_test_<frac>.csv` (métricas, incluida average precision, y tiempos), `pr_curves_<frac>.png` y `thresholds_<frac>.png`. Al final imprime semillas e hiperparámetros.

## Qué se instala
| Librería | Uso |
|---|---|
| numpy, pandas, scipy | Datos y cálculo numérico |
| scikit-learn | Modelos clásicos, métricas, validación cruzada |
| imbalanced-learn | SMOTENC y pipelines con remuestreo |
| xgboost | Gradient boosting |
| tensorflow-cpu, keras | Red neuronal (MLP) en CPU |
| matplotlib | Gráficos |
| joblib | Paralelismo y guardado de modelos |

## Estructura
```
data/              CSV del dataset (no se versionan)
src/               código fuente
results/           tablas y figuras generadas (no se suben al repo)
perfil_ieee.txt    salida de profile_ieee.py
```
Todos los experimentos usan `random_state=42`; el MLP fija su semilla con `keras.utils.set_random_seed(42)`.
