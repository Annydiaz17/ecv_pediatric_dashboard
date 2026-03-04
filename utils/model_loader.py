"""
Cargador del modelo LogisticRegression y scaler del notebook SEMMA.
Usa modelo_final_riesgo_pediatrico.joblib + scaler_riesgo_pediatrico.joblib.
"""
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

_model = None
_scaler = None
_metrics = None
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Features del modelo (orden del scaler)
FEATURES_MODELO = [
    "edad", "genero", "peso_kg", "pa_sistolica",
    "frecuencia_cardiaca", "colesterol_mgdl"
]

FEATURE_LABELS = {
    "edad": "Edad (años)",
    "genero": "Género",
    "peso_kg": "Peso (kg)",
    "pa_sistolica": "PA Sistólica (mmHg)",
    "frecuencia_cardiaca": "Frecuencia Cardíaca (lpm)",
    "colesterol_mgdl": "Colesterol (mg/dL)",
}


def get_model():
    """Carga y cachea el modelo LogisticRegression."""
    global _model
    if _model is None:
        model_path = os.path.join(
            _BASE_DIR, "models", "modelo_final_riesgo_pediatrico.joblib"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path}. "
                "Se requiere: modelo_final_riesgo_pediatrico.joblib"
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _model = joblib.load(model_path)
        print(f"[OK] Modelo cargado: {type(_model).__name__}")
    return _model


def get_scaler():
    """Carga y cachea el StandardScaler."""
    global _scaler
    if _scaler is None:
        scaler_path = os.path.join(
            _BASE_DIR, "models", "scaler_riesgo_pediatrico.joblib"
        )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler no encontrado en {scaler_path}. "
                "Se requiere: scaler_riesgo_pediatrico.joblib"
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _scaler = joblib.load(scaler_path)
        print(f"[OK] Scaler cargado: {type(_scaler).__name__}")
    return _scaler


def get_metrics():
    """Carga y cachea las métricas del modelo."""
    global _metrics
    if _metrics is None:
        metrics_path = os.path.join(_BASE_DIR, "models", "model_metrics.json")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"Métricas no encontradas en {metrics_path}."
            )
        with open(metrics_path, "r", encoding="utf-8") as f:
            _metrics = json.load(f)
    return _metrics


def get_feature_names():
    """Retorna la lista de features del modelo."""
    return FEATURES_MODELO.copy()


def _get_feature_importances(model=None):
    """
    Extrae feature importances del modelo.
    Para LogisticRegression, usa |coef_| normalizado.
    """
    if model is None:
        model = get_model()

    if hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).flatten()
        total = coefs.sum()
        if total > 0:
            return {
                name: round(float(c / total), 4)
                for name, c in zip(FEATURES_MODELO, coefs)
            }

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return {
            name: round(float(v), 4)
            for name, v in zip(FEATURES_MODELO, importances)
        }

    # Fallback
    n = len(FEATURES_MODELO)
    return {name: round(1.0 / n, 4) for name in FEATURES_MODELO}


def predict_risk(genero, edad, peso_kg, pa_sistolica,
                 frecuencia_cardiaca, colesterol_mgdl):
    """
    Predicción de riesgo cardiovascular individual.
    Aplica scaler antes de predecir (como en el notebook SEMMA).

    Args:
        genero: 0 (Femenino) o 1 (Masculino)
        edad: años
        peso_kg: peso en kg
        pa_sistolica: presión sistólica mmHg
        frecuencia_cardiaca: latidos por minuto
        colesterol_mgdl: colesterol en mg/dL

    Returns:
        dict con probabilidad, clasificación, nivel, importancias
    """
    model = get_model()
    scaler = get_scaler()

    # Convertir genero string a numérico si es necesario
    if isinstance(genero, str):
        genero = 1 if genero.strip().lower() in ("masculino", "m", "1") else 0

    # Orden de features del scaler
    X = np.array([[
        float(edad), float(genero), float(peso_kg),
        float(pa_sistolica), float(frecuencia_cardiaca), float(colesterol_mgdl)
    ]])

    # 1) Escalar
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_scaled = scaler.transform(X)

    # 2) Predecir
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        probabilidad = model.predict_proba(X_scaled)[0][1]

    clasificacion = "Alto Riesgo" if probabilidad >= 0.5 else "Bajo Riesgo"

    # Nivel para semáforo
    if probabilidad < 0.3:
        nivel = "bajo"
    elif probabilidad < 0.6:
        nivel = "moderado"
    else:
        nivel = "alto"

    importances = _get_feature_importances(model)

    return {
        "probabilidad": round(float(probabilidad), 4),
        "porcentaje": round(float(probabilidad) * 100, 1),
        "clasificacion": clasificacion,
        "nivel": nivel,
        "feature_importances": importances,
        "features_input": dict(zip(FEATURES_MODELO, X[0].tolist())),
    }
