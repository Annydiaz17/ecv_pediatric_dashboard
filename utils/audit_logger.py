"""
Sistema de auditoría para registrar predicciones realizadas.
Adaptado para el modelo LogisticRegression (sin presión diastólica, con peso_kg).
"""
import os
import pandas as pd
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_PATH = os.path.join(_BASE_DIR, "data", "audit_log.csv")

AUDIT_COLUMNS = [
    "timestamp", "sexo", "edad", "peso_kg",
    "pa_sistolica", "frecuencia_cardiaca",
    "colesterol_mgdl", "probabilidad", "clasificacion"
]


def log_prediction(sexo, edad, peso_kg, pa_sistolica,
                   frecuencia_cardiaca, colesterol_mgdl,
                   probabilidad, clasificacion, **kwargs):
    """Registra una predicción en el log de auditoría."""
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sexo": sexo,
        "edad": edad,
        "peso_kg": peso_kg,
        "pa_sistolica": pa_sistolica,
        "frecuencia_cardiaca": frecuencia_cardiaca,
        "colesterol_mgdl": colesterol_mgdl,
        "probabilidad": probabilidad,
        "clasificacion": clasificacion
    }

    new_row = pd.DataFrame([record])

    if os.path.exists(AUDIT_PATH):
        new_row.to_csv(AUDIT_PATH, mode="a", header=False, index=False)
    else:
        new_row.to_csv(AUDIT_PATH, index=False)


def get_audit_log():
    """Recupera el log de auditoría completo."""
    if not os.path.exists(AUDIT_PATH):
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    try:
        df = pd.read_csv(AUDIT_PATH)
        if df.empty:
            return pd.DataFrame(columns=AUDIT_COLUMNS)
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=AUDIT_COLUMNS)


def get_audit_stats():
    """Calcula estadísticas del log de auditoría."""
    df = get_audit_log()

    if df.empty:
        return {
            "total_predictions": 0,
            "high_risk_count": 0,
            "low_risk_count": 0,
            "avg_probability": 0,
        }

    return {
        "total_predictions": len(df),
        "high_risk_count": int((df["clasificacion"] == "Alto Riesgo").sum()),
        "low_risk_count": int((df["clasificacion"] == "Bajo Riesgo").sum()),
        "avg_probability": round(float(df["probabilidad"].mean()), 4),
    }
