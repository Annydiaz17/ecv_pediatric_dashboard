"""
Generador de datos sintéticos y modelo de ejemplo.
Ejecutar este script para crear el dataset de demostración
y entrenar el modelo RandomForest.

Uso:
    python data/generate_sample_data.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    recall_score, precision_score, f1_score
)
import joblib
import json
from datetime import datetime

# Semilla para reproducibilidad
np.random.seed(42)

# ─── Configuración ───────────────────────────────────────────────
N_SAMPLES = 2000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "models")


def generate_dataset(n=N_SAMPLES):
    """Genera un dataset pediátrico sintético con relaciones clínicas realistas."""
    print(f"[INFO] Generando {n} registros sintéticos...")

    # Variables base
    sexo = np.random.binomial(1, 0.52, n)  # 52% masculino
    edad = np.random.uniform(2, 17, n).round(1)

    # Colesterol: distribución normal con variación por edad
    colesterol_base = 160 + edad * 1.5
    colesterol_total = np.random.normal(colesterol_base, 25, n).clip(100, 300).round(1)

    # Presión sistólica: aumenta con edad
    pas_base = 85 + edad * 1.8
    presion_sistolica = np.random.normal(pas_base, 10, n).clip(70, 160).round(0).astype(int)

    # Presión diastólica: correlacionada con sistólica
    presion_diastolica = (presion_sistolica * 0.6 + np.random.normal(0, 5, n)).clip(40, 100).round(0).astype(int)

    # Frecuencia cardíaca: disminuye con edad
    fc_base = 110 - edad * 2.5
    frecuencia_cardiaca = np.random.normal(fc_base, 10, n).clip(50, 140).round(0).astype(int)

    # ─── Target: riesgo_alto ─────────────────────────────────────
    # Lógica clínica para generar target realista
    logit = (
        -4.0
        + 0.15 * (colesterol_total - 170) / 25
        + 0.3 * (presion_sistolica - 105) / 10
        + 0.15 * (presion_diastolica - 65) / 5
        + 0.1 * (frecuencia_cardiaca - 85) / 10
        + 0.05 * sexo
        + 0.08 * (edad - 10) / 5
    )
    prob = 1 / (1 + np.exp(-logit))
    # Añadir ruido
    prob = np.clip(prob + np.random.normal(0, 0.05, n), 0, 1)
    riesgo_alto = (prob > 0.5).astype(int)

    # Ajustar prevalencia a ~18-22%
    threshold = np.percentile(prob, 80)
    riesgo_alto = (prob > threshold).astype(int)

    df = pd.DataFrame({
        "sexo": sexo,
        "edad": edad,
        "colesterol_total": colesterol_total,
        "presion_sistolica": presion_sistolica,
        "presion_diastolica": presion_diastolica,
        "frecuencia_cardiaca": frecuencia_cardiaca,
        "riesgo_alto": riesgo_alto
    })

    return df


def train_model(df):
    """Entrena un modelo RandomForest y guarda métricas."""
    print("[INFO] Entrenando modelo RandomForest...")

    features = ["sexo", "edad", "colesterol_total",
                "presion_sistolica", "presion_diastolica", "frecuencia_cardiaca"]
    X = df[features]
    y = df["riesgo_alto"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Métricas
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "prevalencia": round(y.mean(), 4),
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "RandomForestClassifier",
        "features": features,
        "feature_importances": dict(zip(features, model.feature_importances_.round(4).tolist()))
    }

    # Modelos comparativos (métricas de referencia)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier

    comparison = []
    alt_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    for name, m in alt_models.items():
        m.fit(X_train, y_train)
        yp = m.predict_proba(X_test)[:, 1]
        ypred = m.predict(X_test)
        comparison.append({
            "modelo": name,
            "roc_auc": round(roc_auc_score(y_test, yp), 4),
            "recall": round(recall_score(y_test, ypred), 4),
            "precision": round(precision_score(y_test, ypred), 4),
            "f1": round(f1_score(y_test, ypred), 4)
        })

    # Agregar RandomForest a la comparación
    comparison.append({
        "modelo": "Random Forest (Final)",
        "roc_auc": metrics["roc_auc"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1_score"]
    })

    metrics["model_comparison"] = comparison

    print(f"[INFO] Métricas del modelo:")
    print(f"  ROC-AUC:    {metrics['roc_auc']}")
    print(f"  PR-AUC:     {metrics['pr_auc']}")
    print(f"  Recall:     {metrics['recall']}")
    print(f"  Precision:  {metrics['precision']}")
    print(f"  F1-Score:   {metrics['f1_score']}")
    print(f"  Prevalencia: {metrics['prevalencia']}")

    return model, metrics, X_test, y_test


def save_artifacts(df, model, metrics):
    """Guarda dataset, modelo y métricas."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Dataset
    csv_path = os.path.join(OUTPUT_DIR, "dataset_ecv_pediatrico.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Dataset guardado en: {csv_path}")

    # Modelo
    model_path = os.path.join(MODELS_DIR, "modelo_final_GradientBoosting.joblib")
    joblib.dump(model, model_path)
    print(f"[OK] Modelo guardado en: {model_path}")

    # Métricas
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] Métricas guardadas en: {metrics_path}")

    # Crear audit log vacío
    audit_path = os.path.join(OUTPUT_DIR, "audit_log.csv")
    if not os.path.exists(audit_path):
        pd.DataFrame(columns=[
            "timestamp", "sexo", "edad", "colesterol_total",
            "presion_sistolica", "presion_diastolica",
            "frecuencia_cardiaca", "probabilidad", "clasificacion"
        ]).to_csv(audit_path, index=False)
        print(f"[OK] Audit log creado en: {audit_path}")


if __name__ == "__main__":
    df = generate_dataset()
    model, metrics, X_test, y_test = train_model(df)
    save_artifacts(df, model, metrics)
    print("\n[OK] Todos los artefactos generados exitosamente.")
