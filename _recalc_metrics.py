"""Script para recalcular model_metrics.json con el modelo correcto."""
import json
import warnings
import sys
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    average_precision_score
)

from utils.data_loader import get_data, get_combined_stats, FEATURES_MODELO, TARGET
from utils.model_loader import get_model, get_scaler, _get_feature_importances

# Cargar modelo y scaler
model = get_model()
scaler = get_scaler()

# Cargar dataset Timbiquí para evaluación
df = get_data("timbiqui")
X = df[FEATURES_MODELO].values
y = df[TARGET].values

# Split igual que el notebook (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Escalar y predecir
X_test_scaled = scaler.transform(X_test)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# Métricas
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred)

# Stats
stats = get_combined_stats()

# Feature importances
fi = _get_feature_importances(model)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"Recall:  {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1:      {f1:.4f}")
print(f"Feature importances: {fi}")
print(f"Stats: {stats}")

# Guardar
metrics = {
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(pr_auc, 4),
    "recall": round(recall, 4),
    "precision": round(precision, 4),
    "f1_score": round(f1, 4),
    "prevalencia_global": round(stats["prevalencia_global"], 4),
    "prevalencia_timbiqui": round(stats["prevalencia_timbiqui"], 4),
    "prevalencia_real": round(stats["prevalencia_real"], 4),
    "prevalencia_sintetico": round(stats["prevalencia_sintetico"], 4),
    "n_samples": stats["n_total"],
    "n_timbiqui": stats["n_timbiqui"],
    "n_real": stats["n_real"],
    "n_sintetico": stats["n_sintetico"],
    "algorithm": "Logistic Regression",
    "feature_importances": fi,
    "model_comparison": [
        {"modelo": "Logistic Regression (Final)", "roc_auc": round(roc_auc, 4),
         "recall": round(recall, 4), "precision": round(precision, 4),
         "f1": round(f1, 4)},
    ]
}

with open("models/model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\n[OK] model_metrics.json actualizado")
