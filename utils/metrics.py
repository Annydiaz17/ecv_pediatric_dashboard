"""
Utilidades de cálculo de métricas para evaluación del modelo.
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    f1_score, roc_curve, precision_recall_curve, auc
)


def compute_threshold_metrics(y_true, y_proba, threshold=0.5):
    """
    Calcula métricas de clasificación para un threshold dado.

    Returns:
        dict con recall, precision, specificity, f1, confusion_matrix
    """
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": int(len(y_true))
    }


def compute_roc_data(y_true, y_proba):
    """Calcula datos para la curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc_val


def compute_pr_data(y_true, y_proba):
    """Calcula datos para la curva Precision-Recall."""
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc_val = auc(recall_vals, precision_vals)
    return precision_vals, recall_vals, thresholds, pr_auc_val


def compute_subgroup_metrics(y_true, y_pred, groups):
    """
    Calcula métricas por subgrupo.

    Args:
        y_true: etiquetas reales
        y_pred: predicciones
        groups: array con etiquetas de grupo

    Returns:
        dict con métricas por grupo
    """
    unique_groups = np.unique(groups)
    results = {}

    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 2:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        results[g] = {
            "n": int(mask.sum()),
            "prevalencia": round(float(yt.mean()), 4),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        }

    return results
