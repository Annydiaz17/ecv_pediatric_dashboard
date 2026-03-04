"""
Sistema centralizado de íconos profesionales.
Usa Dash Iconify con Material Design Icons (mdi).
Todos los íconos del dashboard se definen aquí para consistencia.
"""
from dash_iconify import DashIconify


def icon(name, size=20, color=None, **kwargs):
    """
    Crea un ícono DashIconify con estilo uniforme.

    Args:
        name: Nombre del ícono MDI (sin prefijo 'mdi:')
        size: Tamaño en píxeles (default: 20)
        color: Color CSS opcional
        **kwargs: Atributos extra para DashIconify
    """
    style = {"verticalAlign": "middle", "display": "inline-block"}
    if color:
        style["color"] = color
    return DashIconify(
        icon=f"mdi:{name}",
        width=size,
        height=size,
        style=style,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════
# NAVEGACIÓN — 4 Pestañas SEMMA
# ═══════════════════════════════════════════════════════════════
NAV_ICONS = {
    "home": "view-dashboard-outline",
    "segmentation": "scatter-plot",
    "model": "chart-line",
    "xai_simulator": "stethoscope",
}

# ═══════════════════════════════════════════════════════════════
# KPI / MÉTRICAS
# ═══════════════════════════════════════════════════════════════
KPI_ICONS = {
    "roc_auc": "chart-bell-curve-cumulative",
    "pr_auc": "chart-timeline-variant",
    "recall": "target",
    "precision": "check-decagram",
    "f1": "scale-balance",
    "prevalencia": "chart-pie",
    "accuracy": "check-circle-outline",
    "model": "brain",
    "dataset": "database-outline",
    "patients": "account-group",
    "age": "calendar-clock",
    "weight": "weight-kilogram",
    "cholesterol": "water",
}

# ═══════════════════════════════════════════════════════════════
# SECCIONES Y ACCIONES
# ═══════════════════════════════════════════════════════════════
SECTION_ICONS = {
    # Datos y exploración
    "filter": "filter-variant",
    "histogram": "chart-histogram",
    "boxplot": "chart-box-outline",
    "correlation": "link-variant",
    "table": "table",
    "export": "download",
    "import": "upload",

    # Modelo
    "trophy": "trophy-outline",
    "roc_curve": "chart-bell-curve",
    "pr_curve": "chart-timeline-variant",
    "threshold": "tune-vertical",
    "confusion": "grid",

    # Clínico
    "patient": "account-outline",
    "predict": "magnify-scan",
    "feature_importance": "key-variant",
    "heart": "heart-pulse",
    "vitals": "clipboard-pulse-outline",

    # SHAP / XAI
    "shap_global": "earth",
    "shap_individual": "account-search",
    "shap_bar": "chart-bar",
    "info": "information-outline",

    # Segmentación
    "clustering": "chart-bubble",
    "pca": "axis-arrow",
    "cluster_profile": "account-details",

    # Alertas
    "alert_info": "information-outline",
    "alert_warning": "alert-outline",
    "alert_danger": "alert-circle-outline",
    "alert_success": "check-circle-outline",

    # General
    "prevalencia": "alert-circle-outline",
    "group": "account-group",
}

# ═══════════════════════════════════════════════════════════════
# SEMÁFORO DE RIESGO
# ═══════════════════════════════════════════════════════════════
SEMAFORO_ICONS = {
    "bajo": "shield-check",
    "moderado": "alert-rhombus-outline",
    "alto": "alert-octagon",
}

# ═══════════════════════════════════════════════════════════════
# LOGO
# ═══════════════════════════════════════════════════════════════
LOGO_ICON = "heart-pulse"

# ═══════════════════════════════════════════════════════════════
# SIDEBAR NAV ITEMS — 4 Pestañas SEMMA
# ═══════════════════════════════════════════════════════════════
SIDEBAR_NAV = [
    {"page": "/",               "icon": NAV_ICONS["home"],           "label": "Resumen y EDA"},
    {"page": "/segmentation",   "icon": NAV_ICONS["segmentation"],   "label": "Segmentación"},
    {"page": "/assessment",     "icon": NAV_ICONS["model"],          "label": "Evaluación de Modelos"},
    {"page": "/xai-simulator",  "icon": NAV_ICONS["xai_simulator"],  "label": "XAI y Simulador"},
]
