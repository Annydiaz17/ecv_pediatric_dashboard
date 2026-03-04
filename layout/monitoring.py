"""
Página 6 — MONITOREO Y AUDITORÍA
Tabla de predicciones, métricas por subgrupo, alertas.
"""
from dash import html, dcc, dash_table
from utils.icons import icon, SECTION_ICONS


def create_monitoring_layout():
    """Genera el layout de la página de monitoreo."""
    return html.Div([
        # Header
        html.Div([
            html.H1("Monitoreo y Auditoría", className="page-title"),
            html.P("Seguimiento de predicciones realizadas y métricas por subgrupo",
                   className="page-subtitle"),
        ], className="page-header"),

        # KPIs de auditoría
        html.Div(id="monitoring-kpis", className="kpi-grid"),

        # Alertas
        html.Div(id="monitoring-alerts", style={"marginBottom": "24px"}),

        # Métricas por subgrupo
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["group"], size=18, color="#1a4076"),
                    html.Span(" Métricas por Subgrupo (Sexo)",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
                html.Button(
                    [
                        icon(SECTION_ICONS["refresh"], size=16, color="#1a4076"),
                        html.Span(" Actualizar", style={"marginLeft": "4px"}),
                    ],
                    id="monitoring-refresh-btn",
                    className="btn-secondary",
                    style={"fontSize": "13px"},
                    n_clicks=0,
                ),
            ], className="card-header"),
            html.Div(id="monitoring-subgroup-metrics"),
            dcc.Graph(id="monitoring-subgroup-chart",
                     config={"displayModeBar": False}),
        ], className="card"),

        # Distribución de probabilidades
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["probability"], size=18, color="#1a4076"),
                    html.Span(" Distribución de Probabilidades Predichas",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            dcc.Graph(id="monitoring-prob-dist", config={"displayModeBar": True}),
        ], className="card"),

        # Tabla de auditoría
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["audit"], size=18, color="#1a4076"),
                    html.Span(" Registro de Predicciones",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
                html.Div([
                    html.Button([
                        icon(SECTION_ICONS["export"], size=16, color="#1a4076"),
                        " Exportar Log",
                    ], id="monitoring-export-btn",
                       className="btn-secondary",
                       style={"fontSize": "13px", "marginRight": "8px"}),
                    html.Button([
                        icon(SECTION_ICONS["delete"], size=16, color="#ffffff"),
                        " Limpiar Log",
                    ], id="monitoring-clear-btn",
                       className="btn-danger",
                       style={"fontSize": "13px"}),
                ], style={"display": "flex", "gap": "8px"}),
            ], className="card-header"),
            dcc.Download(id="monitoring-download"),
            html.Div(id="monitoring-audit-table"),
        ], className="card"),

        # Store para confirmación
        dcc.ConfirmDialog(
            id="monitoring-confirm-clear",
            message="¿Está seguro de limpiar el registro de auditoría? "
                    "Esta acción no se puede deshacer.",
        ),
    ])
