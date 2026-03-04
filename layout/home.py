"""
Pestaña 1 — RESUMEN DEMOGRÁFICO Y EXPLORACIÓN DE DATOS (EDA)
KPIs, gráfico de dona, histogramas comparativos por clase, heatmap de correlación.
"""
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils.model_loader import get_metrics
from utils.data_loader import get_data, TARGET
from utils.icons import icon, KPI_ICONS, SECTION_ICONS


def create_home_layout():
    """Genera el layout de la página Resumen + EDA."""
    try:
        metrics = get_metrics()
    except FileNotFoundError:
        return html.Div([
            html.Div([
                html.H1([
                    icon("alert-outline", size=28, color="#d97706"),
                    " Modelo no encontrado",
                ], className="page-title"),
                html.P("Verifique que los archivos del modelo estén en la carpeta 'models/'.",
                       className="page-subtitle"),
            ], className="page-header"),
        ])

    # Obtener datos
    try:
        df = get_data()
        n_total = len(df)
        n_alto = int(df[TARGET].sum())
        n_bajo = n_total - n_alto
    except Exception:
        n_total = 4998
        n_alto = 1100
        n_bajo = 3898

    # Métricas del JSON
    edad_prom = metrics.get("edad_promedio", 11.4)
    peso_prom = metrics.get("peso_promedio", 43.43)
    col_prom = metrics.get("colesterol_promedio", 159.9)

    return html.Div([
        # Header
        html.Div([
            html.H1("Resumen Demográfico y Exploración de Datos", className="page-title"),
            html.P("Análisis exploratorio del dataset pediátrico — Metodología SEMMA",
                   className="page-subtitle"),
        ], className="page-header"),

        # ═══════════════ KPIs ═══════════════
        html.Div([
            # Total pacientes
            html.Div([
                html.Div(
                    icon(KPI_ICONS["patients"], size=28, color="#3a6fa8"),
                    className="kpi-icon",
                ),
                html.Div(f"{n_total:,}", className="kpi-value"),
                html.Div("Total Pacientes", className="kpi-label"),
                html.Div("Pacientes analizados (6-17 años)", className="kpi-sublabel"),
            ], className="kpi-card kpi-info"),

            # Edad promedio
            html.Div([
                html.Div(
                    icon(KPI_ICONS["age"], size=28, color="#1a4076"),
                    className="kpi-icon",
                ),
                html.Div(f"{edad_prom} años", className="kpi-value"),
                html.Div("Edad Promedio", className="kpi-label"),
                html.Div("Rango pediátrico evaluado", className="kpi-sublabel"),
            ], className="kpi-card"),

            # Peso promedio
            html.Div([
                html.Div(
                    icon(KPI_ICONS["weight"], size=28, color="#059669"),
                    className="kpi-icon",
                ),
                html.Div(f"{peso_prom} kg", className="kpi-value"),
                html.Div("Peso Promedio", className="kpi-label"),
                html.Div("Peso corporal medio", className="kpi-sublabel"),
            ], className="kpi-card kpi-success"),

            # Colesterol promedio
            html.Div([
                html.Div(
                    icon(KPI_ICONS["cholesterol"], size=28, color="#d97706"),
                    className="kpi-icon",
                ),
                html.Div(f"{col_prom} mg/dL", className="kpi-value"),
                html.Div("Colesterol Promedio", className="kpi-label"),
                html.Div("Marcador metabólico clave", className="kpi-sublabel"),
            ], className="kpi-card kpi-warning"),
        ], className="kpi-grid"),

        # ═══════════════ Dona de Desbalance + Info ═══════════════
        html.Div([
            # Dona
            html.Div([
                html.Div([
                    icon(SECTION_ICONS.get("prevalencia", "alert-circle-outline"), size=18, color="#1a4076"),
                    html.Span(" Distribución de Riesgo Cardiovascular",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
                dcc.Graph(
                    figure=_build_riesgo_donut(n_alto, n_bajo),
                    config={"displayModeBar": False},
                    style={"height": "320px"},
                ),
                html.Div([
                    html.Div([
                        icon("alert-circle-outline", size=14, color="#1e40af"),
                        html.Span(
                            " Desbalance de clases: 78% sin riesgo vs 22% con riesgo. "
                            "Se usa class_weight='balanced' para compensar.",
                            style={"marginLeft": "6px", "fontSize": "12px", "color": "#4b5563"},
                        ),
                    ], style={"display": "flex", "alignItems": "flex-start"}),
                ], className="alert-box alert-info",
                   style={"marginTop": "12px", "padding": "10px 14px"}),
            ], className="card", style={"flex": "1"}),

            # Métricas de riesgo
            html.Div([
                html.Div([
                    icon(SECTION_ICONS.get("group", "account-group"), size=18, color="#1a4076"),
                    html.Span(" Detalle por Categoría de Riesgo",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
                html.Div(style={"height": "16px"}),
                # Sin riesgo
                html.Div([
                    html.Div([
                        html.Div("Sin Riesgo (Clase 0)",
                                 style={"fontSize": "14px", "color": "#6b7280",
                                        "marginBottom": "8px"}),
                        html.Div(f"{n_bajo:,}", style={
                            "fontSize": "42px", "fontWeight": "800", "color": "#059669",
                        }),
                        html.Div(f"{n_bajo / n_total * 100:.1f}%", style={
                            "fontSize": "18px", "fontWeight": "600", "color": "#059669",
                        }),
                    ], style={"textAlign": "center", "padding": "20px",
                              "background": "#d1fae5", "borderRadius": "12px",
                              "marginBottom": "16px"}),
                ]),
                # Con riesgo
                html.Div([
                    html.Div([
                        html.Div("Con Riesgo (Clase 1)",
                                 style={"fontSize": "14px", "color": "#6b7280",
                                        "marginBottom": "8px"}),
                        html.Div(f"{n_alto:,}", style={
                            "fontSize": "42px", "fontWeight": "800", "color": "#dc2626",
                        }),
                        html.Div(f"{n_alto / n_total * 100:.1f}%", style={
                            "fontSize": "18px", "fontWeight": "600", "color": "#dc2626",
                        }),
                    ], style={"textAlign": "center", "padding": "20px",
                              "background": "#fee2e2", "borderRadius": "12px"}),
                ]),
            ], className="card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),

        # ═══════════════ Histogramas Comparativos ═══════════════
        html.Div([
            html.Div([
                icon(SECTION_ICONS["histogram"], size=18, color="#1a4076"),
                html.Span(" Histogramas Comparativos por Clase de Riesgo",
                          style={"marginLeft": "6px"}),
            ], className="card-title", style={"marginBottom": "8px"}),
            html.P("Distribución cruzada de variables clínicas: Riesgo vs No Riesgo",
                   style={"fontSize": "13px", "color": "#6b7280", "marginBottom": "16px"}),
        ], style={"marginTop": "8px"}),

        # Selector de variable
        html.Div([
            html.Label("Variable a visualizar:", className="filter-title",
                       style={"marginRight": "12px"}),
            dcc.Dropdown(
                id="eda-variable-selector",
                options=[
                    {"label": "Edad", "value": "edad"},
                    {"label": "Peso (kg)", "value": "peso_kg"},
                    {"label": "Frecuencia Cardíaca", "value": "frecuencia_cardiaca"},
                    {"label": "Colesterol (mg/dL)", "value": "colesterol_mgdl"},
                    {"label": "PA Sistólica", "value": "pa_sistolica"},
                ],
                value="colesterol_mgdl",
                clearable=False,
                style={"maxWidth": "300px", "fontFamily": "Inter, sans-serif"},
            ),
        ], style={"marginBottom": "16px"}),

        # Histograma + Boxplot
        html.Div([
            html.Div([
                dcc.Graph(id="eda-histogram", config={"displayModeBar": True}),
            ], className="card", style={"flex": "1"}),

            html.Div([
                dcc.Graph(id="eda-boxplot", config={"displayModeBar": True}),
            ], className="card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),

        # ═══════════════ Mapa de Calor (Heatmap) ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["correlation"], size=18, color="#1a4076"),
                    html.Span(" Matriz de Correlación de Pearson", style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            html.P([
                "Destaca la fuerte correlación entre ",
                html.Strong("Edad y Peso (0.91)"),
                " y la correlación del ",
                html.Strong("Colesterol con Riesgo CV (0.41)"),
                "."
            ], style={"fontSize": "13px", "color": "#6b7280", "marginBottom": "8px"}),
            dcc.Graph(id="eda-correlation", config={"displayModeBar": True}),
        ], className="card"),
    ])


def _build_riesgo_donut(n_alto, n_bajo):
    """Donut chart de distribución de riesgo CV."""
    total = n_alto + n_bajo
    fig = go.Figure(data=[go.Pie(
        labels=["Sin Riesgo (78.0%)", "Con Riesgo (22.0%)"],
        values=[n_bajo, n_alto],
        hole=0.55,
        marker=dict(colors=["#4682B4", "#CD5C5C"]),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=12),
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    )])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        font_family="Inter",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
