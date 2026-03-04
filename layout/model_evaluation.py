"""
Pestaña 3 — EVALUACIÓN Y COMPARACIÓN DE MODELOS (Assess)
Tabla de 5 modelos, matriz de confusión, curvas ROC y PR.
"""
from dash import html, dcc, dash_table
from utils.model_loader import get_metrics
from utils.icons import icon, SECTION_ICONS


def create_model_eval_layout():
    """Genera el layout de la página de evaluación del modelo."""
    try:
        metrics = get_metrics()
    except FileNotFoundError:
        return html.Div([
            html.Div([
                html.H1([
                    icon("alert-outline", size=28, color="#d97706"),
                    " Modelo no encontrado",
                ], className="page-title"),
            ], className="page-header"),
        ])

    comparison = metrics.get("model_comparison", [])

    return html.Div([
        # Header
        html.Div([
            html.H1("Evaluación y Comparación de Modelos", className="page-title"),
            html.P("Fase ASSESS — Comparación de los 5 algoritmos entrenados con la metodología SEMMA",
                   className="page-subtitle"),
        ], className="page-header"),

        # Info banner
        html.Div([
            html.Div([
                html.Div("Modelo Ganador", className="info-label"),
                html.Div("Logistic Regression", className="info-value"),
            ], className="info-item"),
            html.Div([
                html.Div("Prioridad Clínica", className="info-label"),
                html.Div("Recall/Sensibilidad", className="info-value",
                         style={"color": "#2E8B57"}),
            ], className="info-item"),
            html.Div([
                html.Div("Recall Logístico", className="info-label"),
                html.Div("83.18%", className="info-value",
                         style={"color": "#2E8B57"}),
            ], className="info-item"),
            html.Div([
                html.Div("ROC-AUC", className="info-label"),
                html.Div("0.889", className="info-value"),
            ], className="info-item"),
        ], className="model-info-banner"),

        # Alert box
        html.Div([
            icon(SECTION_ICONS["info"], size=18, color="#1e40af"),
            html.Span(
                " En pediatría preventiva, minimizar los falsos negativos es crítico. "
                "Un niño con riesgo no detectado (FN) tiene mayor costo clínico que un falso positivo. "
                "Por eso se prioriza el Recall como métrica principal.",
                style={"marginLeft": "8px"},
            ),
        ], className="alert-box alert-info", style={"marginBottom": "24px"}),

        # ═══════════════ Tabla Comparativa ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["trophy"], size=18, color="#1a4076"),
                    html.Span(" Métricas Clínicas — 5 Algoritmos",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),

            dash_table.DataTable(
                id="model-comparison-table",
                columns=[
                    {"name": "Modelo", "id": "modelo"},
                    {"name": "ROC-AUC", "id": "roc_auc", "type": "numeric",
                     "format": {"specifier": ".4f"}},
                    {"name": "Recall", "id": "recall", "type": "numeric",
                     "format": {"specifier": ".4f"}},
                    {"name": "Precision", "id": "precision", "type": "numeric",
                     "format": {"specifier": ".4f"}},
                    {"name": "F1-Score", "id": "f1", "type": "numeric",
                     "format": {"specifier": ".4f"}},
                    {"name": "Accuracy", "id": "accuracy", "type": "numeric",
                     "format": {"specifier": ".4f"}},
                ],
                data=[
                    {k: v for k, v in m.items() if k not in ("cm", "winner")}
                    for m in comparison
                ],
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "center",
                    "fontFamily": "Inter, sans-serif",
                    "fontSize": "14px",
                    "padding": "12px 16px",
                },
                style_header={
                    "backgroundColor": "#0f2442",
                    "color": "white",
                    "fontWeight": "600",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{modelo} = "Logistic Regression"'},
                        "backgroundColor": "#e8f1f8",
                        "fontWeight": "700",
                        "borderLeft": "4px solid #2E8B57",
                    }
                ],
            ),
        ], className="card"),

        # ═══════════════ Selector de modelo para confusión ═══════════════
        html.Div([
            html.Label("Modelo para Matriz de Confusión:", className="filter-title",
                       style={"marginRight": "12px"}),
            dcc.Dropdown(
                id="cm-model-selector",
                options=[
                    {"label": m["modelo"], "value": m["modelo"]}
                    for m in comparison
                ],
                value="Logistic Regression",
                clearable=False,
                style={"maxWidth": "300px", "fontFamily": "Inter, sans-serif"},
            ),
        ], style={"marginBottom": "16px"}),

        # ═══════════════ Matriz de Confusión ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["confusion"], size=18, color="#1a4076"),
                    html.Span(" Matriz de Confusión", style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            dcc.Graph(id="confusion-matrix", config={"displayModeBar": False}),
        ], className="card"),

        # ═══════════════ Curvas ROC y PR ═══════════════
        html.Div([
            html.Div([
                html.Div([
                    html.Span([
                        icon(SECTION_ICONS["roc_curve"], size=18, color="#1a4076"),
                        html.Span(" Curva ROC — Comparación de Modelos",
                                  style={"marginLeft": "6px"}),
                    ], className="card-title"),
                ], className="card-header"),
                dcc.Graph(id="roc-curve", config={"displayModeBar": True}),
            ], className="card", style={"flex": "1"}),

            html.Div([
                html.Div([
                    html.Span([
                        icon(SECTION_ICONS["pr_curve"], size=18, color="#1a4076"),
                        html.Span(" Curva Precision-Recall",
                                  style={"marginLeft": "6px"}),
                    ], className="card-title"),
                ], className="card-header"),
                dcc.Graph(id="pr-curve", config={"displayModeBar": True}),
            ], className="card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),
    ])
