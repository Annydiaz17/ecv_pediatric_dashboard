"""
Pestaña 4 — EXPLICABILIDAD (XAI) Y SIMULADOR PREDICTIVO
Feature importance + calculadora interactiva de riesgo con sliders.
Diseño compacto: diagnóstico integrado junto al semáforo.
"""
from dash import html, dcc
from utils.model_loader import get_metrics
from utils.icons import icon, SECTION_ICONS


def create_xai_simulator_layout():
    """Genera el layout de la página XAI + Simulador."""
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

    importances = metrics.get("feature_importances", {})

    return html.Div([
        # Header
        html.Div([
            html.H1("Explicabilidad y Simulador Predictivo", className="page-title"),
            html.P("Importancia de variables + calculadora interactiva de riesgo cardiovascular",
                   className="page-subtitle"),
        ], className="page-header"),

        # ═══════════════ Feature Importance (compacto) ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["feature_importance"], size=18, color="#1a4076"),
                    html.Span(" Importancia de Variables — Modelo Logístico",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
                html.Span(
                    "Top: Colesterol → Peso → PAS",
                    style={"fontSize": "12px", "color": "#6b7280",
                           "fontWeight": "500"},
                ),
            ], className="card-header"),
            dcc.Graph(
                id="feature-importance-chart",
                figure=_build_feature_importance(importances),
                config={"displayModeBar": False},
            ),
        ], className="card"),

        # ═══════════════ Simulador ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["vitals"], size=18, color="#1a4076"),
                    html.Span(" Calculadora Interactiva de Riesgo Cardiovascular",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),

            # 3 columnas: Sliders | Semáforo + Diagnóstico | Feature bars
            html.Div([
                # Col 1: Sliders del paciente
                html.Div([
                    _slider_group("Edad (años)", "sim-edad", 6, 17, 1, 11,
                                  "6-17 años"),
                    html.Div([
                        html.Label("Sexo", className="form-label"),
                        dcc.Dropdown(
                            id="sim-sexo",
                            options=[
                                {"label": "Femenino", "value": 0},
                                {"label": "Masculino", "value": 1},
                            ],
                            value=0, clearable=False,
                            style={"fontFamily": "Inter, sans-serif"},
                        ),
                    ], className="form-group"),
                    _slider_group("Peso (kg)", "sim-peso", 5, 120, 1, 43,
                                  "5-120 kg"),
                    _slider_group("PAS (mmHg)", "sim-pas", 60, 160, 1, 110,
                                  "60-160 mmHg"),
                    _slider_group("FC (lpm)", "sim-fc", 40, 200, 1, 85,
                                  "40-200 lpm"),
                    _slider_group("Colesterol (mg/dL)", "sim-colesterol", 50, 350, 1, 160,
                                  "50-350 mg/dL"),

                    html.Button(
                        [
                            icon(SECTION_ICONS["predict"], size=18, color="#ffffff"),
                            html.Span(" Evaluar Riesgo", style={"marginLeft": "6px"}),
                        ],
                        id="sim-predict-btn",
                        className="btn-primary",
                        style={"width": "100%", "justifyContent": "center",
                               "marginTop": "8px", "padding": "12px"},
                        n_clicks=0,
                    ),
                ], style={"flex": "0 0 280px", "minWidth": "260px"}),

                # Col 2: Resultado unificado (semáforo + diagnóstico integrado)
                html.Div(id="sim-resultado-panel", style={
                    "flex": "1", "minWidth": "420px",
                }),
            ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),
        ], className="card"),

        # Contenedores ocultos para mantener compatibilidad de callbacks
        html.Div(id="sim-semaforo", style={"display": "none"}),
        html.Div(id="sim-feature-importance", style={"display": "none"}),
        html.Div(id="sim-diagnostico", style={"display": "none"}),
    ])


def _slider_group(label, slider_id, min_val, max_val, step, default, sublabel):
    """Slider compacto."""
    return html.Div([
        html.Div([
            html.Label(label, className="form-label",
                       style={"fontSize": "12px", "marginBottom": "2px"}),
        ]),
        dcc.Slider(
            id=slider_id,
            min=min_val, max=max_val, step=step, value=default,
            marks={min_val: str(min_val), max_val: str(max_val)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(sublabel, className="form-sublabel",
                 style={"fontSize": "10px", "marginTop": "-2px"}),
    ], className="form-group", style={"marginBottom": "10px"})


def _build_feature_importance(importances):
    """Barras horizontales de feature importance (compactas)."""
    import plotly.graph_objects as go

    if not importances:
        fig = go.Figure()
        fig.add_annotation(text="No disponible", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    labels_map = {
        "colesterol_mgdl": "Colesterol",
        "peso_kg": "Peso",
        "pa_sistolica": "PAS",
        "frecuencia_cardiaca": "FC",
        "edad": "Edad",
        "genero": "Género",
    }

    sorted_features = sorted(importances.items(), key=lambda x: x[1])
    names = [labels_map.get(k, k) for k, _ in sorted_features]
    values = [v for _, v in sorted_features]

    colors = []
    for i, (k, v) in enumerate(sorted_features):
        rank = len(sorted_features) - i
        if rank == 1:
            colors.append("#4682B4")
        elif rank == 2:
            colors.append("#2E8B57")
        elif rank == 3:
            colors.append("#DAA520")
        else:
            colors.append("#9CA3AF")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(family="Inter", size=11, color="#374151"),
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Importancia",
        template="plotly_white",
        font_family="Inter",
        margin=dict(l=10, r=60, t=10, b=30),
        height=220,
        showlegend=False,
    )
    return fig
