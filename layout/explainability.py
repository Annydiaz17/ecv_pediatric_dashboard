"""
Página 5 — EXPLICABILIDAD (SHAP)
SHAP global summary, individual waterfall, y explicación por paciente.
"""
from dash import html, dcc
from utils.icons import icon, SECTION_ICONS


def create_explainability_layout():
    """Genera el layout de la página de explicabilidad."""
    return html.Div([
        # Header
        html.Div([
            html.H1("Explicabilidad del Modelo", className="page-title"),
            html.P("Análisis SHAP para interpretar las decisiones del modelo",
                   className="page-subtitle"),
        ], className="page-header"),

        # Info box
        html.Div([
            icon(SECTION_ICONS["info"], size=18, color="#1e40af"),
            html.Span(
                " SHAP (SHapley Additive exPlanations) descompone cada predicción "
                "en la contribución individual de cada variable. Valores positivos "
                "aumentan la probabilidad de riesgo alto; valores negativos la disminuyen.",
                style={"marginLeft": "8px"},
            ),
        ], className="alert-box alert-info", style={"marginBottom": "24px"}),

        # SHAP Global Summary
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["shap_global"], size=18, color="#1a4076"),
                    html.Span(" SHAP Summary Plot (Global)",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            html.Div(
                "Muestra la importancia y dirección del efecto de cada variable "
                "sobre todas las predicciones del dataset de prueba.",
                style={"fontSize": "13px", "color": "#6b7280", "marginBottom": "16px"}
            ),
            dcc.Loading(
                dcc.Graph(id="shap-summary-plot", config={"displayModeBar": True}),
                type="circle",
                color="#1a4076",
            ),
        ], className="card"),

        # SHAP Individual
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["shap_individual"], size=18, color="#1a4076"),
                    html.Span(" SHAP Individual — Paciente del Simulador",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            html.Div(
                "Ingrese datos en el Simulador Clínico para ver la explicación SHAP "
                "individual del paciente. Si ya realizó una predicción, "
                "los datos aparecerán automáticamente.",
                style={"fontSize": "13px", "color": "#6b7280", "marginBottom": "16px"},
                id="shap-individual-info"
            ),

            # Formulario inline para paciente
            html.Div([
                html.Div([
                    html.Label("Género", className="form-label"),
                    dcc.Dropdown(
                        id="shap-sexo",
                        options=[
                            {"label": "Femenino", "value": 0},
                            {"label": "Masculino", "value": 1},
                        ],
                        value=0,
                        clearable=False,
                        style={"fontFamily": "Inter"},
                    ),
                ], style={"flex": "1", "minWidth": "120px"}),

                html.Div([
                    html.Label("Edad", className="form-label"),
                    dcc.Input(id="shap-edad", type="number",
                             min=2, max=17, value=10, step=0.5),
                ], style={"flex": "1", "minWidth": "100px"}),

                html.Div([
                    html.Label("Peso (kg)", className="form-label"),
                    dcc.Input(id="shap-peso", type="number",
                             min=10, max=120, value=35, step=0.5),
                ], style={"flex": "1", "minWidth": "100px"}),

                html.Div([
                    html.Label("PAS", className="form-label"),
                    dcc.Input(id="shap-pas", type="number",
                             min=70, max=160, value=110),
                ], style={"flex": "1", "minWidth": "100px"}),

                html.Div([
                    html.Label("FC", className="form-label"),
                    dcc.Input(id="shap-fc", type="number",
                             min=50, max=140, value=85),
                ], style={"flex": "1", "minWidth": "100px"}),

                html.Div([
                    html.Label("Colesterol", className="form-label"),
                    dcc.Input(id="shap-colesterol", type="number",
                             min=100, max=300, value=170),
                ], style={"flex": "1", "minWidth": "100px"}),

                html.Div([
                    html.Label(" ", className="form-label"),
                    html.Button(
                        [
                            icon("calculator-variant", size=16, color="#ffffff"),
                            html.Span(" Calcular SHAP", style={"marginLeft": "4px"}),
                        ],
                        id="shap-compute-btn",
                        className="btn-primary",
                        style={"width": "100%", "padding": "10px"},
                        n_clicks=0,
                    ),
                ], style={"flex": "1.5", "minWidth": "160px"}),
            ], style={
                "display": "flex", "gap": "12px", "flexWrap": "wrap",
                "marginBottom": "24px",
                "padding": "20px",
                "background": "#f9fafb",
                "borderRadius": "12px",
                "border": "1px solid #e5e7eb",
            }),

            # Waterfall Plot
            dcc.Loading(
                html.Div(id="shap-waterfall-container"),
                type="circle",
                color="#1a4076",
            ),
        ], className="card"),

        # SHAP Force Plot Individual
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["shap_bar"], size=18, color="#1a4076"),
                    html.Span(" SHAP Bar Plot Individual",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            dcc.Loading(
                html.Div(id="shap-bar-individual"),
                type="circle",
                color="#1a4076",
            ),
        ], className="card"),
    ])
