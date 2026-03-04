"""
Página 4 — SIMULADOR CLÍNICO
Formulario de entrada de datos clínicos y predicción interactiva.
Usa el modelo LogisticRegression del notebook SEMMA.
"""
from dash import html, dcc
from utils.icons import icon, SECTION_ICONS


def create_simulator_layout():
    """Genera el layout del simulador clínico."""
    return html.Div([
        # Header
        html.Div([
            html.H1("Simulador Clínico", className="page-title"),
            html.P("Ingrese los datos del paciente para evaluar el riesgo cardiovascular",
                   className="page-subtitle"),
        ], className="page-header"),

        html.Div([
            # Panel izquierdo: Formulario
            html.Div([
                html.Div([
                    html.Div([
                        html.Span([
                            icon(SECTION_ICONS["vitals"], size=18, color="#1a4076"),
                            html.Span(" Datos del Paciente",
                                      style={"marginLeft": "6px"}),
                        ], className="card-title"),
                    ], className="card-header"),

                    # Sexo / Género
                    html.Div([
                        html.Label("Género", className="form-label"),
                        dcc.Dropdown(
                            id="sim-sexo",
                            options=[
                                {"label": "Femenino", "value": 0},
                                {"label": "Masculino", "value": 1},
                            ],
                            value=0,
                            clearable=False,
                            style={"fontFamily": "Inter, sans-serif"},
                        ),
                    ], className="form-group"),

                    # Edad
                    html.Div([
                        html.Label("Edad (años)", className="form-label"),
                        dcc.Input(
                            id="sim-edad",
                            type="number",
                            min=2, max=17, step=0.5,
                            value=10,
                            style={"width": "100%"},
                        ),
                        html.Div("Rango: 2-17 años", className="form-sublabel"),
                    ], className="form-group"),

                    # Peso (kg) — NUEVO: reemplaza PAD
                    html.Div([
                        html.Label("Peso (kg)", className="form-label"),
                        dcc.Input(
                            id="sim-peso",
                            type="number",
                            min=10, max=120, step=0.5,
                            value=35,
                            style={"width": "100%"},
                        ),
                        html.Div("Rango pediátrico típico: 10-80 kg",
                                className="form-sublabel"),
                    ], className="form-group"),

                    # PA Sistólica
                    html.Div([
                        html.Label("PA Sistólica (mmHg)", className="form-label"),
                        dcc.Input(
                            id="sim-pas",
                            type="number",
                            min=70, max=160, step=1,
                            value=110,
                            style={"width": "100%"},
                        ),
                        html.Div("Rango normal pediátrico: 80-120 mmHg",
                                className="form-sublabel"),
                    ], className="form-group"),

                    # Frecuencia Cardíaca
                    html.Div([
                        html.Label("Frecuencia Cardíaca (lpm)", className="form-label"),
                        dcc.Input(
                            id="sim-fc",
                            type="number",
                            min=50, max=140, step=1,
                            value=85,
                            style={"width": "100%"},
                        ),
                        html.Div("Varía según la edad del paciente",
                                className="form-sublabel"),
                    ], className="form-group"),

                    # Colesterol
                    html.Div([
                        html.Label("Colesterol (mg/dL)", className="form-label"),
                        dcc.Input(
                            id="sim-colesterol",
                            type="number",
                            min=100, max=300, step=1,
                            value=170,
                            style={"width": "100%"},
                        ),
                        html.Div("Rango normal: 120-200 mg/dL",
                                 className="form-sublabel"),
                    ], className="form-group"),

                    # Botón de predicción
                    html.Button(
                        [
                            icon(SECTION_ICONS["predict"], size=18, color="#ffffff"),
                            html.Span(" Evaluar Riesgo", style={"marginLeft": "6px"}),
                        ],
                        id="sim-predict-btn",
                        className="btn-primary",
                        style={"width": "100%", "justifyContent": "center",
                               "marginTop": "8px", "padding": "14px"},
                        n_clicks=0,
                    ),

                ], className="card"),
            ], style={"flex": "1", "minWidth": "320px"}),

            # Panel derecho: Resultado
            html.Div([
                # Semáforo
                html.Div(id="sim-semaforo", style={"marginBottom": "24px"}),

                # Feature Importance
                html.Div(id="sim-feature-importance"),

            ], style={"flex": "1.5", "minWidth": "400px"}),

        ], style={"display": "flex", "gap": "32px", "flexWrap": "wrap"}),
    ])
