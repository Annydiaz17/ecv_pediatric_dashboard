"""
Callbacks para el Simulador Clínico.
Usa LogisticRegression + StandardScaler del notebook SEMMA.
"""
from dash import Input, Output, State, callback, html, no_update
import plotly.graph_objects as go

from utils.model_loader import predict_risk, FEATURE_LABELS
from utils.audit_logger import log_prediction
from utils.icons import icon, SECTION_ICONS, SEMAFORO_ICONS

PLOT_TEMPLATE = "plotly_white"


@callback(
    Output("sim-semaforo", "children"),
    Output("sim-feature-importance", "children"),
    Input("sim-predict-btn", "n_clicks"),
    State("sim-sexo", "value"),
    State("sim-edad", "value"),
    State("sim-peso", "value"),
    State("sim-pas", "value"),
    State("sim-fc", "value"),
    State("sim-colesterol", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, genero, edad, peso, pas, fc, colesterol):
    """Ejecuta la predicción y muestra resultado."""
    if not n_clicks:
        return no_update, no_update

    # Validación
    if any(v is None for v in [genero, edad, peso, pas, fc, colesterol]):
        return html.Div([
            html.Div([
                icon(SECTION_ICONS["alert_warning"], size=18, color="#92400e"),
                html.Span(" Complete todos los campos para continuar.",
                           style={"marginLeft": "8px"}),
            ], className="alert-box alert-warning"),
        ]), ""

    # Predicción (orden: genero, edad, peso_kg, pa_sistolica, fc, colesterol)
    result = predict_risk(genero, edad, peso, pas, fc, colesterol)

    # Convertir genero a label
    genero_label = "Femenino" if genero == 0 else "Masculino"

    # Registrar en auditoría
    log_prediction(
        sexo=genero_label, edad=edad, peso_kg=peso,
        pa_sistolica=pas, frecuencia_cardiaca=fc,
        colesterol_mgdl=colesterol,
        probabilidad=result["probabilidad"],
        clasificacion=result["clasificacion"]
    )

    # ─── Semáforo Visual ─────────────────────────────────────
    nivel = result["nivel"]
    if nivel == "bajo":
        semaforo_class = "semaforo-verde"
        semaforo_icon = SEMAFORO_ICONS["bajo"]
        icon_color = "#065f46"
        label = "RIESGO BAJO"
        color_text = "#065f46"
    elif nivel == "moderado":
        semaforo_class = "semaforo-amarillo"
        semaforo_icon = SEMAFORO_ICONS["moderado"]
        icon_color = "#92400e"
        label = "RIESGO MODERADO"
        color_text = "#92400e"
    else:
        semaforo_class = "semaforo-rojo"
        semaforo_icon = SEMAFORO_ICONS["alto"]
        icon_color = "#991b1b"
        label = "RIESGO ALTO"
        color_text = "#991b1b"

    semaforo = html.Div([
        html.Div([
            html.Span(
                icon(semaforo_icon, size=64, color=icon_color),
                className="semaforo-icon",
            ),
            html.Div(label, className="semaforo-label"),
            html.Div(f"{result['porcentaje']}%", className="semaforo-prob"),
            html.Div("Probabilidad de riesgo cardiovascular",
                     className="semaforo-sublabel"),
        ], className=f"semaforo-container {semaforo_class}"),

        # Clasificación
        html.Div([
            html.Div([
                html.Span("Clasificación: "),
                html.Span(result["clasificacion"],
                         style={"fontWeight": "700", "color": color_text}),
            ], style={
                "textAlign": "center", "marginTop": "16px",
                "fontSize": "16px", "color": "#374151",
            }),
        ]),

        # Datos del paciente
        html.Div([
            html.Div([
                icon(SECTION_ICONS["patient"], size=18, color="#1a4076"),
                html.Span(" Resumen del Paciente",
                          style={"marginLeft": "6px"}),
            ], className="card-title", style={"marginBottom": "16px"}),
            html.Div([
                _patient_field("Género", genero_label),
                _patient_field("Edad", f"{edad} años"),
                _patient_field("Peso", f"{peso} kg"),
                _patient_field("PA Sistólica", f"{pas} mmHg"),
                _patient_field("Frecuencia Cardíaca", f"{fc} lpm"),
                _patient_field("Colesterol", f"{colesterol} mg/dL"),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                      "gap": "8px"}),
        ], className="card", style={"marginTop": "16px"}),
    ])

    # ─── Feature Importance ──────────────────────────────────
    importances = result["feature_importances"]
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    max_val = max(v for _, v in sorted_imp)

    fi_card = html.Div([
        html.Div([
            icon(SECTION_ICONS["feature_importance"], size=18, color="#1a4076"),
            html.Span(" Importancia de Variables (Modelo)",
                      style={"marginLeft": "6px"}),
        ], className="card-title", style={"marginBottom": "20px"}),

        html.Div(
            "Las variables con mayor importancia tienen más peso en la decisión del modelo.",
            style={"fontSize": "13px", "color": "#6b7280", "marginBottom": "16px"}
        ),

        *[html.Div([
            html.Div([
                html.Span(
                    FEATURE_LABELS.get(feat, feat),
                    style={"fontSize": "13px", "fontWeight": "600", "color": "#374151"}
                ),
                html.Span(
                    f"{val:.4f}",
                    style={"fontSize": "13px", "fontWeight": "700", "color": "#1a4076"}
                ),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "marginBottom": "4px"}),
            html.Div(
                html.Div(style={
                    "width": f"{(val / max_val) * 100}%",
                    "height": "10px",
                    "borderRadius": "5px",
                    "background": "linear-gradient(90deg, #1a4076, #3a6fa8)",
                    "transition": "width 0.8s ease",
                }),
                style={
                    "width": "100%", "height": "10px",
                    "borderRadius": "5px", "background": "#f3f4f6",
                }
            ),
        ], style={"marginBottom": "16px"}) for feat, val in sorted_imp],
    ], className="card")

    return semaforo, fi_card


def _patient_field(label, value):
    """Crea un campo de resumen del paciente."""
    return html.Div([
        html.Span(f"{label}: ", style={
            "fontSize": "13px", "color": "#6b7280"
        }),
        html.Span(value, style={
            "fontSize": "13px", "fontWeight": "600", "color": "#374151"
        }),
    ], style={
        "padding": "8px 12px",
        "background": "#f9fafb",
        "borderRadius": "6px",
    })
