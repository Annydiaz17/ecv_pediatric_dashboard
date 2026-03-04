"""
Callbacks para la Pestaña 4 — XAI + Simulador.
Predicción interactiva con semáforo + diagnóstico clínico basado en
umbrales fisiológicos reales (no porcentajes por variable).

Interpretación clínica:
  - Colesterol ≥200 mg/dL → Perfil compatible con Dislipidemia
  - PAS ≥130 mmHg → Perfil compatible con Hipertensión Arterial
  - Peso elevado + colesterol/PAS → Perfil compatible con Síndrome Metabólico
  - FC >100 o <60 → Perfil compatible con Trastorno del Ritmo Cardíaco

IMPORTANTE: Siempre se usa "Perfil compatible con..." porque el modelo
predice riesgo global, no diagnóstico específico.
"""
from dash import Input, Output, State, callback, html, no_update

from utils.model_loader import predict_risk, FEATURE_LABELS
from utils.icons import icon, SECTION_ICONS, SEMAFORO_ICONS


# ═══════════════════════════════════════════════════════════════
# UMBRALES CLÍNICOS PEDIÁTRICOS
# ═══════════════════════════════════════════════════════════════
UMBRALES = {
    "colesterol": {
        "normal": 170,       # < 170 mg/dL
        "limite": 200,       # 170-199 mg/dL = límite alto
        "alto": 200,         # ≥ 200 mg/dL = hipercolesterolemia
    },
    "pas": {
        "normal": 110,       # < 110 mmHg
        "elevada": 130,      # 110-129 mmHg = elevada
        "hipertension": 130, # ≥ 130 mmHg = sospecha HTA
    },
    "fc": {
        "bradicardia": 60,   # < 60 lpm
        "normal_min": 60,
        "normal_max": 100,
        "taquicardia": 100,  # > 100 lpm
    },
}

# Percentiles de peso aproximados por edad (simplificado)
# Fuente: curvas OMS/CDC pediátricas
# Formato: edad -> (percentil_85, percentil_95)
PESO_PERCENTILES = {
    6: (23, 27), 7: (26, 30), 8: (29, 34), 9: (33, 39),
    10: (37, 44), 11: (42, 50), 12: (47, 57), 13: (53, 63),
    14: (58, 69), 15: (62, 74), 16: (65, 78), 17: (68, 82),
}


def _evaluar_perfil_clinico(edad, peso, pas, fc, colesterol):
    """
    Evalúa los valores clínicos del paciente contra umbrales
    fisiológicos reales y determina los perfiles compatibles.

    Retorna lista de perfiles detectados (puede haber varios).
    """
    perfiles = []

    # ─── Colesterol ──────────────────────────────────────────
    if colesterol >= UMBRALES["colesterol"]["alto"]:
        perfiles.append({
            "tipo": "dislipidemia",
            "titulo": "Perfil compatible con Dislipidemia",
            "subtitulo": f"Colesterol: {colesterol} mg/dL (≥ 200 = hipercolesterolemia)",
            "icono": "water-alert",
            "color": "#9333ea",
            "bg": "#f5f3ff",
            "border": "#c4b5fd",
            "descripcion": (
                f"El valor de colesterol ({colesterol} mg/dL) supera el umbral de "
                "200 mg/dL establecido para población pediátrica, lo que sugiere "
                "un perfil compatible con dislipidemia y riesgo de aterosclerosis temprana."
            ),
            "acciones": [
                "Solicitar perfil lipídico completo (LDL, HDL, triglicéridos)",
                "Evaluar antecedentes familiares de hipercolesterolemia",
                "Iniciar intervención dietética y reevaluar en 3-6 meses",
                "Considerar referencia a endocrinología si LDL > 160 mg/dL",
            ],
        })
    elif colesterol >= UMBRALES["colesterol"]["normal"]:
        perfiles.append({
            "tipo": "colesterol_limite",
            "titulo": "Colesterol en rango límite alto",
            "subtitulo": f"Colesterol: {colesterol} mg/dL (170-199 = límite alto)",
            "icono": "water",
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fcd34d",
            "descripcion": (
                f"El colesterol ({colesterol} mg/dL) se encuentra en el rango "
                "límite alto (170-199 mg/dL). Vigilancia recomendada."
            ),
            "acciones": [
                "Repetir medición en ayunas para confirmar",
                "Evaluar hábitos alimentarios",
                "Seguimiento en próximo control pediátrico",
            ],
        })

    # ─── Presión Arterial Sistólica ──────────────────────────
    if pas >= UMBRALES["pas"]["hipertension"]:
        perfiles.append({
            "tipo": "hipertension",
            "titulo": "Perfil compatible con Hipertensión Arterial",
            "subtitulo": f"PAS: {pas} mmHg (≥ 130 = sospecha de HTA)",
            "icono": "heart-pulse",
            "color": "#dc2626",
            "bg": "#fef2f2",
            "border": "#fca5a5",
            "descripcion": (
                f"La presión arterial sistólica ({pas} mmHg) supera el umbral "
                "de 130 mmHg, lo que sugiere un perfil compatible con hipertensión "
                "arterial pediátrica."
            ),
            "acciones": [
                "Confirmar con 3 mediciones en visitas separadas (ABPM si disponible)",
                "Evaluar según tablas de percentiles AAP por edad, sexo y talla",
                "Descartar causas secundarias: renales, endocrinas, coartación",
                "Ecocardiograma para evaluar hipertrofia ventricular si HTA confirmada",
            ],
        })
    elif pas >= UMBRALES["pas"]["normal"]:
        perfiles.append({
            "tipo": "pas_elevada",
            "titulo": "Presión arterial elevada",
            "subtitulo": f"PAS: {pas} mmHg (110-129 = elevada)",
            "icono": "heart-outline",
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fcd34d",
            "descripcion": (
                f"La PAS ({pas} mmHg) se encuentra elevada. "
                "Requiere seguimiento en próximas consultas."
            ),
            "acciones": [
                "Repetir medición en reposo en próximas visitas",
                "Evaluar actividad física y consumo de sodio",
            ],
        })

    # ─── Frecuencia Cardíaca ─────────────────────────────────
    if fc > UMBRALES["fc"]["taquicardia"]:
        perfiles.append({
            "tipo": "taquicardia",
            "titulo": "Perfil compatible con Trastorno del Ritmo Cardíaco",
            "subtitulo": f"FC: {fc} lpm (> 100 = taquicardia)",
            "icono": "heart-flash",
            "color": "#0891b2",
            "bg": "#ecfeff",
            "border": "#67e8f9",
            "descripcion": (
                f"La frecuencia cardíaca ({fc} lpm) supera el límite superior "
                "normal (100 lpm), lo que puede sugerir taquicardia o "
                "desregulación autonómica."
            ),
            "acciones": [
                "ECG de 12 derivaciones para evaluar ritmo y conducción",
                "Descartar causas reversibles: anemia, fiebre, ansiedad, cafeína",
                "Holter de 24 horas si hay síntomas intermitentes",
                "Referencia a cardiología pediátrica si persiste",
            ],
        })
    elif fc < UMBRALES["fc"]["bradicardia"]:
        perfiles.append({
            "tipo": "bradicardia",
            "titulo": "Perfil compatible con Bradicardia",
            "subtitulo": f"FC: {fc} lpm (< 60 = bradicardia)",
            "icono": "heart-flash",
            "color": "#0891b2",
            "bg": "#ecfeff",
            "border": "#67e8f9",
            "descripcion": (
                f"La frecuencia cardíaca ({fc} lpm) está por debajo del "
                "límite inferior normal (60 lpm)."
            ),
            "acciones": [
                "ECG para evaluar conducción AV",
                "Descartar hipotiroidismo o medicación bradicardizante",
                "Evaluación cardiológica si sintomático",
            ],
        })

    # ─── Peso (percentiles por edad) ─────────────────────────
    edad_int = max(6, min(17, int(edad)))
    p85, p95 = PESO_PERCENTILES.get(edad_int, (50, 60))
    if peso >= p95:
        perfiles.append({
            "tipo": "obesidad",
            "titulo": "Perfil compatible con Riesgo Metabólico por Obesidad",
            "subtitulo": f"Peso: {peso} kg (> percentil 95 para {edad_int} años = {p95} kg)",
            "icono": "weight-kilogram",
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fcd34d",
            "descripcion": (
                f"El peso ({peso} kg) supera el percentil 95 para {edad_int} años "
                f"(≈{p95} kg), indicando obesidad. Asociado a síndrome metabólico "
                "y remodelamiento cardíaco temprano."
            ),
            "acciones": [
                "Calcular IMC y confirmar percentil en curvas OMS/CDC",
                "Evaluación de circunferencia abdominal",
                "Screening metabólico: glucosa, insulina, triglicéridos, HDL",
                "Programa de intervención nutricional y actividad física",
            ],
        })
    elif peso >= p85:
        perfiles.append({
            "tipo": "sobrepeso",
            "titulo": "Sobrepeso detectado",
            "subtitulo": f"Peso: {peso} kg (> percentil 85 para {edad_int} años = {p85} kg)",
            "icono": "weight-kilogram",
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fcd34d",
            "descripcion": (
                f"El peso ({peso} kg) supera el percentil 85 para {edad_int} años "
                f"(≈{p85} kg), indicando sobrepeso."
            ),
            "acciones": [
                "Evaluar IMC y tendencia de crecimiento",
                "Orientación nutricional preventiva",
                "Control en próxima visita",
            ],
        })

    # ─── Síndrome Metabólico (combinación) ───────────────────
    tiene_colesterol_alto = colesterol >= UMBRALES["colesterol"]["alto"]
    tiene_pas_alta = pas >= UMBRALES["pas"]["hipertension"]
    tiene_peso_alto = peso >= p85

    factores_metabolicos = sum([tiene_colesterol_alto, tiene_pas_alta, tiene_peso_alto])
    if factores_metabolicos >= 2:
        # Reemplazar perfiles individuales menores con síndrome metabólico
        componentes = []
        if tiene_colesterol_alto:
            componentes.append(f"Colesterol {colesterol} mg/dL")
        if tiene_pas_alta:
            componentes.append(f"PAS {pas} mmHg")
        if tiene_peso_alto:
            componentes.append(f"Peso {peso} kg (> P85)")

        perfiles.insert(0, {
            "tipo": "sindrome_metabolico",
            "titulo": "Perfil compatible con Síndrome Metabólico",
            "subtitulo": " + ".join(componentes),
            "icono": "alert-octagon",
            "color": "#dc2626",
            "bg": "#fef2f2",
            "border": "#fca5a5",
            "descripcion": (
                f"Se detectan {factores_metabolicos} factores de riesgo combinados "
                f"({', '.join(componentes)}), configurando un perfil compatible con "
                "síndrome metabólico pediátrico."
            ),
            "acciones": [
                "Evaluación integral: perfil lipídico, glucosa, insulina, HbA1c",
                "Medición de circunferencia abdominal e índice cintura-talla",
                "Referencia a nutrición y endocrinología pediátrica",
                "Plan de intervención multidisciplinario con seguimiento trimestral",
            ],
        })

    return perfiles


@callback(
    Output("sim-resultado-panel", "children"),
    Output("sim-semaforo", "children"),
    Output("sim-feature-importance", "children"),
    Output("sim-diagnostico", "children"),
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
    """Ejecuta predicción y muestra resultado unificado con diagnóstico clínico."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    if any(v is None for v in [genero, edad, peso, pas, fc, colesterol]):
        warn = html.Div([
            icon(SECTION_ICONS["alert_warning"], size=18, color="#92400e"),
            html.Span(" Complete todos los campos.", style={"marginLeft": "8px"}),
        ], className="alert-box alert-warning")
        return warn, "", "", ""

    result = predict_risk(genero, edad, peso, pas, fc, colesterol)
    genero_label = "Femenino" if genero == 0 else "Masculino"

    nivel = result["nivel"]
    if nivel == "bajo":
        sem_cls, sem_ico, ico_c, lbl, txt_c = (
            "semaforo-verde", SEMAFORO_ICONS["bajo"], "#065f46", "RIESGO BAJO", "#065f46")
    elif nivel == "moderado":
        sem_cls, sem_ico, ico_c, lbl, txt_c = (
            "semaforo-amarillo", SEMAFORO_ICONS["moderado"], "#92400e", "RIESGO MODERADO", "#92400e")
    else:
        sem_cls, sem_ico, ico_c, lbl, txt_c = (
            "semaforo-rojo", SEMAFORO_ICONS["alto"], "#991b1b", "RIESGO ALTO", "#991b1b")

    importances = result["feature_importances"]
    probabilidad = result["probabilidad"]

    # Evaluación clínica basada en umbrales reales
    perfiles = _evaluar_perfil_clinico(edad, peso, pas, fc, colesterol)

    # ─── Panel unificado ─────────────────────────────────────
    panel = html.Div([
        # Fila 1: Semáforo + Diagnóstico clínico lado a lado
        html.Div([
            # Semáforo
            html.Div([
                html.Div([
                    html.Span(icon(sem_ico, size=48, color=ico_c),
                              className="semaforo-icon"),
                    html.Div(lbl, className="semaforo-label",
                             style={"fontSize": "16px"}),
                    html.Div(f"{result['porcentaje']}%",
                             className="semaforo-prob",
                             style={"fontSize": "32px"}),
                    html.Div("Probabilidad estimada de riesgo CV",
                             className="semaforo-sublabel",
                             style={"fontSize": "11px"}),
                ], className=f"semaforo-container {sem_cls}",
                   style={"padding": "20px"}),

                # Leyenda de interpretación
                html.Div([
                    _prob_legend("0-30%", "Riesgo bajo", "#059669"),
                    _prob_legend("31-69%", "Riesgo moderado", "#d97706"),
                    _prob_legend("≥70%", "Riesgo alto", "#dc2626"),
                ], style={"marginTop": "10px", "padding": "8px",
                          "background": "#f9fafb", "borderRadius": "8px"}),
            ], style={"flex": "1", "minWidth": "200px"}),

            # Panel de diagnóstico clínico
            html.Div(
                _build_diagnostico_panel(perfiles, nivel, probabilidad),
                style={"flex": "1.3", "minWidth": "300px"},
            ),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "16px"}),

        # Fila 2: Paciente + Variables
        html.Div([
            # Datos del paciente con semáforos por variable
            html.Div([
                html.Div([
                    icon(SECTION_ICONS["patient"], size=16, color="#1a4076"),
                    html.Span(" Valores del Paciente",
                              style={"marginLeft": "4px", "fontSize": "13px",
                                     "fontWeight": "600", "color": "#1a4076"}),
                ], style={"marginBottom": "10px"}),
                html.Div([
                    _pf_with_status("Sexo", genero_label, "normal"),
                    _pf_with_status("Edad", f"{edad} años", "normal"),
                    _pf_with_status("Peso", f"{peso} kg",
                                    _peso_status(edad, peso)),
                    _pf_with_status("PAS", f"{pas} mmHg",
                                    _pas_status(pas)),
                    _pf_with_status("FC", f"{fc} lpm",
                                    _fc_status(fc)),
                    _pf_with_status("Colesterol", f"{colesterol} mg/dL",
                                    _col_status(colesterol)),
                ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                          "gap": "6px"}),
            ], style={"flex": "1", "padding": "14px", "background": "#f9fafb",
                      "borderRadius": "10px", "border": "1px solid #e5e7eb"}),

            # Feature importance
            html.Div([
                html.Div([
                    icon(SECTION_ICONS["feature_importance"], size=16, color="#1a4076"),
                    html.Span(" Contribución al Modelo",
                              style={"marginLeft": "4px", "fontSize": "13px",
                                     "fontWeight": "600", "color": "#1a4076"}),
                ], style={"marginBottom": "10px"}),
                *_build_importance_bars(importances),
            ], style={"flex": "1", "padding": "14px", "background": "#f9fafb",
                      "borderRadius": "10px", "border": "1px solid #e5e7eb"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # Disclaimer
        html.Div([
            icon("information-outline", size=14, color="#9ca3af"),
            html.Span(
                " El modelo predice la probabilidad global de riesgo cardiovascular. "
                "La interpretación clínica se basa en umbrales fisiológicos reconocidos "
                "(AAP, OMS/CDC). No constituye diagnóstico médico.",
                style={"marginLeft": "6px", "fontSize": "11px",
                       "color": "#9ca3af", "fontStyle": "italic"},
            ),
        ], style={"marginTop": "16px", "display": "flex",
                  "alignItems": "flex-start"}),
    ])

    return panel, "", "", ""


# ═══════════════════════════════════════════════════════════════
# COMPONENTES VISUALES
# ═══════════════════════════════════════════════════════════════

def _build_diagnostico_panel(perfiles, nivel, probabilidad):
    """Panel de diagnóstico clínico basado en umbrales reales."""
    if not perfiles and nivel == "bajo":
        return html.Div([
            html.Div([
                icon("shield-check", size=28, color="#059669"),
                html.Span(" Sin alertas clínicas",
                          style={"marginLeft": "8px", "fontSize": "15px",
                                 "fontWeight": "700", "color": "#065f46"}),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "12px"}),
            html.P("Todos los valores se encuentran dentro de rangos normales "
                   "para población pediátrica. Mantener controles de rutina.",
                   style={"fontSize": "13px", "color": "#4b5563",
                          "lineHeight": "1.5", "margin": "0"}),
        ], style={
            "padding": "20px", "background": "#d1fae5",
            "border": "2px solid #6ee7b7", "borderRadius": "12px",
            "height": "100%", "display": "flex", "flexDirection": "column",
            "justifyContent": "center",
        })

    if not perfiles:
        return html.Div([
            html.Div([
                icon("check-circle-outline", size=28, color="#059669"),
                html.Span(" Valores en rango normal",
                          style={"marginLeft": "8px", "fontSize": "15px",
                                 "fontWeight": "700", "color": "#374151"}),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "8px"}),
            html.P("Los valores clínicos individuales están dentro de rangos "
                   "aceptables. El riesgo detectado proviene de la combinación "
                   "de factores evaluada por el modelo.",
                   style={"fontSize": "13px", "color": "#4b5563",
                          "lineHeight": "1.5", "margin": "0"}),
        ], style={
            "padding": "20px", "background": "#f9fafb",
            "border": "1px solid #e5e7eb", "borderRadius": "12px",
            "height": "100%",
        })

    # Mostrar el perfil principal (más grave primero)
    principal = perfiles[0]
    otros = perfiles[1:]

    children = [
        html.Div([
            icon("stethoscope", size=18, color="#1a4076"),
            html.Span(" Interpretación Clínica",
                      style={"marginLeft": "6px", "fontSize": "14px",
                             "fontWeight": "700", "color": "#1a4076"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "12px"}),

        # Perfil principal
        _perfil_card(principal),
    ]

    # Otros hallazgos (compactos)
    if otros:
        # Filtrar duplicados por tipo
        tipos_vistos = {principal["tipo"]}
        otros_unicos = []
        for p in otros:
            if p["tipo"] not in tipos_vistos:
                tipos_vistos.add(p["tipo"])
                otros_unicos.append(p)

        if otros_unicos:
            children.append(html.Div([
                html.Div("Otros hallazgos:", style={
                    "fontSize": "11px", "fontWeight": "700",
                    "color": "#6b7280", "marginBottom": "6px",
                    "marginTop": "10px",
                }),
                *[_perfil_mini(p) for p in otros_unicos[:3]],
            ]))

    return html.Div(children, style={
        "padding": "16px", "background": "#ffffff",
        "border": "1px solid #e5e7eb", "borderRadius": "12px",
        "height": "100%",
    })


def _perfil_card(perfil):
    """Tarjeta del perfil clínico principal."""
    return html.Div([
        html.Div([
            icon(perfil["icono"], size=20, color=perfil["color"]),
            html.Div([
                html.Div(perfil["titulo"], style={
                    "fontSize": "13px", "fontWeight": "700",
                    "color": perfil["color"],
                }),
                html.Div(perfil["subtitulo"], style={
                    "fontSize": "11px", "color": "#6b7280", "marginTop": "2px",
                }),
            ], style={"marginLeft": "10px", "flex": "1"}),
        ], style={"display": "flex", "alignItems": "flex-start",
                  "marginBottom": "8px"}),

        html.P(perfil["descripcion"], style={
            "fontSize": "12px", "color": "#4b5563",
            "lineHeight": "1.5", "margin": "0 0 8px 0",
        }),

        html.Div([
            html.Div("Acciones recomendadas:", style={
                "fontSize": "11px", "fontWeight": "700",
                "color": "#374151", "marginBottom": "4px",
            }),
            html.Ul([
                html.Li(a, style={"fontSize": "11px", "color": "#4b5563",
                                  "marginBottom": "2px", "lineHeight": "1.4"})
                for a in perfil["acciones"]
            ], style={"paddingLeft": "16px", "margin": "0"}),
        ], style={"padding": "8px 10px", "background": perfil["bg"],
                  "borderRadius": "6px", "border": f"1px solid {perfil['border']}"}),
    ])


def _perfil_mini(perfil):
    """Mini-tarjeta para hallazgos secundarios."""
    return html.Div([
        icon(perfil["icono"], size=14, color=perfil["color"]),
        html.Span(f" {perfil['titulo']}", style={
            "marginLeft": "4px", "fontSize": "11px",
            "fontWeight": "600", "color": perfil["color"],
        }),
        html.Span(f" — {perfil['subtitulo']}", style={
            "fontSize": "11px", "color": "#6b7280",
        }),
    ], style={"display": "flex", "alignItems": "center",
              "marginBottom": "4px", "padding": "4px 8px",
              "background": perfil["bg"], "borderRadius": "4px",
              "border": f"1px solid {perfil['border']}"})


def _prob_legend(rango, texto, color):
    """Fila de leyenda de probabilidad."""
    return html.Div([
        html.Span("●", style={"color": color, "fontSize": "10px",
                               "marginRight": "6px"}),
        html.Span(rango, style={"fontSize": "10px", "fontWeight": "700",
                                "color": "#374151", "minWidth": "45px",
                                "display": "inline-block"}),
        html.Span(texto, style={"fontSize": "10px", "color": "#6b7280"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "2px"})


def _pf_with_status(label, value, status):
    """Campo de paciente con indicador de estado clínico."""
    colors = {
        "normal": ("#059669", "#d1fae5", "#6ee7b7"),
        "limite": ("#d97706", "#fef3c7", "#fcd34d"),
        "alto": ("#dc2626", "#fee2e2", "#fca5a5"),
    }
    txt_c, bg_c, brd_c = colors.get(status, colors["normal"])
    dot = "●" if status != "normal" else ""

    return html.Div([
        html.Div([
            html.Span(f"{label}: ", style={"fontSize": "11px", "color": "#6b7280"}),
            html.Span(value, style={"fontSize": "11px", "fontWeight": "600",
                                    "color": "#374151"}),
        ]),
        html.Span(dot, style={"fontSize": "8px", "color": txt_c,
                               "position": "absolute", "top": "4px", "right": "6px"})
        if dot else None,
    ], style={"padding": "5px 8px", "background": bg_c,
              "borderRadius": "4px", "border": f"1px solid {brd_c}",
              "position": "relative"})


def _col_status(col):
    if col >= 200: return "alto"
    if col >= 170: return "limite"
    return "normal"

def _pas_status(pas):
    if pas >= 130: return "alto"
    if pas >= 110: return "limite"
    return "normal"

def _fc_status(fc):
    if fc > 100 or fc < 60: return "alto"
    return "normal"

def _peso_status(edad, peso):
    edad_int = max(6, min(17, int(edad)))
    p85, p95 = PESO_PERCENTILES.get(edad_int, (50, 60))
    if peso >= p95: return "alto"
    if peso >= p85: return "limite"
    return "normal"


def _build_importance_bars(importances):
    """Barras horizontales compactas de feature importance."""
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    max_val = max(v for _, v in sorted_imp) if sorted_imp else 1

    bars = []
    for feat, val in sorted_imp:
        bars.append(html.Div([
            html.Div([
                html.Span(FEATURE_LABELS.get(feat, feat),
                          style={"fontSize": "11px", "color": "#374151"}),
                html.Span(f"{val:.3f}",
                          style={"fontSize": "11px", "fontWeight": "700",
                                 "color": "#1a4076"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "marginBottom": "2px"}),
            html.Div(
                html.Div(style={
                    "width": f"{(val / max_val) * 100}%",
                    "height": "6px", "borderRadius": "3px",
                    "background": "linear-gradient(90deg, #4682B4, #3a6fa8)",
                }),
                style={"width": "100%", "height": "6px",
                       "borderRadius": "3px", "background": "#e5e7eb"},
            ),
        ], style={"marginBottom": "8px"}))
    return bars


def _pf(label, value):
    """Campo ultra-compacto."""
    return html.Div([
        html.Span(f"{label}: ", style={"fontSize": "11px", "color": "#6b7280"}),
        html.Span(value, style={"fontSize": "11px", "fontWeight": "600",
                                "color": "#374151"}),
    ], style={"padding": "5px 8px", "background": "#fff",
              "borderRadius": "4px", "border": "1px solid #e5e7eb"})
