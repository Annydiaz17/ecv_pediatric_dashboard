"""
Página 2 — EXPLORACIÓN DE DATOS (EDA)
Filtros interactivos, histogramas, boxplots, correlación, y tabla exportable.
"""
from dash import html, dcc, dash_table
from utils.data_loader import get_data
from utils.icons import icon, SECTION_ICONS


def create_eda_layout():
    """Genera el layout de la página de exploración de datos."""
    try:
        df = get_data("combinado")
    except FileNotFoundError:
        return html.Div([
            html.Div([
                html.H1([
                    icon("alert-outline", size=28, color="#d97706"),
                    " Dataset no encontrado",
                ], className="page-title"),
                html.P("Verifique que los archivos de datos estén en la carpeta 'data/'.",
                       className="page-subtitle"),
            ], className="page-header"),
        ])

    return html.Div([
        # Header
        html.Div([
            html.H1("Exploración de Datos", className="page-title"),
            html.P("Análisis interactivo del dataset de riesgo cardiovascular pediátrico",
                   className="page-subtitle"),
        ], className="page-header"),

        # Filtros
        html.Div([
            html.Div([
                icon(SECTION_ICONS["filter"], size=18, color="#1a4076"),
                html.Span(" Filtros Interactivos", style={"marginLeft": "6px"}),
            ], className="card-title", style={"marginBottom": "20px"}),

            html.Div([
                # Género
                html.Div([
                    html.Label("Género", className="filter-title"),
                    dcc.Dropdown(
                        id="eda-filter-sexo",
                        options=[
                            {"label": "Todos", "value": "todos"},
                            {"label": "Femenino (0)", "value": "0"},
                            {"label": "Masculino (1)", "value": "1"},
                        ],
                        value="todos",
                        clearable=False,
                        style={"fontFamily": "Inter, sans-serif"},
                    ),
                ], style={"flex": "1", "minWidth": "150px"}),

                # Edad
                html.Div([
                    html.Label("Edad (años)", className="filter-title"),
                    dcc.RangeSlider(
                        id="eda-filter-edad",
                        min=int(df["edad"].min()),
                        max=min(int(df["edad"].max()), 20),  # Cap at 20 for display
                        step=0.5,
                        value=[int(df["edad"].min()), min(int(df["edad"].max()), 20)],
                        marks={i: str(i) for i in range(
                            int(df["edad"].min()),
                            min(int(df["edad"].max()), 20) + 1, 3
                        )},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "2", "minWidth": "200px"}),

                # Colesterol
                html.Div([
                    html.Label("Colesterol (mg/dL)", className="filter-title"),
                    dcc.RangeSlider(
                        id="eda-filter-colesterol",
                        min=int(df["colesterol_mgdl"].min()),
                        max=int(df["colesterol_mgdl"].max()),
                        step=5,
                        value=[int(df["colesterol_mgdl"].min()),
                               int(df["colesterol_mgdl"].max())],
                        marks={i: str(i) for i in range(
                            int(df["colesterol_mgdl"].min()),
                            int(df["colesterol_mgdl"].max()) + 1, 40
                        )},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "2", "minWidth": "200px"}),
            ], style={
                "display": "flex", "gap": "24px", "flexWrap": "wrap",
                "marginBottom": "16px"
            }),

            html.Div([
                # PAS
                html.Div([
                    html.Label("PA Sistólica (mmHg)", className="filter-title"),
                    dcc.RangeSlider(
                        id="eda-filter-pas",
                        min=int(df["pa_sistolica"].min()),
                        max=int(df["pa_sistolica"].max()),
                        step=5,
                        value=[int(df["pa_sistolica"].min()),
                               int(df["pa_sistolica"].max())],
                        marks={i: str(i) for i in range(
                            int(df["pa_sistolica"].min()),
                            int(df["pa_sistolica"].max()) + 1, 20
                        )},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "1", "minWidth": "200px"}),

                # Peso (reemplaza PAD)
                html.Div([
                    html.Label("Peso (kg)", className="filter-title"),
                    dcc.RangeSlider(
                        id="eda-filter-pad",   # mantener el id para callbacks
                        min=int(df["peso_kg"].min()),
                        max=int(df["peso_kg"].max()),
                        step=1,
                        value=[int(df["peso_kg"].min()),
                               int(df["peso_kg"].max())],
                        marks={i: str(i) for i in range(
                            int(df["peso_kg"].min()),
                            int(df["peso_kg"].max()) + 1, 10
                        )},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "1", "minWidth": "200px"}),

                # FC
                html.Div([
                    html.Label("Frecuencia Cardíaca (lpm)", className="filter-title"),
                    dcc.RangeSlider(
                        id="eda-filter-fc",
                        min=int(df["frecuencia_cardiaca"].min()),
                        max=int(df["frecuencia_cardiaca"].max()),
                        step=5,
                        value=[int(df["frecuencia_cardiaca"].min()),
                                int(df["frecuencia_cardiaca"].max())],
                        marks={i: str(i) for i in range(
                            int(df["frecuencia_cardiaca"].min()),
                            int(df["frecuencia_cardiaca"].max()) + 1, 20
                        )},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "2", "minWidth": "200px"}),


            ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),

            # Counter
            html.Div(id="eda-record-count", style={
                "marginTop": "16px", "padding": "8px 16px",
                "background": "#e8f1f8", "borderRadius": "8px",
                "fontSize": "14px", "fontWeight": "600", "color": "#1a4076",
            }),
        ], className="filter-panel"),

        # Dropdown para seleccionar variable
        html.Div([
            html.Label("Variable para histograma y boxplot:", className="filter-title"),
            dcc.Dropdown(
                id="eda-variable-selector",
                options=[
                    {"label": "Edad", "value": "edad"},
                    {"label": "Peso (kg)", "value": "peso_kg"},
                    {"label": "PA Sistólica", "value": "pa_sistolica"},
                    {"label": "Colesterol (mg/dL)", "value": "colesterol_mgdl"},
                    {"label": "Frecuencia Cardíaca", "value": "frecuencia_cardiaca"},
                ],
                value="colesterol_mgdl",
                clearable=False,
                style={"maxWidth": "300px", "fontFamily": "Inter, sans-serif"},
            ),
        ], style={"marginBottom": "24px"}),

        # Gráficos: Histograma + Boxplot
        html.Div([
            html.Div([
                html.Div([
                    html.Span([
                        icon(SECTION_ICONS["histogram"], size=18, color="#1a4076"),
                        html.Span(" Histograma por Clase", style={"marginLeft": "6px"}),
                    ], className="card-title"),
                ], className="card-header"),
                dcc.Graph(id="eda-histogram", config={"displayModeBar": True}),
            ], className="card", style={"flex": "1"}),

            html.Div([
                html.Div([
                    html.Span([
                        icon(SECTION_ICONS["boxplot"], size=18, color="#1a4076"),
                        html.Span(" Boxplot por Clase", style={"marginLeft": "6px"}),
                    ], className="card-title"),
                ], className="card-header"),
                dcc.Graph(id="eda-boxplot", config={"displayModeBar": True}),
            ], className="card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),

        # Matriz de correlación
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["correlation"], size=18, color="#1a4076"),
                    html.Span(" Matriz de Correlación", style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            dcc.Graph(id="eda-correlation", config={"displayModeBar": True}),
        ], className="card"),

        # Tabla interactiva
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["table"], size=18, color="#1a4076"),
                    html.Span(" Datos del Dataset", style={"marginLeft": "6px"}),
                ], className="card-title"),
                html.Button([
                    icon(SECTION_ICONS["export"], size=16, color="#1a4076"),
                    " Exportar CSV",
                ], id="eda-export-btn",
                   className="btn-secondary", style={"fontSize": "13px"}),
            ], className="card-header"),
            dcc.Download(id="eda-download"),
            dash_table.DataTable(
                id="eda-data-table",
                columns=[
                    {"name": "Género", "id": "genero"},
                    {"name": "Edad", "id": "edad"},
                    {"name": "Peso (kg)", "id": "peso_kg"},
                    {"name": "PA Sistólica", "id": "pa_sistolica"},
                    {"name": "Colesterol", "id": "colesterol_mgdl"},
                    {"name": "FC (lpm)", "id": "frecuencia_cardiaca"},
                    {"name": "Riesgo CV", "id": "riesgo_cv"},
                ],
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "center",
                    "fontFamily": "Inter, sans-serif",
                    "fontSize": "13px",
                    "padding": "10px 14px",
                },
                style_header={
                    "backgroundColor": "#0f2442",
                    "color": "white",
                    "fontWeight": "600",
                    "textAlign": "center",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{riesgo_cv} = 1"},
                        "backgroundColor": "#fee2e2",
                        "color": "#991b1b",
                    },
                    {
                        "if": {"state": "active"},
                        "backgroundColor": "#e8f1f8",
                        "border": "1px solid #1a4076",
                    },
                ],
            ),
        ], className="card"),
    ])
