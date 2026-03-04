"""
Callbacks para la Pestaña 1 — Resumen y EDA.
Histogramas comparativos, boxplots y matriz de correlación.
"""
from dash import Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.data_loader import get_data, TARGET

PLOT_TEMPLATE = "plotly_white"
COLOR_DISCRETE = ["#4682B4", "#CD5C5C"]

FEATURE_LABELS = {
    "edad": "Edad (años)",
    "genero": "Género (0=F, 1=M)",
    "peso_kg": "Peso (kg)",
    "pa_sistolica": "PA Sistólica (mmHg)",
    "colesterol_mgdl": "Colesterol (mg/dL)",
    "frecuencia_cardiaca": "Frecuencia Cardíaca (lpm)",
}


@callback(
    Output("eda-histogram", "figure"),
    Output("eda-boxplot", "figure"),
    Output("eda-correlation", "figure"),
    Input("eda-variable-selector", "value"),
)
def update_eda(variable):
    """Actualiza histograma, boxplot y correlación."""
    df = get_data()
    n_total = len(df)
    var_label = FEATURE_LABELS.get(variable, variable)

    # ─── Histograma ──────────────────────────────────────────
    if n_total > 0:
        df_hist = df.copy()
        df_hist["Clase"] = df_hist[TARGET].map({0: "Sin Riesgo", 1: "Con Riesgo"})
        fig_hist = px.histogram(
            df_hist, x=variable, color="Clase",
            barmode="overlay", opacity=0.75,
            color_discrete_sequence=COLOR_DISCRETE,
            labels={variable: var_label},
            template=PLOT_TEMPLATE,
        )
        fig_hist.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
            font_family="Inter",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            xaxis_title=var_label,
            yaxis_title="Frecuencia",
            title=dict(text=f"Distribución de {var_label} por Riesgo",
                       font=dict(size=14)),
        )
    else:
        fig_hist = _empty_figure("Sin datos")

    # ─── Boxplot ─────────────────────────────────────────────
    if n_total > 0:
        df_box = df.copy()
        df_box["Clase"] = df_box[TARGET].map({0: "Sin Riesgo", 1: "Con Riesgo"})
        fig_box = px.box(
            df_box, x="Clase", y=variable, color="Clase",
            color_discrete_sequence=COLOR_DISCRETE,
            labels={variable: var_label},
            template=PLOT_TEMPLATE,
            points="outliers",
        )
        fig_box.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
            font_family="Inter",
            showlegend=False,
            yaxis_title=var_label,
            title=dict(text=f"Boxplot de {var_label} por Riesgo",
                       font=dict(size=14)),
        )
    else:
        fig_box = _empty_figure("Sin datos")

    # ─── Correlación ─────────────────────────────────────────
    if n_total > 5:
        numeric_cols = ["edad", "peso_kg", "pa_sistolica",
                        "colesterol_mgdl", "frecuencia_cardiaca", TARGET]
        corr = df[numeric_cols].corr()

        labels_corr = ["Edad", "Peso", "PAS", "Colesterol", "FC", "Riesgo CV"]
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=labels_corr,
            y=labels_corr,
            colorscale=[
                [0, "#2563eb"],
                [0.5, "#ffffff"],
                [1, "#dc2626"]
            ],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 13, "family": "Inter"},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            font_family="Inter",
            template=PLOT_TEMPLATE,
            height=480,
        )
    else:
        fig_corr = _empty_figure("Datos insuficientes")

    return fig_hist, fig_box, fig_corr


def _empty_figure(msg="Sin datos"):
    """Crea una figura vacía con un mensaje."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#9ca3af", family="Inter"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
    )
    return fig
