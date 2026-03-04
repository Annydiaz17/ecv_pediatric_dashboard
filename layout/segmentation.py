"""
Pestaña 2 — SEGMENTACIÓN AVANZADA (Clustering K-Means y PCA)
Scatter PCA 2D, barras de clústeres, perfiles de cluster.
"""
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from utils.model_loader import get_metrics
from utils.data_loader import get_data, TARGET
from utils.icons import icon, SECTION_ICONS


def create_segmentation_layout():
    """Genera el layout de la página de Segmentación."""
    try:
        metrics = get_metrics()
    except FileNotFoundError:
        return html.Div([
            html.Div([
                html.H1([
                    icon("alert-outline", size=28, color="#d97706"),
                    " Datos no encontrados",
                ], className="page-title"),
            ], className="page-header"),
        ])

    clustering = metrics.get("clustering", {})
    pca_info = metrics.get("pca", {})
    clusters = clustering.get("clusters", [])

    return html.Div([
        # Header
        html.Div([
            html.H1("Segmentación Avanzada", className="page-title"),
            html.P("Análisis no supervisado: Clustering K-Means + PCA — SEMMA Phase 2B",
                   className="page-subtitle"),
        ], className="page-header"),

        # Info box
        html.Div([
            icon(SECTION_ICONS["info"], size=18, color="#1e40af"),
            html.Span(
                " El clustering se aplica sobre el dataset limpio completo con un scaler independiente. "
                "Al ser un análisis descriptivo (no predictivo), los clusters no se usan como features "
                "en los modelos y no constituyen data leakage.",
                style={"marginLeft": "8px"},
            ),
        ], className="alert-box alert-info", style={"marginBottom": "24px"}),

        # ═══════════════ PCA 2D Scatter ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["pca"], size=18, color="#1a4076"),
                    html.Span(" Visualización PCA en 2D — Separación de Pacientes",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),

            html.Div([
                html.Div([
                    html.Strong("PC1"),
                    html.Span(f" ({pca_info.get('pc1_variance', 51.17)}% varianza): "),
                    html.Span(pca_info.get('pc1_label', 'Desarrollo Físico'),
                              style={"color": "#4682B4", "fontWeight": "600"}),
                ], style={"marginBottom": "4px"}),
                html.Div([
                    html.Strong("PC2"),
                    html.Span(f" ({pca_info.get('pc2_variance', 16.83)}% varianza): "),
                    html.Span(pca_info.get('pc2_label', 'Riesgo Metabólico'),
                              style={"color": "#CD5C5C", "fontWeight": "600"}),
                ]),
            ], style={"fontSize": "13px", "color": "#4b5563", "marginBottom": "16px"}),

            # Dropdown para colorear PCA
            html.Div([
                html.Label("Colorear por:", className="filter-title",
                           style={"marginRight": "12px"}),
                dcc.Dropdown(
                    id="pca-color-selector",
                    options=[
                        {"label": "Cluster K-Means", "value": "cluster"},
                        {"label": "Riesgo CV Real", "value": "riesgo"},
                    ],
                    value="cluster",
                    clearable=False,
                    style={"maxWidth": "250px", "fontFamily": "Inter, sans-serif"},
                ),
            ], style={"marginBottom": "16px"}),

            dcc.Loading(
                dcc.Graph(id="pca-scatter", config={"displayModeBar": True}),
                type="circle",
                color="#1a4076",
            ),
        ], className="card"),

        # ═══════════════ Barras de Clústeres ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["clustering"], size=18, color="#1a4076"),
                    html.Span(" Resultados del Clustering K-Means (K=4 óptimo)",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),

            # Cluster cards
            html.Div([
                _build_cluster_card(c, is_highlight=(c["id"] == 1))
                for c in clusters
            ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                "gap": "16px", "marginBottom": "24px",
            }),

            # Gráfico de barras: prevalencia por cluster
            dcc.Graph(
                id="cluster-prevalence-chart",
                figure=_build_cluster_prevalence(clusters),
                config={"displayModeBar": False},
            ),
        ], className="card"),

        # ═══════════════ Perfil Clínico de Clústeres ═══════════════
        html.Div([
            html.Div([
                html.Span([
                    icon(SECTION_ICONS["cluster_profile"], size=18, color="#1a4076"),
                    html.Span(" Perfil Clínico Promedio por Cluster",
                              style={"marginLeft": "6px"}),
                ], className="card-title"),
            ], className="card-header"),
            dcc.Graph(
                id="cluster-profile-heatmap",
                figure=_build_cluster_profile_heatmap(clusters),
                config={"displayModeBar": False},
            ),
        ], className="card"),
    ])


def _build_cluster_card(cluster, is_highlight=False):
    """Tarjeta individual de cluster."""
    border_color = "#CD5C5C" if is_highlight else "#e5e7eb"
    bg_color = "#fef2f2" if is_highlight else "#f9fafb"
    badge = ""
    if is_highlight:
        badge = html.Div(
            "⚠ Mayor Riesgo",
            style={
                "background": "#fee2e2", "color": "#991b1b",
                "fontSize": "11px", "fontWeight": "700",
                "padding": "2px 10px", "borderRadius": "12px",
                "display": "inline-block", "marginBottom": "8px",
            }
        )

    return html.Div([
        badge,
        html.Div(f"Cluster {cluster['id']}", style={
            "fontSize": "18px", "fontWeight": "700",
            "color": "#1a4076", "marginBottom": "4px",
        }),
        html.Div(f"{cluster['n_pacientes']:,} pacientes ({cluster['pct']}%)", style={
            "fontSize": "13px", "color": "#6b7280", "marginBottom": "12px",
        }),
        _cluster_stat("Riesgo CV", f"{cluster['pct_riesgo']}%",
                       color="#dc2626" if cluster['pct_riesgo'] > 25 else "#4b5563"),
        _cluster_stat("Edad media", f"{cluster['edad_mean']} años"),
        _cluster_stat("Peso medio", f"{cluster['peso_mean']} kg"),
        _cluster_stat("Colesterol", f"{cluster['colesterol_mean']} mg/dL"),
        _cluster_stat("PAS", f"{cluster['pa_sistolica_mean']} mmHg"),
    ], style={
        "padding": "16px",
        "background": bg_color,
        "borderRadius": "12px",
        "border": f"2px solid {border_color}",
    })


def _cluster_stat(label, value, color="#374151"):
    """Una fila de estadística dentro de la tarjeta de cluster."""
    return html.Div([
        html.Span(f"{label}: ", style={"fontSize": "12px", "color": "#6b7280"}),
        html.Span(value, style={"fontSize": "12px", "fontWeight": "600", "color": color}),
    ], style={"marginBottom": "4px"})


def _build_cluster_prevalence(clusters):
    """Gráfico de barras: prevalencia de riesgo por cluster."""
    if not clusters:
        return go.Figure()

    ids = [f"Cluster {c['id']}" for c in clusters]
    pcts = [c["pct_riesgo"] for c in clusters]
    colors_list = ["#4682B4", "#CD5C5C", "#2E8B57", "#DAA520"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ids, y=pcts,
        marker_color=colors_list[:len(ids)],
        text=[f"{p}%" for p in pcts],
        textposition="outside",
        textfont=dict(family="Inter", size=13, color="#374151"),
        hovertemplate="<b>%{x}</b><br>Riesgo CV: %{y}%<extra></extra>",
    ))
    # Media global
    media_global = 22.0
    fig.add_hline(y=media_global, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"Media global ({media_global}%)",
                  annotation_position="top right",
                  annotation_font=dict(size=12, color="#dc2626"))

    fig.update_layout(
        yaxis_title="% Pacientes con Riesgo CV",
        xaxis_title="Cluster",
        template="plotly_white",
        font_family="Inter",
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        showlegend=False,
    )
    return fig


def _build_cluster_profile_heatmap(clusters):
    """Heatmap del perfil clínico promedio por cluster."""
    if not clusters:
        return go.Figure()

    variables = ["Edad", "Peso (kg)", "Colesterol", "PAS", "FC", "% Riesgo"]
    cluster_ids = [f"Cluster {c['id']}" for c in clusters]

    z_vals = []
    text_vals = []
    for c in clusters:
        row = [c["edad_mean"], c["peso_mean"], c["colesterol_mean"],
               c["pa_sistolica_mean"], c["fc_mean"], c["pct_riesgo"]]
        z_vals.append(row)
        text_vals.append([str(v) for v in row])

    # Normalize for color
    z_arr = np.array(z_vals, dtype=float)
    z_norm = np.zeros_like(z_arr)
    for j in range(z_arr.shape[1]):
        col_min = z_arr[:, j].min()
        col_max = z_arr[:, j].max()
        if col_max > col_min:
            z_norm[:, j] = (z_arr[:, j] - col_min) / (col_max - col_min)
        else:
            z_norm[:, j] = 0.5

    fig = go.Figure(data=go.Heatmap(
        z=z_norm,
        x=variables,
        y=cluster_ids,
        text=text_vals,
        texttemplate="%{text}",
        textfont={"size": 13, "family": "Inter"},
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="Normalizado"),
        hovertemplate="<b>%{y}</b> — %{x}<br>Valor: %{text}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        font_family="Inter",
        margin=dict(l=20, r=20, t=20, b=40),
        height=280,
    )
    return fig
