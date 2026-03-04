"""
Callbacks para la página de Monitoreo y Auditoría.
"""
import os
from dash import Input, Output, State, callback, html, dcc, no_update, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.audit_logger import get_audit_log, get_audit_stats, AUDIT_PATH, AUDIT_COLUMNS
from utils.model_loader import get_model
from utils.metrics import compute_subgroup_metrics
from utils.icons import icon, KPI_ICONS, SECTION_ICONS

PLOT_TEMPLATE = "plotly_white"


@callback(
    Output("monitoring-kpis", "children"),
    Output("monitoring-alerts", "children"),
    Output("monitoring-subgroup-metrics", "children"),
    Output("monitoring-subgroup-chart", "figure"),
    Output("monitoring-prob-dist", "figure"),
    Output("monitoring-audit-table", "children"),
    Input("monitoring-refresh-btn", "n_clicks"),
    Input("url", "pathname"),
)
def update_monitoring(n_clicks, pathname):
    """Actualiza toda la página de monitoreo."""
    if pathname != "/monitoring":
        return [], "", "", go.Figure(), go.Figure(), ""

    stats = get_audit_stats()
    df = get_audit_log()

    # ─── KPIs ────────────────────────────────────────────────
    kpis = [
        html.Div([
            html.Div(
                icon("clipboard-list-outline", size=28, color="#3a6fa8"),
                className="kpi-icon",
            ),
            html.Div(f"{stats['total_predictions']:,}", className="kpi-value"),
            html.Div("Total Predicciones", className="kpi-label"),
        ], className="kpi-card kpi-info"),

        html.Div([
            html.Div(
                icon("alert-octagon", size=28, color="#dc2626"),
                className="kpi-icon",
            ),
            html.Div(f"{stats['high_risk_count']:,}", className="kpi-value"),
            html.Div("Alto Riesgo", className="kpi-label"),
        ], className="kpi-card kpi-danger"),

        html.Div([
            html.Div(
                icon("shield-check-outline", size=28, color="#059669"),
                className="kpi-icon",
            ),
            html.Div(f"{stats['low_risk_count']:,}", className="kpi-value"),
            html.Div("Bajo Riesgo", className="kpi-label"),
        ], className="kpi-card kpi-success"),

        html.Div([
            html.Div(
                icon("chart-donut", size=28, color="#1a4076"),
                className="kpi-icon",
            ),
            html.Div(f"{stats['avg_probability'] * 100:.1f}%", className="kpi-value"),
            html.Div("Prob. Promedio", className="kpi-label"),
        ], className="kpi-card"),
    ]

    # ─── Alertas ─────────────────────────────────────────────
    alerts = []
    if df.empty:
        alerts.append(
            html.Div([
                icon(SECTION_ICONS["alert_info"], size=18, color="#1e40af"),
                html.Span(
                    " No hay predicciones registradas. Use el Simulador Clínico "
                    "para generar predicciones.",
                    style={"marginLeft": "8px"},
                ),
            ], className="alert-box alert-info")
        )
    else:
        # Alerta si hay muchos casos de alto riesgo
        if stats["total_predictions"] > 0:
            high_risk_pct = stats["high_risk_count"] / stats["total_predictions"]
            if high_risk_pct > 0.5:
                alerts.append(html.Div([
                    icon(SECTION_ICONS["alert_warning"], size=18, color="#92400e"),
                    html.Span(
                        f" Atención: {high_risk_pct * 100:.0f}% de las predicciones "
                        "son de alto riesgo. Verificar posible sesgo en la muestra.",
                        style={"marginLeft": "8px"},
                    ),
                ], className="alert-box alert-warning"))

        # Alertas por subgrupo
        if len(df) >= 5:
            df_sex = df.copy()
            df_sex["riesgo_pred"] = (df_sex["clasificacion"] == "Alto Riesgo").astype(int)

            for sex_label in ["Femenino", "Masculino"]:
                mask = df_sex["sexo"] == sex_label
                if mask.sum() > 0:
                    sex_df = df_sex[mask]
                    high_risk_rate = sex_df["riesgo_pred"].mean()
                    if high_risk_rate > 0.6:
                        alerts.append(html.Div([
                            icon(SECTION_ICONS["alert_danger"], size=18, color="#991b1b"),
                            html.Span(
                                f" Alerta de equidad: {high_risk_rate * 100:.0f}% de "
                                f"predicciones para sexo {sex_label} son de alto riesgo.",
                                style={"marginLeft": "8px"},
                            ),
                        ], className="alert-box alert-danger"))

    alerts_div = html.Div(alerts) if alerts else ""

    # ─── Métricas por subgrupo ───────────────────────────────
    subgroup_content = ""
    fig_subgroup = go.Figure()

    if not df.empty and len(df) >= 3:
        df_metrics = df.copy()
        df_metrics["sexo_label"] = df_metrics["sexo"]

        # Tabla resumen por sexo
        summary = df_metrics.groupby("sexo_label").agg(
            N=("probabilidad", "count"),
            Prob_Promedio=("probabilidad", "mean"),
            Alto_Riesgo=("clasificacion", lambda x: (x == "Alto Riesgo").sum()),
        ).reset_index()
        summary.columns = ["Sexo", "N", "Probabilidad Promedio", "Alto Riesgo"]
        summary["Probabilidad Promedio"] = summary["Probabilidad Promedio"].round(4)
        summary["Tasa Alto Riesgo"] = (summary["Alto Riesgo"] / summary["N"]).round(4)

        subgroup_content = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in summary.columns],
            data=summary.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center",
                "fontFamily": "Inter, sans-serif",
                "fontSize": "13px",
                "padding": "10px",
            },
            style_header={
                "backgroundColor": "#0f2442",
                "color": "white",
                "fontWeight": "600",
            },
        )

        # Gráfico
        fig_subgroup = px.bar(
            summary, x="Sexo", y="Tasa Alto Riesgo",
            color="Sexo",
            color_discrete_sequence=["#3a6fa8", "#dc2626"],
            template=PLOT_TEMPLATE,
            labels={"Tasa Alto Riesgo": "Tasa de Alto Riesgo"},
        )
        fig_subgroup.update_layout(
            font_family="Inter",
            margin=dict(l=40, r=20, t=30, b=40),
            showlegend=False,
            height=300,
            yaxis=dict(range=[0, 1], tickformat=".0%"),
        )

    # ─── Distribución de probabilidades ──────────────────────
    fig_prob = go.Figure()
    if not df.empty:
        fig_prob = px.histogram(
            df, x="probabilidad", nbins=20,
            color_discrete_sequence=["#1a4076"],
            template=PLOT_TEMPLATE,
            labels={"probabilidad": "Probabilidad Predicha"},
        )
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color="#dc2626",
                          annotation_text="Umbral 0.5")
        fig_prob.update_layout(
            font_family="Inter",
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Probabilidad de Riesgo Alto",
            yaxis_title="Frecuencia",
            height=300,
        )

    # ─── Tabla de auditoría ──────────────────────────────────
    if not df.empty:
        # Formato de columnas
        df_display = df.copy()
        df_display["sexo"] = df_display["sexo"].str.slice(0, 1)  # "F" o "M"
        df_display["probabilidad"] = df_display["probabilidad"].apply(
            lambda x: f"{x * 100:.1f}%"
        )

        audit_table = dash_table.DataTable(
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Sexo", "id": "sexo"},
                {"name": "Edad", "id": "edad"},
                {"name": "Colesterol", "id": "colesterol_total"},
                {"name": "PAS", "id": "presion_sistolica"},
                {"name": "PAD", "id": "presion_diastolica"},
                {"name": "FC", "id": "frecuencia_cardiaca"},
                {"name": "Probabilidad", "id": "probabilidad"},
                {"name": "Clasificación", "id": "clasificacion"},
            ],
            data=df_display.to_dict("records"),
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center",
                "fontFamily": "Inter, sans-serif",
                "fontSize": "12px",
                "padding": "8px 12px",
            },
            style_header={
                "backgroundColor": "#0f2442",
                "color": "white",
                "fontWeight": "600",
                "fontSize": "12px",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": '{clasificacion} = "Alto Riesgo"'},
                    "backgroundColor": "#fee2e2",
                    "color": "#991b1b",
                },
            ],
        )
    else:
        audit_table = html.Div(
            "No hay predicciones registradas.",
            style={
                "textAlign": "center", "padding": "32px",
                "color": "#9ca3af", "fontSize": "14px",
            }
        )

    return kpis, alerts_div, subgroup_content, fig_subgroup, fig_prob, audit_table


@callback(
    Output("monitoring-download", "data"),
    Input("monitoring-export-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_audit_log(n_clicks):
    """Exporta el log de auditoría como CSV."""
    if not n_clicks:
        return no_update
    df = get_audit_log()
    if df.empty:
        return no_update
    return dcc.send_data_frame(df.to_csv, "audit_log_export.csv", index=False)


@callback(
    Output("monitoring-confirm-clear", "displayed"),
    Input("monitoring-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def show_clear_confirm(n_clicks):
    """Muestra diálogo de confirmación."""
    if n_clicks:
        return True
    return False


@callback(
    Output("monitoring-refresh-btn", "n_clicks"),
    Input("monitoring-confirm-clear", "submit_n_clicks"),
    prevent_initial_call=True,
)
def clear_audit_log(submit_n_clicks):
    """Limpia el log de auditoría tras confirmación."""
    if submit_n_clicks:
        # Recrear archivo vacío
        pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(AUDIT_PATH, index=False)
        return 1  # Trigger refresh
    return no_update
