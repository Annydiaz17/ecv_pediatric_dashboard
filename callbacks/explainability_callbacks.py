"""
Callbacks para la página de Explicabilidad (SHAP).
Adaptado para LogisticRegression + StandardScaler del notebook SEMMA.
"""
from dash import Input, Output, State, callback, html, no_update, dcc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

from utils.model_loader import get_model, get_scaler, FEATURE_LABELS, FEATURES_MODELO
from utils.data_loader import get_data, TARGET

PLOT_TEMPLATE = "plotly_white"

# Cache del explainer SHAP
_shap_cache = {}


def _get_shap_explainer():
    """Crea o cachea el explainer SHAP para LogisticRegression."""
    import shap

    if "explainer" not in _shap_cache:
        model = get_model()
        scaler = get_scaler()

        # Obtener datos de Timbiquí para background
        df = get_data("timbiqui")
        X = df[FEATURES_MODELO].values

        # Escalar
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            X_scaled = scaler.transform(X)

        # LinearExplainer para modelos lineales (más rápido que KernelExplainer)
        try:
            explainer = shap.LinearExplainer(model, X_scaled)
        except Exception:
            # Fallback a KernelExplainer con background reducido
            background = shap.sample(X_scaled, min(100, len(X_scaled)))
            explainer = shap.KernelExplainer(model.predict_proba, background)

        _shap_cache["explainer"] = explainer

    return _shap_cache["explainer"]


def _get_test_shap_values():
    """Calcula SHAP values para el set de test."""
    if "test_shap_values" not in _shap_cache:
        model = get_model()
        scaler = get_scaler()

        df = get_data("timbiqui")
        X = df[FEATURES_MODELO].values
        y = df[TARGET].values

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            X_test_scaled = scaler.transform(X_test)

        explainer = _get_shap_explainer()
        shap_values = explainer.shap_values(X_test_scaled)

        # Para LinearExplainer: puede devolver array o lista
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Clase positiva
        else:
            shap_vals = shap_values

        _shap_cache["test_shap_values"] = shap_vals
        _shap_cache["X_test"] = X_test
        _shap_cache["X_test_scaled"] = X_test_scaled

    return _shap_cache["test_shap_values"], _shap_cache["X_test"]


@callback(
    Output("shap-summary-plot", "figure"),
    Input("url", "pathname"),
)
def update_shap_summary(pathname):
    """Genera el SHAP summary plot global."""
    if pathname != "/explainability":
        return go.Figure()

    try:
        shap_vals, X_test = _get_test_shap_values()
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error calculando SHAP: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#dc2626", family="Inter"),
        )
        return fig

    features = FEATURES_MODELO
    feature_labels = [FEATURE_LABELS[f] for f in features]

    # Mean absolute SHAP values para ordenamiento
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs_shap).tolist()

    fig = go.Figure()

    tick_vals = []
    tick_text = []

    # Beeswarm-like: scatter por feature
    for i, idx in enumerate(order):
        feat = features[idx]
        feat_label = FEATURE_LABELS[feat]
        sv_feature = shap_vals[:, idx]

        tick_vals.append(i)
        tick_text.append(feat_label)

        # Valores originales para color
        feat_values = X_test[:, idx].astype(float)

        # Normalizar para colorscale
        fmin, fmax = feat_values.min(), feat_values.max()
        if fmax > fmin:
            feat_norm = (feat_values - fmin) / (fmax - fmin)
        else:
            feat_norm = np.zeros_like(feat_values)

        # Jitter vertical
        jitter = np.random.RandomState(42).normal(0, 0.15, len(sv_feature))

        fig.add_trace(go.Scatter(
            x=sv_feature,
            y=np.full(len(sv_feature), i) + jitter,
            mode="markers",
            marker=dict(
                size=5,
                color=feat_norm,
                colorscale=[[0, "#3a6fa8"], [1, "#dc2626"]],
                opacity=0.6,
            ),
            name=feat_label,
            showlegend=False,
            hovertemplate=(
                f"<b>{feat_label}</b><br>"
                "SHAP: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title="SHAP Value (impacto en la predicción)",
        yaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            zeroline=False,
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        template=PLOT_TEMPLATE,
        font_family="Inter",
        margin=dict(l=20, r=40, t=30, b=50),
        height=max(350, 55 * len(features) + 80),
        xaxis=dict(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=2),
    )

    return fig


@callback(
    Output("shap-waterfall-container", "children"),
    Output("shap-bar-individual", "children"),
    Input("shap-compute-btn", "n_clicks"),
    State("shap-sexo", "value"),
    State("shap-edad", "value"),
    State("shap-peso", "value"),
    State("shap-pas", "value"),
    State("shap-fc", "value"),
    State("shap-colesterol", "value"),
    prevent_initial_call=True,
)
def compute_individual_shap(n_clicks, genero, edad, peso, pas, fc, colesterol):
    """Calcula SHAP individual para un paciente específico."""
    if not n_clicks:
        return no_update, no_update

    if any(v is None for v in [genero, edad, peso, pas, fc, colesterol]):
        return html.Div("Complete todos los campos.",
                       className="alert-box alert-warning"), ""

    try:
        import shap

        model = get_model()
        scaler = get_scaler()
        explainer = _get_shap_explainer()
        features = FEATURES_MODELO
        feature_labels_list = [FEATURE_LABELS[f] for f in features]

        # Orden del modelo: edad, genero, peso_kg, pa_sistolica, fc, colesterol
        X_patient = np.array([[
            float(edad), float(genero), float(peso),
            float(pas), float(fc), float(colesterol)
        ]])
        values_display = [edad, genero, peso, pas, fc, colesterol]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            X_patient_scaled = scaler.transform(X_patient)

        shap_values = explainer.shap_values(X_patient_scaled)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    except Exception as e:
        return html.Div(
            f"Error calculando SHAP individual: {str(e)}",
            className="alert-box alert-danger"
        ), ""

    # ─── Waterfall Chart ─────────────────────────────────────
    sorted_idx = np.argsort(np.abs(sv))[::-1].tolist()

    fig_waterfall = go.Figure()

    # Base value
    fig_waterfall.add_trace(go.Bar(
        x=[base_value],
        y=["Base Value"],
        orientation="h",
        marker_color="#6b7280",
        text=[f"{base_value:.4f}"],
        textposition="outside",
        textfont=dict(family="Inter", size=12),
        name="Base Value",
        showlegend=False,
    ))

    # SHAP contributions
    for idx in sorted_idx:
        color = "#dc2626" if sv[idx] > 0 else "#3a6fa8"
        label = f"{feature_labels_list[idx]} = {values_display[idx]}"

        fig_waterfall.add_trace(go.Bar(
            x=[sv[idx]],
            y=[label],
            orientation="h",
            marker_color=color,
            text=[f"{sv[idx]:+.4f}"],
            textposition="outside",
            textfont=dict(family="Inter", size=11),
            name=label,
            showlegend=False,
        ))

    prediction = base_value + sv.sum()
    fig_waterfall.add_annotation(
        text=f"<b>f(x) = {prediction:.4f}</b>",
        xref="paper", yref="paper",
        x=0.98, y=-0.08,
        showarrow=False,
        font=dict(size=14, family="Inter", color="#1a4076"),
    )

    fig_waterfall.update_layout(
        xaxis_title="SHAP Value",
        template=PLOT_TEMPLATE,
        font_family="Inter",
        margin=dict(l=20, r=80, t=20, b=60),
        height=max(300, 50 * len(features) + 100),
        barmode="relative",
    )

    waterfall_section = html.Div([
        html.Div(
            f"Valor base (E[f(x)]): {base_value:.4f} -> Prediccion: {prediction:.4f}",
            style={
                "textAlign": "center", "fontSize": "14px",
                "color": "#374151", "marginBottom": "12px",
                "fontWeight": "600",
            }
        ),
        dcc.Graph(figure=fig_waterfall, config={"displayModeBar": False}),
    ])

    # ─── Bar Plot Individual ─────────────────────────────────
    sorted_by_value = sorted(
        zip(feature_labels_list, sv, values_display),
        key=lambda x: abs(x[1])
    )

    fig_bar = go.Figure()
    for label, shap_val, feat_val in sorted_by_value:
        color = "#dc2626" if shap_val > 0 else "#3a6fa8"
        fig_bar.add_trace(go.Bar(
            x=[shap_val],
            y=[f"{label} = {feat_val}"],
            orientation="h",
            marker_color=color,
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>SHAP: {shap_val:+.4f}<extra></extra>",
        ))

    fig_bar.update_layout(
        xaxis_title="SHAP Value (contribucion a la prediccion)",
        template=PLOT_TEMPLATE,
        font_family="Inter",
        margin=dict(l=20, r=40, t=20, b=50),
        height=max(250, 45 * len(features) + 80),
        xaxis=dict(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=2),
    )

    bar_section = dcc.Graph(figure=fig_bar, config={"displayModeBar": False})

    return waterfall_section, bar_section
