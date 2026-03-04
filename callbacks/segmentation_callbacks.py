"""
Callbacks para la Pestaña 2 — Segmentación (PCA + Clustering).
Genera el scatter PCA 2D coloreado por cluster o riesgo.
"""
from dash import Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.data_loader import get_data, FEATURES_MODELO, TARGET

PLOT_TEMPLATE = "plotly_white"

# Cache
_seg_cache = {}

CLUSTER_COLORS = ["#4682B4", "#CD5C5C", "#2E8B57", "#DAA520"]
RISK_COLORS = {0: "#4682B4", 1: "#CD5C5C"}


def _compute_pca_clusters():
    """Computa PCA 2D y clusters K-Means (cacheado)."""
    if "pca_df" in _seg_cache:
        return _seg_cache["pca_df"]

    df = get_data()
    X = df[FEATURES_MODELO].values.copy()
    y = df[TARGET].values.copy()

    # Imputar nulos
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)

    # Escalar (scaler propio para clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means K=4
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": cluster_labels.astype(str),
        "riesgo_cv": y.astype(str),
    })

    var1, var2 = pca.explained_variance_ratio_ * 100
    _seg_cache["pca_df"] = df_pca
    _seg_cache["var1"] = var1
    _seg_cache["var2"] = var2

    return df_pca


@callback(
    Output("pca-scatter", "figure"),
    Input("pca-color-selector", "value"),
)
def update_pca_scatter(color_by):
    """Genera scatter PCA coloreado por cluster o riesgo."""
    df_pca = _compute_pca_clusters()
    var1 = _seg_cache.get("var1", 51.17)
    var2 = _seg_cache.get("var2", 16.83)

    if color_by == "riesgo":
        color_col = "riesgo_cv"
        color_map = {"0": "#4682B4", "1": "#CD5C5C"}
        labels_map = {"0": "Sin Riesgo", "1": "Con Riesgo"}
        legend_title = "Riesgo CV"
    else:
        color_col = "cluster"
        color_map = {"0": "#4682B4", "1": "#CD5C5C", "2": "#2E8B57", "3": "#DAA520"}
        labels_map = {"0": "Cluster 0", "1": "Cluster 1", "2": "Cluster 2", "3": "Cluster 3"}
        legend_title = "Cluster"

    df_plot = df_pca.copy()
    df_plot["label"] = df_plot[color_col].map(labels_map)

    fig = px.scatter(
        df_plot, x="PC1", y="PC2",
        color="label",
        color_discrete_map={v: color_map[k] for k, v in labels_map.items()},
        opacity=0.55,
        template=PLOT_TEMPLATE,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        xaxis_title=f"PC1 ({var1:.1f}% varianza)",
        yaxis_title=f"PC2 ({var2:.1f}% varianza)",
        font_family="Inter",
        legend_title=legend_title,
        margin=dict(l=40, r=20, t=30, b=40),
        height=500,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(family="Inter", size=12),
        ),
    )
    return fig
