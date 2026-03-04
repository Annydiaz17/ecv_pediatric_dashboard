"""
ECV Pediátrico Dashboard — Versión Streamlit
=============================================
Dashboard de riesgo cardiovascular pediátrico con 4 pestañas SEMMA.
Para desplegar en Streamlit Community Cloud.

Ejecución local:
    streamlit run streamlit_app.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ECV Pediátrico · Riesgo Cardiovascular",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_MODELO = ["edad", "genero", "peso_kg", "pa_sistolica",
                   "frecuencia_cardiaca", "colesterol_mgdl"]
TARGET = "riesgo_cv"
FEATURE_LABELS = {
    "edad": "Edad (años)", "genero": "Género",
    "peso_kg": "Peso (kg)", "pa_sistolica": "PA Sistólica (mmHg)",
    "frecuencia_cardiaca": "FC (lpm)", "colesterol_mgdl": "Colesterol (mg/dL)",
}
MODEL_COLORS = {
    "Logistic Regression": "#4682B4", "Decision Tree": "#DAA520",
    "K-Nearest Neighbors": "#CD5C5C", "Random Forest": "#2E8B57",
    "Gradient Boosting": "#8B008B",
}
PESO_PERCENTILES = {
    6: (23, 27), 7: (26, 30), 8: (29, 34), 9: (33, 39),
    10: (37, 44), 11: (42, 50), 12: (47, 57), 13: (53, 63),
    14: (58, 69), 15: (62, 74), 16: (65, 78), 17: (68, 82),
}

# ═══════════════════════════════════════════════════════════════
# CARGA DE DATOS Y MODELO (cacheados)
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Carga el dataset Timbiquí."""
    path = os.path.join(BASE_DIR, "data", "dataset_timbiqui.csv")
    df = pd.read_csv(path)
    # Limpiar espacios en nombres de columna
    df.columns = df.columns.str.strip()
    # Mapeo real del CSV: EDAD, genero, Peso_kg, PA_Sistolica, frecuencia_Cardiaca, Colesterol_mgdl, RIESGO
    rename = {
        "EDAD": "edad", "Edad": "edad",
        "genero": "genero", "Genero": "genero",
        "Peso_kg": "peso_kg", "peso_kg": "peso_kg",
        "PA_Sistolica": "pa_sistolica", "pa_sistolica": "pa_sistolica",
        "frecuencia_Cardiaca": "frecuencia_cardiaca", "Frecuencia_Cardiaca": "frecuencia_cardiaca",
        "Colesterol_mgdl": "colesterol_mgdl", "Colesterol_mg_dl": "colesterol_mgdl",
        "RIESGO": "riesgo_cv", "Riesgo_CV": "riesgo_cv", "riesgo_cv": "riesgo_cv",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    # Limpiar género
    def clean_gen(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if s in ("masculino", "m", "masc", "male", "h", "hombre"): return 1
        if s in ("femenino", "f", "fem", "female", "mujer"): return 0
        return np.nan
    if "genero" in df.columns:
        df["genero"] = df["genero"].apply(clean_gen)
    # Limpiar PAS
    if "pa_sistolica" in df.columns:
        df["pa_sistolica"] = df["pa_sistolica"].apply(
            lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan)
    # Filtrar rangos
    rangos = {"edad": (6, 17), "peso_kg": (5, 150), "pa_sistolica": (50, 200),
              "frecuencia_cardiaca": (30, 250), "colesterol_mgdl": (50, 400)}
    for col, (lo, hi) in rangos.items():
        if col in df.columns:
            df = df[df[col].between(lo, hi) | df[col].isna()]
    for col in FEATURES_MODELO:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    """Carga el modelo y scaler."""
    model_path = os.path.join(BASE_DIR, "models", "modelo_final_riesgo_pediatrico.joblib")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler_riesgo_pediatrico.joblib")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    return model, scaler


@st.cache_data
def load_metrics():
    """Carga las métricas del modelo."""
    path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_risk(model, scaler, edad, genero, peso, pas, fc, col):
    """Predice riesgo cardiovascular."""
    X = np.array([[float(edad), float(genero), float(peso),
                   float(pas), float(fc), float(col)]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_s = scaler.transform(X)
        prob = model.predict_proba(X_s)[0][1]
    coefs = np.abs(model.coef_).flatten()
    total = coefs.sum()
    importancias = {n: round(float(c / total), 4) for n, c in zip(FEATURES_MODELO, coefs)}
    return prob, importancias


# ═══════════════════════════════════════════════════════════════
# ESTILOS CSS PERSONALIZADOS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .main .block-container { max-width: 1200px; padding-top: 1rem; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; }
    .kpi-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e5e7eb;
        border-top: 4px solid #1e5090; text-align: center;
    }
    .kpi-value { font-size: 28px; font-weight: 800; color: #0f2442; }
    .kpi-label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .semaforo-box {
        border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 12px;
    }
    .sem-verde { background: linear-gradient(135deg, #d1fae5, #ecfdf5); border: 2px solid #6ee7b7; }
    .sem-amarillo { background: linear-gradient(135deg, #fef3c7, #fffbeb); border: 2px solid #fcd34d; }
    .sem-rojo { background: linear-gradient(135deg, #fee2e2, #fef2f2); border: 2px solid #fca5a5; }
    .diag-card {
        border-radius: 12px; padding: 16px; margin-top: 8px;
    }
    .footer-text {
        text-align: center; font-size: 11px; color: #9ca3af;
        padding: 16px 0; border-top: 1px solid #e5e7eb; margin-top: 32px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ❤️ ECV Pediátrico")
    st.caption("Sistema de Apoyo Clínico — SEMMA")
    st.divider()
    st.markdown("**Navegación**")
    tab_selection = st.radio(
        "Seleccione una sección:",
        ["📊 Resumen y EDA", "🔬 Segmentación", "📈 Evaluación de Modelos",
         "🩺 XAI y Simulador"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
    <div style="text-align:center; font-size:11px; color:#8bb5d8;">
        <p>© 2026 Ana Díaz · Yeison Ramírez</p>
        <p>Fundación Universitaria de Popayán</p>
        <p>Ingeniería de Sistemas</p>
        <p style="font-size:9px; opacity:0.7; margin-top:4px;">Todos los derechos reservados</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 1: RESUMEN Y EDA
# ═══════════════════════════════════════════════════════════════
def render_tab_eda():
    st.title("Resumen Demográfico y Exploración de Datos")
    st.caption("Análisis exploratorio del dataset pediátrico — Metodología SEMMA")

    df = load_data()
    metrics = load_metrics()
    n_total = len(df)
    n_alto = int(df[TARGET].sum())
    n_bajo = n_total - n_alto

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{n_total:,}</div>
            <div class="kpi-label">Total Pacientes</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card" style="border-top-color: #1a4076;">
            <div class="kpi-value">{metrics.get('edad_promedio', 11.4)} años</div>
            <div class="kpi-label">Edad Promedio</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card" style="border-top-color: #059669;">
            <div class="kpi-value">{metrics.get('peso_promedio', 43.43)} kg</div>
            <div class="kpi-label">Peso Promedio</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card" style="border-top-color: #d97706;">
            <div class="kpi-value">{metrics.get('colesterol_promedio', 159.9)}</div>
            <div class="kpi-label">Colesterol Promedio (mg/dL)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Donut + Detalle
    col_d, col_r = st.columns([1, 1])
    with col_d:
        st.subheader("Distribución de Riesgo CV")
        fig = go.Figure(go.Pie(
            labels=["Sin Riesgo (78%)", "Con Riesgo (22%)"],
            values=[n_bajo, n_alto], hole=0.55,
            marker=dict(colors=["#4682B4", "#CD5C5C"]),
            textinfo="label+percent",
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                          height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("⚖️ Desbalance: 78% sin riesgo vs 22% con riesgo. Se usa `class_weight='balanced'`.")

    with col_r:
        st.subheader("Detalle por Categoría")
        st.metric("Sin Riesgo (Clase 0)", f"{n_bajo:,}", f"{n_bajo/n_total*100:.1f}%")
        st.metric("Con Riesgo (Clase 1)", f"{n_alto:,}", f"{n_alto/n_total*100:.1f}%")

    # Histogramas
    st.subheader("Histogramas Comparativos por Clase")
    var = st.selectbox("Variable:", ["colesterol_mgdl", "edad", "peso_kg",
                                      "frecuencia_cardiaca", "pa_sistolica"],
                       format_func=lambda x: FEATURE_LABELS.get(x, x))
    df_h = df.copy()
    df_h["Clase"] = df_h[TARGET].map({0: "Sin Riesgo", 1: "Con Riesgo"})

    ch, cb = st.columns(2)
    with ch:
        fig_h = px.histogram(df_h, x=var, color="Clase", barmode="overlay", opacity=0.7,
                             color_discrete_sequence=["#4682B4", "#CD5C5C"],
                             labels={var: FEATURE_LABELS.get(var, var)}, template="plotly_white")
        fig_h.update_layout(font_family="Inter", height=350,
                           legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
        st.plotly_chart(fig_h, use_container_width=True)
    with cb:
        fig_b = px.box(df_h, x="Clase", y=var, color="Clase",
                       color_discrete_sequence=["#4682B4", "#CD5C5C"],
                       template="plotly_white", points="outliers")
        fig_b.update_layout(font_family="Inter", height=350, showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    # Correlación
    st.subheader("Matriz de Correlación de Pearson")
    st.caption("Destaca: **Edad-Peso (0.91)** y **Colesterol-Riesgo CV (0.41)**")
    cols_corr = ["edad", "peso_kg", "pa_sistolica", "colesterol_mgdl", "frecuencia_cardiaca", TARGET]
    corr = df[cols_corr].corr()
    labels_c = ["Edad", "Peso", "PAS", "Colesterol", "FC", "Riesgo CV"]
    fig_c = go.Figure(go.Heatmap(
        z=corr.values, x=labels_c, y=labels_c,
        colorscale=[[0, "#2563eb"], [0.5, "#fff"], [1, "#dc2626"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=12),
    ))
    fig_c.update_layout(height=450, template="plotly_white", font_family="Inter",
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_c, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: SEGMENTACIÓN
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def compute_pca_clusters(_df):
    """Computa PCA y K-Means."""
    X = _df[FEATURES_MODELO].values.copy()
    y = _df[TARGET].values.copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init="auto")
    labels = km.fit_predict(X_s)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s)
    return pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        "Cluster": labels.astype(str),
        "Riesgo": y.astype(int),
    }), pca.explained_variance_ratio_ * 100


def render_tab_segmentation():
    st.title("Segmentación Avanzada")
    st.caption("Clustering K-Means + PCA — SEMMA Phase 2B")

    st.info("ℹ️ El clustering es análisis descriptivo con scaler independiente. No constituye data leakage.")

    df = load_data()
    metrics = load_metrics()
    df_pca, var_ratio = compute_pca_clusters(df)
    clustering = metrics.get("clustering", {})
    clusters = clustering.get("clusters", [])

    # PCA scatter
    st.subheader("Visualización PCA en 2D")
    color_by = st.radio("Colorear por:", ["Cluster K-Means", "Riesgo CV Real"], horizontal=True)
    if color_by == "Cluster K-Means":
        color_col, cmap = "Cluster", {"0": "#4682B4", "1": "#CD5C5C", "2": "#2E8B57", "3": "#DAA520"}
    else:
        df_pca["RiesgoLabel"] = df_pca["Riesgo"].map({0: "Sin Riesgo", 1: "Con Riesgo"})
        color_col, cmap = "RiesgoLabel", {"Sin Riesgo": "#4682B4", "Con Riesgo": "#CD5C5C"}

    fig_s = px.scatter(df_pca, x="PC1", y="PC2", color=color_col,
                       color_discrete_map=cmap, opacity=0.5, template="plotly_white")
    fig_s.update_traces(marker_size=4)
    fig_s.update_layout(
        xaxis_title=f"PC1 ({var_ratio[0]:.1f}% varianza)",
        yaxis_title=f"PC2 ({var_ratio[1]:.1f}% varianza)",
        height=500, font_family="Inter",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # Cluster cards
    st.subheader("Resultados del Clustering (K=4)")
    cols = st.columns(4)
    for i, c in enumerate(clusters):
        with cols[i]:
            riesgo_color = "🔴" if c["pct_riesgo"] > 25 else "🟢"
            st.markdown(f"""
            **Cluster {c['id']}** {'⚠️' if c['id'] == 1 else ''}
            - Pacientes: **{c['n_pacientes']:,}** ({c['pct']}%)
            - {riesgo_color} Riesgo: **{c['pct_riesgo']}%**
            - Edad: {c['edad_mean']} años
            - Peso: {c['peso_mean']} kg
            - Col: {c['colesterol_mean']} mg/dL
            - PAS: {c['pa_sistolica_mean']} mmHg
            """)

    # Prevalencia
    fig_p = go.Figure()
    ids = [f"Cluster {c['id']}" for c in clusters]
    pcts = [c["pct_riesgo"] for c in clusters]
    fig_p.add_trace(go.Bar(x=ids, y=pcts, marker_color=["#4682B4", "#CD5C5C", "#2E8B57", "#DAA520"],
                           text=[f"{p}%" for p in pcts], textposition="outside"))
    fig_p.add_hline(y=22.0, line_dash="dash", line_color="#dc2626",
                    annotation_text="Media global (22%)")
    fig_p.update_layout(yaxis_title="% Riesgo CV", height=350, template="plotly_white",
                       font_family="Inter")
    st.plotly_chart(fig_p, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: EVALUACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════
def render_tab_assessment():
    st.title("Evaluación y Comparación de Modelos")
    st.caption("Fase ASSESS — Comparación de los 5 algoritmos SEMMA")

    metrics = load_metrics()
    comparison = metrics.get("model_comparison", [])

    # Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f2442, #1a4076); color: white;
                border-radius: 12px; padding: 20px; display: flex; justify-content: space-around;
                flex-wrap: wrap; gap: 12px; margin-bottom: 24px;">
        <div style="text-align:center;"><div style="font-size:10px; text-transform:uppercase; color:#8bb5d8;">Modelo Ganador</div><div style="font-weight:700;">Logistic Regression</div></div>
        <div style="text-align:center;"><div style="font-size:10px; text-transform:uppercase; color:#8bb5d8;">Prioridad</div><div style="font-weight:700; color:#2E8B57;">Recall</div></div>
        <div style="text-align:center;"><div style="font-size:10px; text-transform:uppercase; color:#8bb5d8;">Recall</div><div style="font-weight:700; color:#2E8B57;">83.18%</div></div>
        <div style="text-align:center;"><div style="font-size:10px; text-transform:uppercase; color:#8bb5d8;">ROC-AUC</div><div style="font-weight:700;">0.889</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.info("⚠️ En pediatría preventiva, minimizar falsos negativos es crítico. Se prioriza el **Recall**.")

    # Tabla
    st.subheader("Métricas — 5 Algoritmos")
    df_comp = pd.DataFrame(comparison)
    st.dataframe(
        df_comp[["modelo", "roc_auc", "recall", "precision", "f1", "accuracy"]].style
        .highlight_max(subset=["recall", "roc_auc"], color="#d1fae5")
        .format({c: "{:.4f}" for c in ["roc_auc", "recall", "precision", "f1", "accuracy"]}),
        use_container_width=True, hide_index=True,
    )

    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    sel_model = st.selectbox("Modelo:", [m["modelo"] for m in comparison])
    cm = [[632, 148], [37, 183]]
    for m in comparison:
        if m["modelo"] == sel_model:
            cm = m.get("cm", cm)
            break
    cm = np.array(cm)
    labels = ["Bajo Riesgo", "Alto Riesgo"]
    ann = [[f"VN\n{cm[0][0]}", f"FP\n{cm[0][1]}"], [f"FN\n{cm[1][0]}", f"VP\n{cm[1][1]}"]]
    fig_cm = go.Figure(go.Heatmap(z=cm, x=labels, y=labels,
                                   colorscale=[[0, "#e8f1f8"], [1, "#1a4076"]],
                                   text=ann, texttemplate="%{text}", textfont=dict(size=16),
                                   showscale=False))
    fig_cm.update_layout(xaxis_title="Predicción", yaxis_title="Real",
                        yaxis=dict(autorange="reversed"), height=350,
                        template="plotly_white", font_family="Inter",
                        title=f"Matriz de Confusión — {sel_model}")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Curvas ROC / PR
    cr, cp = st.columns(2)
    model, scaler = load_model()
    df = load_data()
    X = df[FEATURES_MODELO].fillna(df[FEATURES_MODELO].median()).values
    y = df[TARGET].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]

    with cr:
        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"LR (AUC={auc_val:.3f})",
                                     line=dict(color="#4682B4", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Aleatorio",
                                     line=dict(color="#d1d5db", dash="dash")))
        fig_roc.update_layout(height=400, template="plotly_white", font_family="Inter",
                             xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

    with cp:
        st.subheader("Curva Precision-Recall")
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, name=f"LR (PR-AUC={pr_auc:.3f})",
                                    line=dict(color="#4682B4", width=3)))
        baseline = y_test.mean()
        fig_pr.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], name=f"Baseline ({baseline:.3f})",
                                    line=dict(color="#d1d5db", dash="dash")))
        fig_pr.update_layout(height=400, template="plotly_white", font_family="Inter",
                            xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4: XAI + SIMULADOR
# ═══════════════════════════════════════════════════════════════
def render_tab_xai():
    st.title("Explicabilidad y Simulador Predictivo")
    st.caption("Importancia de variables + calculadora de riesgo cardiovascular")

    metrics = load_metrics()
    model, scaler = load_model()

    # Feature importance
    st.subheader("Importancia de Variables — Modelo Logístico")
    imps = metrics.get("feature_importances", {})
    if imps:
        sorted_f = sorted(imps.items(), key=lambda x: x[1])
        names = [FEATURE_LABELS.get(k, k) for k, _ in sorted_f]
        vals = [v for _, v in sorted_f]
        colors = ["#9CA3AF"] * len(vals)
        colors[-1] = "#4682B4"
        colors[-2] = "#2E8B57"
        colors[-3] = "#DAA520"
        fig_fi = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker_color=colors,
                                   text=[f"{v:.3f}" for v in vals], textposition="outside"))
        fig_fi.update_layout(height=250, template="plotly_white", font_family="Inter",
                            margin=dict(l=10, r=60, t=10, b=30), xaxis_title="Importancia")
        st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # Simulador
    st.subheader("🩺 Calculadora Interactiva de Riesgo")
    st.info("Ajuste los valores del paciente. El modelo devuelve la **probabilidad global de riesgo CV**, "
            "no un diagnóstico específico.")

    col_form, col_result = st.columns([1, 1.5])

    with col_form:
        edad = st.slider("Edad (años)", 6, 17, 11)
        sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
        genero = 1 if sexo == "Masculino" else 0
        peso = st.slider("Peso (kg)", 5, 120, 43)
        pas = st.slider("PA Sistólica (mmHg)", 60, 160, 110)
        fc = st.slider("Frecuencia Cardíaca (lpm)", 40, 200, 85)
        colesterol = st.slider("Colesterol (mg/dL)", 50, 350, 160)

        predict_clicked = st.button("🔍 Evaluar Riesgo", type="primary", use_container_width=True)

    with col_result:
        if predict_clicked:
            prob, importancias = predict_risk(model, scaler, edad, genero, peso, pas, fc, colesterol)
            pct = round(prob * 100, 1)

            # Semáforo
            if prob < 0.3:
                cls, lbl, clr = "sem-verde", "RIESGO BAJO", "#065f46"
            elif prob < 0.7:
                cls, lbl, clr = "sem-amarillo", "RIESGO MODERADO", "#92400e"
            else:
                cls, lbl, clr = "sem-rojo", "RIESGO ALTO", "#991b1b"

            st.markdown(f"""
            <div class="semaforo-box {cls}">
                <div style="font-size:16px; font-weight:700; color:{clr};">{lbl}</div>
                <div style="font-size:36px; font-weight:800; color:{clr};">{pct}%</div>
                <div style="font-size:12px; color:#6b7280;">Probabilidad estimada de riesgo CV</div>
            </div>
            """, unsafe_allow_html=True)

            # Leyenda
            st.caption("📊 0-30% = Bajo · 31-69% = Moderado · ≥70% = Alto")

            # ── Diagnóstico por umbrales clínicos ────────────
            perfiles = _evaluar_clinico(edad, peso, pas, fc, colesterol)

            if perfiles:
                st.markdown("---")
                st.markdown("#### 🏥 Interpretación Clínica")
                for p in perfiles:
                    with st.expander(f"{p['emoji']} {p['titulo']}", expanded=(p == perfiles[0])):
                        st.markdown(f"**{p['subtitulo']}**")
                        st.write(p["descripcion"])
                        st.markdown("**Acciones recomendadas:**")
                        for a in p["acciones"]:
                            st.markdown(f"- {a}")
                st.caption("⚠️ *Perfil sugerido por umbrales clínicos. No constituye diagnóstico médico.*")
            else:
                if prob < 0.3:
                    st.success("✅ Todos los valores en rango normal. Mantener controles de rutina.")

            # Datos del paciente
            st.markdown("---")
            st.markdown("**Valores del paciente:**")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Sexo", sexo)
            dc1.metric("Edad", f"{edad} años")
            dc2.metric("Peso", f"{peso} kg", _peso_label(edad, peso))
            dc2.metric("PAS", f"{pas} mmHg", _pas_label(pas))
            dc3.metric("FC", f"{fc} lpm", _fc_label(fc))
            dc3.metric("Colesterol", f"{colesterol} mg/dL", _col_label(colesterol))
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#9ca3af;">
                <div style="font-size:48px;">🩺</div>
                <p style="font-size:16px; margin-top:12px;">Ajuste los valores del paciente y presione <b>Evaluar Riesgo</b></p>
            </div>
            """, unsafe_allow_html=True)


def _evaluar_clinico(edad, peso, pas, fc, colesterol):
    """Evalúa umbrales clínicos reales."""
    perfiles = []
    edad_int = max(6, min(17, int(edad)))
    p85, p95 = PESO_PERCENTILES.get(edad_int, (50, 60))

    tiene_col_alto = colesterol >= 200
    tiene_pas_alta = pas >= 130
    tiene_peso_alto = peso >= p85

    # Síndrome metabólico (prioridad)
    if sum([tiene_col_alto, tiene_pas_alta, tiene_peso_alto]) >= 2:
        comps = []
        if tiene_col_alto: comps.append(f"Colesterol {colesterol} mg/dL")
        if tiene_pas_alta: comps.append(f"PAS {pas} mmHg")
        if tiene_peso_alto: comps.append(f"Peso {peso} kg (>P85)")
        perfiles.append({
            "emoji": "🔴", "titulo": "Perfil compatible con Síndrome Metabólico",
            "subtitulo": " + ".join(comps),
            "descripcion": f"Se detectan {len(comps)} factores combinados, configurando un perfil compatible con síndrome metabólico pediátrico.",
            "acciones": ["Perfil lipídico completo + glucosa + insulina + HbA1c",
                         "Circunferencia abdominal e índice cintura-talla",
                         "Referencia a nutrición y endocrinología pediátrica",
                         "Plan multidisciplinario con seguimiento trimestral"],
        })

    if colesterol >= 200:
        perfiles.append({
            "emoji": "🟣", "titulo": "Perfil compatible con Dislipidemia",
            "subtitulo": f"Colesterol: {colesterol} mg/dL (≥200 = hipercolesterolemia)",
            "descripcion": f"El colesterol ({colesterol} mg/dL) supera el umbral de 200 mg/dL para población pediátrica, sugiriendo riesgo de aterosclerosis temprana.",
            "acciones": ["Perfil lipídico completo (LDL, HDL, TGs)",
                         "Evaluar antecedentes familiares", "Intervención dietética, reevaluar en 3-6 meses",
                         "Referencia a endocrinología si LDL > 160"],
        })
    elif colesterol >= 170:
        perfiles.append({
            "emoji": "🟡", "titulo": "Colesterol en rango límite alto",
            "subtitulo": f"Colesterol: {colesterol} mg/dL (170-199 = límite)",
            "descripcion": "Valor en rango límite. Vigilancia recomendada.",
            "acciones": ["Repetir en ayunas", "Evaluar dieta", "Seguimiento"],
        })

    if pas >= 130:
        perfiles.append({
            "emoji": "🔴", "titulo": "Perfil compatible con Hipertensión Arterial",
            "subtitulo": f"PAS: {pas} mmHg (≥130 = sospecha HTA)",
            "descripcion": f"La PAS ({pas} mmHg) supera el umbral de 130 mmHg, compatible con hipertensión arterial pediátrica.",
            "acciones": ["Confirmar con 3 mediciones en visitas separadas",
                         "Evaluar según percentiles AAP", "Descartar causas secundarias",
                         "Ecocardiograma si HTA confirmada"],
        })
    elif pas >= 110:
        perfiles.append({
            "emoji": "🟡", "titulo": "Presión arterial elevada",
            "subtitulo": f"PAS: {pas} mmHg (110-129 = elevada)",
            "descripcion": "PAS elevada. Seguimiento recomendado.",
            "acciones": ["Repetir en reposo", "Evaluar sodio y actividad física"],
        })

    if fc > 100:
        perfiles.append({
            "emoji": "🔵", "titulo": "Perfil compatible con Taquicardia",
            "subtitulo": f"FC: {fc} lpm (>100 = taquicardia)",
            "descripcion": f"FC ({fc} lpm) supera el límite normal, puede sugerir desregulación autonómica.",
            "acciones": ["ECG de 12 derivaciones", "Descartar anemia, fiebre, ansiedad",
                         "Holter si síntomas intermitentes", "Referencia a cardiología si persiste"],
        })
    elif fc < 60:
        perfiles.append({
            "emoji": "🔵", "titulo": "Perfil compatible con Bradicardia",
            "subtitulo": f"FC: {fc} lpm (<60 = bradicardia)",
            "descripcion": f"FC ({fc} lpm) por debajo del límite normal.",
            "acciones": ["ECG", "Descartar hipotiroidismo", "Evaluación cardiológica"],
        })

    if peso >= p95:
        perfiles.append({
            "emoji": "🟠", "titulo": "Perfil compatible con Obesidad",
            "subtitulo": f"Peso: {peso} kg (>P95 para {edad_int} años = {p95} kg)",
            "descripcion": f"Peso ({peso} kg) supera P95 para {edad_int} años, indicando obesidad.",
            "acciones": ["IMC y curvas OMS/CDC", "Circunferencia abdominal",
                         "Screening metabólico", "Intervención nutricional"],
        })
    elif peso >= p85:
        perfiles.append({
            "emoji": "🟡", "titulo": "Sobrepeso detectado",
            "subtitulo": f"Peso: {peso} kg (>P85 para {edad_int} años = {p85} kg)",
            "descripcion": "Sobrepeso. Orientación nutricional.",
            "acciones": ["Evaluar IMC", "Orientación nutricional"],
        })

    return perfiles


def _col_label(v):
    if v >= 200: return "⬆ Alto"
    if v >= 170: return "Límite"
    return "Normal"

def _pas_label(v):
    if v >= 130: return "⬆ Alto"
    if v >= 110: return "Elevada"
    return "Normal"

def _fc_label(v):
    if v > 100: return "⬆ Taquicardia"
    if v < 60: return "⬇ Bradicardia"
    return "Normal"

def _peso_label(edad, peso):
    e = max(6, min(17, int(edad)))
    p85, p95 = PESO_PERCENTILES.get(e, (50, 60))
    if peso >= p95: return "⬆ >P95"
    if peso >= p85: return "Sobrepeso"
    return "Normal"


# ═══════════════════════════════════════════════════════════════
# MAIN — ROUTING
# ═══════════════════════════════════════════════════════════════
if tab_selection.startswith("📊"):
    render_tab_eda()
elif tab_selection.startswith("🔬"):
    render_tab_segmentation()
elif tab_selection.startswith("📈"):
    render_tab_assessment()
elif tab_selection.startswith("🩺"):
    render_tab_xai()

# Footer
st.markdown("""
<div class="footer-text">
    © 2026 Ana Díaz · Yeison Ramírez — Fundación Universitaria de Popayán — Ingeniería de Sistemas<br>
    Todos los derechos reservados
</div>
""", unsafe_allow_html=True)
